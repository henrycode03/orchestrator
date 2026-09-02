"""Provider-free contract tests for durable public Task-to-Session targeting."""

from __future__ import annotations

import asyncio

from app.models import (
    Project,
    Session as SessionModel,
    SessionTask,
    Task,
    TaskExecution,
    TaskStatus,
)
from app.services.session.session_lifecycle_service import start_session_lifecycle
from app.services.session.session_runtime_service import queue_task_for_session


def _task_pair(db_session, *, intended_intent="default"):
    project = Project(name="STS1 Targeting Project")
    db_session.add(project)
    db_session.flush()
    historical = Task(
        project_id=project.id,
        title="Historical retryable task",
        status=TaskStatus.FAILED,
    )
    intended = Task(
        project_id=project.id,
        title="Intended new task",
        status=TaskStatus.PENDING,
        intent_mode=intended_intent,
    )
    db_session.add_all([historical, intended])
    db_session.commit()
    db_session.refresh(project)
    db_session.refresh(historical)
    db_session.refresh(intended)
    return project, historical, intended


def test_public_session_create_persists_explicit_task_link(
    authenticated_client, db_session
):
    project, _, intended = _task_pair(db_session)

    response = authenticated_client.post(
        "/api/v1/sessions",
        json={
            "project_id": project.id,
            "name": "Targeted execution session",
            "description": "Run the intended task.",
            "execution_mode": "automatic",
            "task_id": intended.id,
        },
    )

    assert response.status_code == 201
    session_id = response.json()["id"]
    link = (
        db_session.query(SessionTask).filter(SessionTask.session_id == session_id).one()
    )
    assert link.task_id == intended.id
    assert response.json()["task_id"] == intended.id

    get_response = authenticated_client.get(f"/api/v1/sessions/{session_id}")
    assert get_response.status_code == 200
    assert get_response.json()["task_id"] == intended.id

    list_response = authenticated_client.get("/api/v1/sessions")
    listed = next(item for item in list_response.json() if item["id"] == session_id)
    assert listed["task_id"] == intended.id

    project_list_response = authenticated_client.get(
        f"/api/v1/projects/{project.id}/sessions"
    )
    project_listed = next(
        item for item in project_list_response.json() if item["id"] == session_id
    )
    assert project_listed["task_id"] == intended.id


def test_automatic_start_prefers_durable_link_over_project_order(
    db_session, monkeypatch
):
    project, historical, intended = _task_pair(db_session)
    session = SessionModel(
        project_id=project.id,
        name="Targeted execution session",
        description="Run the intended task; task #1 is unrelated historical work.",
        status="pending",
        execution_mode="automatic",
        is_active=False,
    )
    db_session.add(session)
    db_session.flush()
    db_session.add(
        SessionTask(
            session_id=session.id,
            task_id=intended.id,
            status=TaskStatus.PENDING,
        )
    )
    db_session.commit()

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "provider-free"})()

        async def create_session(self, _task_description):
            return "provider-free-session"

    queued = []
    queue_kwargs = []
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.queue_task_for_session",
        lambda **kwargs: queue_kwargs.append(kwargs)
        or queued.append(kwargs["task_id"])
        or {"task_id": kwargs["task_id"]},
    )

    result = asyncio.run(start_session_lifecycle(db_session, session.id))

    assert result["status"] == "started"
    assert queued == [intended.id]
    assert queued != [historical.id]
    assert queue_kwargs[0]["isolated_retry"] is True
    assert db_session.query(TaskExecution).count() == 0


def test_legacy_automatic_start_keeps_project_selection_when_target_omitted(
    db_session, monkeypatch
):
    project, historical, intended = _task_pair(db_session)
    session = SessionModel(
        project_id=project.id,
        name="Legacy automatic session",
        description="Run the next project task.",
        status="pending",
        execution_mode="automatic",
        is_active=False,
    )
    db_session.add(session)
    db_session.commit()

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "provider-free"})()

        async def create_session(self, _task_description):
            return "provider-free-session"

    queued = []
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.queue_task_for_session",
        lambda **kwargs: queued.append(kwargs["task_id"])
        or {"task_id": kwargs["task_id"]},
    )

    asyncio.run(start_session_lifecycle(db_session, session.id))

    assert queued == [historical.id]
    assert queued != [intended.id]


def test_explicit_nonexistent_task_fails_closed_without_creating_session(
    authenticated_client, db_session
):
    project = Project(name="STS1 Missing Target Project")
    db_session.add(project)
    db_session.commit()
    before = db_session.query(SessionModel).count()

    response = authenticated_client.post(
        "/api/v1/sessions",
        json={
            "project_id": project.id,
            "name": "Missing target session",
            "task_id": 999999,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found for this session"
    assert db_session.query(SessionModel).count() == before
    assert db_session.query(SessionTask).count() == 0


def test_explicit_task_from_another_project_fails_closed(
    authenticated_client, db_session
):
    project = Project(name="STS1 Session Project")
    other_project = Project(name="STS1 Other Project")
    db_session.add_all([project, other_project])
    db_session.flush()
    other_task = Task(
        project_id=other_project.id,
        title="Other project's task",
        status=TaskStatus.PENDING,
    )
    db_session.add(other_task)
    db_session.commit()
    before = db_session.query(SessionModel).count()

    response = authenticated_client.post(
        "/api/v1/sessions",
        json={
            "project_id": project.id,
            "name": "Mismatched target session",
            "task_id": other_task.id,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found for this session"
    assert db_session.query(SessionModel).count() == before


def test_explicit_running_task_is_rejected_at_session_create(
    authenticated_client, db_session
):
    project = Project(name="STS1 Running Target Project")
    db_session.add(project)
    db_session.flush()
    running_task = Task(
        project_id=project.id,
        title="Already running task",
        status=TaskStatus.RUNNING,
    )
    db_session.add(running_task)
    db_session.commit()
    before = db_session.query(SessionModel).count()

    response = authenticated_client.post(
        "/api/v1/sessions",
        json={
            "project_id": project.id,
            "name": "Running target session",
            "task_id": running_task.id,
        },
    )

    assert response.status_code == 409
    assert "already running" in response.json()["detail"]
    assert db_session.query(SessionModel).count() == before


def test_text_selector_remains_compatibility_fallback_without_durable_link(
    db_session, monkeypatch
):
    project, historical, intended = _task_pair(db_session)
    session = SessionModel(
        project_id=project.id,
        name="Compatibility session task #2",
        description="Legacy text selector",
        status="pending",
        execution_mode="automatic",
        is_active=False,
    )
    db_session.add(session)
    db_session.commit()

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "provider-free"})()

        async def create_session(self, _task_description):
            return "provider-free-session"

    queued = []
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.queue_task_for_session",
        lambda **kwargs: queued.append(kwargs["task_id"])
        or {"task_id": kwargs["task_id"]},
    )

    asyncio.run(start_session_lifecycle(db_session, session.id))

    assert queued == [intended.id]
    assert queued != [historical.id]


def test_persisted_target_survives_reload_before_start(db_session, monkeypatch):
    project, _, intended = _task_pair(db_session)
    session = SessionModel(
        project_id=project.id,
        name="Reloaded target session",
        description="Target survives a process boundary.",
        status="pending",
        execution_mode="automatic",
        is_active=False,
    )
    db_session.add(session)
    db_session.flush()
    db_session.add(SessionTask(session_id=session.id, task_id=intended.id))
    db_session.commit()
    session_id = session.id
    db_session.expire_all()
    reloaded = db_session.get(SessionModel, session_id)

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "provider-free"})()

        async def create_session(self, _task_description):
            return "provider-free-session"

    queued = []
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.queue_task_for_session",
        lambda **kwargs: queued.append(kwargs["task_id"])
        or {"task_id": kwargs["task_id"]},
    )

    asyncio.run(start_session_lifecycle(db_session, reloaded.id))

    assert queued == [intended.id]


def test_sessiontask_remains_many_to_many_and_has_no_session_task_id_column():
    session_columns = {column.name for column in SessionModel.__table__.columns}
    assert "task_id" not in session_columns
    assert SessionTask.__table__.c.session_id is not None
    assert SessionTask.__table__.c.task_id is not None


def test_targeted_create_only_task_intent_is_unchanged_by_routing(
    db_session, monkeypatch
):
    project, _, intended = _task_pair(
        db_session,
        intended_intent="create_only",
    )
    session = SessionModel(
        project_id=project.id,
        name="Create-only target session",
        description="Route without changing the task contract.",
        status="pending",
        execution_mode="automatic",
    )
    db_session.add(session)
    db_session.flush()
    db_session.add(SessionTask(session_id=session.id, task_id=intended.id))
    db_session.commit()

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "provider-free"})()

        async def create_session(self, _task_description):
            return "provider-free-session"

    queued = []
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.queue_task_for_session",
        lambda **kwargs: queued.append(kwargs["task_id"])
        or {"task_id": kwargs["task_id"]},
    )

    asyncio.run(start_session_lifecycle(db_session, session.id))

    db_session.refresh(intended)
    assert queued == [intended.id]
    assert intended.intent_mode == "create_only"
    assert intended.plan_id is None
    assert db_session.query(TaskExecution).count() == 0


def test_explicit_terminal_task_statuses_follow_existing_queue_eligibility(
    authenticated_client, db_session
):
    project = Project(name="STS1 Terminal Status Project")
    db_session.add(project)
    db_session.flush()
    tasks = [
        Task(
            project_id=project.id,
            title=f"{status.value} task",
            status=status,
        )
        for status in (TaskStatus.FAILED, TaskStatus.DONE, TaskStatus.CANCELLED)
    ]
    db_session.add_all(tasks)
    db_session.commit()

    for index, task in enumerate(tasks):
        response = authenticated_client.post(
            "/api/v1/sessions",
            json={
                "project_id": project.id,
                "name": f"Target {index}",
                "task_id": task.id,
            },
        )
        assert response.status_code == 201
        assert response.json()["task_id"] == task.id


def test_durable_target_does_not_reduce_session_task_cardinality(
    authenticated_client, db_session
):
    project, historical, intended = _task_pair(db_session)
    response = authenticated_client.post(
        "/api/v1/sessions",
        json={
            "project_id": project.id,
            "name": "Multi-task target session",
            "task_id": intended.id,
        },
    )
    assert response.status_code == 201
    session_id = response.json()["id"]
    db_session.add(
        SessionTask(
            session_id=session_id,
            task_id=historical.id,
            status=TaskStatus.PENDING,
        )
    )
    db_session.commit()

    links = (
        db_session.query(SessionTask)
        .filter(SessionTask.session_id == session_id)
        .order_by(SessionTask.id.asc())
        .all()
    )
    assert [link.task_id for link in links] == [intended.id, historical.id]
    get_response = authenticated_client.get(f"/api/v1/sessions/{session_id}")
    assert get_response.json()["task_id"] == historical.id


def test_stale_durable_target_fails_closed_without_automatic_substitution(
    db_session, monkeypatch
):
    project, historical, intended = _task_pair(db_session)
    session = SessionModel(
        project_id=project.id,
        name="Stale target session",
        description="The durable target is invalid.",
        status="pending",
        execution_mode="automatic",
    )
    db_session.add(session)
    db_session.flush()
    db_session.add(SessionTask(session_id=session.id, task_id=999999))
    db_session.commit()

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "provider-free"})()

        async def create_session(self, _task_description):
            return "provider-free-session"

    queued = []
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.queue_task_for_session",
        lambda **kwargs: queued.append(kwargs["task_id"])
        or {"task_id": kwargs["task_id"]},
    )

    try:
        asyncio.run(start_session_lifecycle(db_session, session.id))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 409
        assert getattr(exc, "detail", None) == "session_task_target_invalid"
    else:
        raise AssertionError("stale durable target was silently substituted")

    assert queued == []
    assert historical.status == TaskStatus.FAILED
    assert intended.status == TaskStatus.PENDING


def test_completed_target_link_allows_existing_automatic_continuation(
    db_session, monkeypatch
):
    project = Project(name="STS1 Continuation Project")
    db_session.add(project)
    db_session.flush()
    completed = Task(
        project_id=project.id,
        title="Completed target",
        status=TaskStatus.DONE,
    )
    next_task = Task(
        project_id=project.id,
        title="Next automatic task",
        status=TaskStatus.PENDING,
    )
    session = SessionModel(
        project_id=project.id,
        name="Continuation session",
        description="Continue project work.",
        status="stopped",
        execution_mode="automatic",
        is_active=False,
    )
    db_session.add_all([completed, next_task, session])
    db_session.flush()
    db_session.add(
        SessionTask(
            session_id=session.id,
            task_id=completed.id,
            status=TaskStatus.DONE,
        )
    )
    db_session.commit()

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "provider-free"})()

        async def create_session(self, _task_description):
            return "provider-free-session"

    queued = []
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.queue_task_for_session",
        lambda **kwargs: queued.append(kwargs["task_id"])
        or {"task_id": kwargs["task_id"]},
    )

    asyncio.run(start_session_lifecycle(db_session, session.id))

    assert queued == [next_task.id]
    assert queued != [completed.id]


def test_actual_queue_persists_exact_task_execution_identity(
    db_session, monkeypatch, tmp_path
):
    project, historical, intended = _task_pair(db_session)
    project.workspace_path = str(tmp_path / "project-workspace")
    session = SessionModel(
        project_id=project.id,
        name="Exact execution identity session",
        description="Queue only the intended task.",
        status="pending",
        execution_mode="automatic",
        is_active=False,
        instance_id="sts1-instance",
    )
    db_session.add(session)
    db_session.flush()
    db_session.add(SessionTask(session_id=session.id, task_id=intended.id))
    db_session.commit()

    captured = {}

    class _FakeDelayResult:
        id = "sts1-queued"

    class _FakeWorkerTask:
        @staticmethod
        def delay(**kwargs):
            captured.update(kwargs)
            return _FakeDelayResult()

    monkeypatch.setattr("app.tasks.worker.execute_orchestration_task", _FakeWorkerTask)
    monkeypatch.setattr(
        "app.services.session.session_runtime_service.ensure_task_workspace",
        lambda *args, **kwargs: {
            "workspace_path": str(tmp_path / "task-workspace"),
            "task_subfolder": "task-intended",
            "stored_task_subfolder": "task-intended",
            "workspace_scope": "isolated_task_workspace",
        },
    )
    monkeypatch.setattr(
        "app.services.session.session_runtime_service.append_orchestration_event",
        lambda **kwargs: {"event_id": "sts1-event"},
    )
    monkeypatch.setattr(
        "app.services.session.session_runtime_service._maybe_compact_checkpoint_before_dispatch",
        lambda *args, **kwargs: None,
    )

    result = queue_task_for_session(
        db_session,
        session,
        intended.id,
        isolated_retry=True,
    )

    execution = (
        db_session.query(TaskExecution)
        .filter(TaskExecution.session_id == session.id)
        .one()
    )
    assert result["task_id"] == intended.id
    assert result["task_execution_id"] == execution.id
    assert captured["task_id"] == intended.id
    assert captured["task_id"] != historical.id
    assert execution.task_id == intended.id
    assert execution.session_id == session.id

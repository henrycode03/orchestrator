"""Regression tests for session lifecycle boundary conditions.

Covers start/stop/pause/resume edge cases that have historically caused
silent failures or confusing error responses.
"""

from __future__ import annotations

import asyncio

import pytest

from app.models import Project, Session as SessionModel, Task, TaskStatus
from app.services.session.session_lifecycle_service import (
    pause_session_lifecycle,
    resume_session_lifecycle,
    start_session_lifecycle,
    stop_session_lifecycle,
)
from fastapi import HTTPException


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_project(db):
    project = Project(name="LC Regression", workspace_path="/tmp/lc_test")
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def _make_session(
    db, project, *, status="stopped", is_active=False, execution_mode="manual"
):
    session = SessionModel(
        project_id=project.id,
        name="Test Session",
        description="test",
        status=status,
        is_active=is_active,
        execution_mode=execution_mode,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _make_task(db, project, *, status=TaskStatus.PENDING):
    task = Task(
        project_id=project.id,
        title="Test task",
        description="do something",
        status=status,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


# ── start boundary conditions ─────────────────────────────────────────────────


def test_start_already_running_session_returns_400(db_session, monkeypatch):
    project = _make_project(db_session)
    session = _make_session(db_session, project, status="running", is_active=True)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(start_session_lifecycle(db_session, session.id))

    assert exc_info.value.status_code == 400
    assert "already" in exc_info.value.detail.lower()


def test_start_already_paused_session_returns_400(db_session):
    project = _make_project(db_session)
    session = _make_session(db_session, project, status="paused", is_active=True)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(start_session_lifecycle(db_session, session.id))

    assert exc_info.value.status_code == 400


def test_start_nonexistent_session_returns_404(db_session):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(start_session_lifecycle(db_session, 99999))

    assert exc_info.value.status_code == 404


def test_start_stuck_pending_session_resets_and_proceeds(db_session, monkeypatch):
    """A session stuck in pending+is_active should be reset then attempted."""
    project = _make_project(db_session)
    session = _make_session(db_session, project, status="pending", is_active=True)

    _make_task(db_session, project, status=TaskStatus.PENDING)

    call_log = []

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "local_openclaw"})()

        async def create_session(self, task_description):
            call_log.append("create_session")
            return "fake-key"

    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *a, **kw: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.queue_task_for_session",
        lambda **kw: {"task_id": kw["task_id"]},
    )

    result = asyncio.run(start_session_lifecycle(db_session, session.id))

    assert result["status"] == "started"
    db_session.refresh(session)
    assert session.status == "running"


# ── stop boundary conditions ──────────────────────────────────────────────────


def test_stop_already_stopped_session_returns_400(db_session):
    project = _make_project(db_session)
    session = _make_session(db_session, project, status="stopped")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(stop_session_lifecycle(db_session, session.id))

    assert exc_info.value.status_code == 400


def test_stop_nonexistent_session_returns_404(db_session):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(stop_session_lifecycle(db_session, 99999))

    assert exc_info.value.status_code == 404


def test_stop_running_session_sets_status_stopped(db_session, monkeypatch):
    project = _make_project(db_session)
    session = _make_session(db_session, project, status="running", is_active=True)

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "local_openclaw"})()

        async def stop_session(self):
            pass

    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *a, **kw: _FakeRuntime(),
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.revoke_session_celery_tasks",
        lambda db, session_id, terminate=False: [],
    )
    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.CheckpointService",
        type(
            "FakeCS",
            (),
            {
                "__init__": lambda self, db: None,
                "load_checkpoint": lambda self, sid: (_ for _ in ()).throw(
                    Exception("no checkpoint")
                ),
            },
        ),
    )

    result = asyncio.run(stop_session_lifecycle(db_session, session.id))

    assert result["status"] == "stopped"
    db_session.refresh(session)
    assert session.status == "stopped"
    assert not session.is_active


# ── pause boundary conditions ─────────────────────────────────────────────────


def test_pause_stopped_session_returns_400(db_session):
    project = _make_project(db_session)
    session = _make_session(db_session, project, status="stopped")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pause_session_lifecycle(db_session, session.id))

    assert exc_info.value.status_code == 400


def test_pause_nonexistent_session_returns_404(db_session):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pause_session_lifecycle(db_session, 99999))

    assert exc_info.value.status_code == 404


# ── manual mode start ─────────────────────────────────────────────────────────


def test_start_manual_session_with_no_tasks_starts_without_queuing(
    db_session, monkeypatch
):
    """Manual sessions should reach running status even with no pending tasks."""
    project = _make_project(db_session)
    session = _make_session(
        db_session, project, status="stopped", execution_mode="manual"
    )

    class _FakeRuntime:
        backend_descriptor = type("D", (), {"name": "local_openclaw"})()

        async def create_session(self, task_description):
            return "fake-key"

    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.create_agent_runtime",
        lambda *a, **kw: _FakeRuntime(),
    )

    result = asyncio.run(start_session_lifecycle(db_session, session.id))

    assert result["status"] == "started"
    db_session.refresh(session)
    assert session.status == "running"


def test_resume_session_rehydrates_session_task_and_queues_requested_checkpoint(
    db_session, monkeypatch
):
    project = _make_project(db_session)
    session = _make_session(db_session, project, status="paused", is_active=True)
    task = _make_task(db_session, project, status=TaskStatus.PENDING)

    captured = {}

    class _FakeCheckpointService:
        def __init__(self, db):
            self.db = db

        def load_resume_checkpoint(self, session_id, checkpoint_name=None):
            captured["requested_checkpoint_name"] = checkpoint_name
            return {
                "_requested_checkpoint_name": checkpoint_name,
                "_resolved_checkpoint_name": checkpoint_name or "paused_latest",
                "checkpoint_name": checkpoint_name or "paused_latest",
                "context": {
                    "task_id": task.id,
                    "task_description": "resume from checkpoint",
                },
                "orchestration_state": {},
                "step_results": [],
            }

        def _checkpoint_restore_fidelity(self, data):
            return {
                "score": 35,
                "status": "low",
                "summary": "Checkpoint replay is fragile; important state is missing",
                "present_signals": ["task id", "task description"],
                "warnings": ["missing workspace path", "missing execution plan"],
            }

    class _FakeDelayResult:
        id = "celery-resume-1"

    class _FakeWorkerTask:
        @staticmethod
        def delay(**kwargs):
            captured["delay_kwargs"] = kwargs
            return _FakeDelayResult()

    monkeypatch.setattr(
        "app.services.session.session_lifecycle_service.CheckpointService",
        _FakeCheckpointService,
    )
    monkeypatch.setattr(
        "app.tasks.worker.execute_orchestration_task",
        _FakeWorkerTask,
    )

    result = asyncio.run(
        resume_session_lifecycle(
            db_session,
            session.id,
            checkpoint_name="paused_20260424_034703",
        )
    )

    db_session.refresh(session)
    db_session.refresh(task)

    assert result["status"] == "resumed"
    assert result["restore_fidelity"]["status"] == "low"
    assert captured["requested_checkpoint_name"] == "paused_20260424_034703"
    assert (
        captured["delay_kwargs"]["resume_checkpoint_name"] == "paused_20260424_034703"
    )
    assert session.status == "running"
    assert session.is_active is True
    assert task.status == TaskStatus.RUNNING

    session_task = (
        db_session.query(SessionModel)
        .filter(SessionModel.id == session.id)
        .first()
        .tasks[0]
    )
    assert session_task.task_id == task.id
    assert session_task.status == TaskStatus.RUNNING

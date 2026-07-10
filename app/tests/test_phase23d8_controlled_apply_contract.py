"""Phase 23D-8 — Controlled Apply Contract Completion (Finding 3 regression).

Finding 3 (phase23d7-runtime-architecture-burn-in.md): when no Runtime
Workspace sandbox was allocated for a canonical-baseline dispatch, change-set
capture must never fall back to diffing/copying the live Project Workspace
(risking `.env`, the live database, etc. landing in a change-set artifact).
It must fail closed instead, with an explicit recorded reason.

``ChangesetService.record_task_execution_change_set_unavailable`` is the
fail-closed counterpart to ``persist_task_execution_change_set`` that
worker.py's dispatch `finally` block now calls in that situation. These
tests exercise it directly: it must never read or copy anything from any
live directory, and must record a disposition callers (change-set read/
accept/reject endpoints) can distinguish from a real capture.
"""

from __future__ import annotations

from app.models import (
    LogEntry,
    Project,
    Session as SessionModel,
    Task,
    TaskExecution,
    TaskExecutionChangeSet,
)
from app.services.workspace.changeset_service import ChangesetService


def _seed_task(db_session, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    # Files that must never be touched/read/copied by the fail-closed path.
    (project_dir / ".env").write_text("SECRET=leak-me-not\n", encoding="utf-8")
    (project_dir / "orchestrator.db").write_text("binary-db-stand-in", encoding="utf-8")
    (project_dir / "dump.rdb").write_text("redis-dump-stand-in", encoding="utf-8")

    project = Project(name="Phase 23D-8", workspace_path=str(project_dir))
    db_session.add(project)
    db_session.flush()
    session = SessionModel(
        project_id=project.id,
        name="Phase 23D-8 Session",
        status="running",
        is_active=True,
        execution_mode="manual",
    )
    task = Task(project_id=project.id, title="Runtime allocation failure task")
    db_session.add_all([session, task])
    db_session.flush()
    execution = TaskExecution(
        session_id=session.id, task_id=task.id, attempt_number=1, started_at=None
    )
    db_session.add(execution)
    db_session.commit()
    return project, task, execution, project_dir


def test_record_change_set_unavailable_never_touches_project_workspace(
    db_session, tmp_path
):
    project, task, execution, project_dir = _seed_task(db_session, tmp_path)
    service = ChangesetService(db_session)

    record = service.record_task_execution_change_set_unavailable(
        project,
        task,
        session_id=None,
        task_execution_id=execution.id,
        snapshot_key="task-1-execution-1-pre-run",
        reason="runtime_not_allocated",
    )

    assert record.disposition == "unavailable"
    assert record.disposition_reason == "runtime_not_allocated"
    assert record.status == "runtime_not_allocated"
    assert record.target_path is None
    assert record.snapshot_path is None
    assert record.snapshot_exists is False
    assert record.added_files == []
    assert record.modified_files == []
    assert record.deleted_files == []

    # Fail-closed: no artifact directory was ever created for this
    # execution, and the live Project Workspace secrets/db files are
    # untouched (still present, unread, uncopied).
    artifact_root = project_dir / ".agent" / "change-sets" / str(execution.id)
    assert not artifact_root.exists()
    assert (project_dir / ".env").read_text(encoding="utf-8") == "SECRET=leak-me-not\n"

    stored = (
        db_session.query(TaskExecutionChangeSet)
        .filter(TaskExecutionChangeSet.task_execution_id == execution.id)
        .one()
    )
    assert stored.disposition == "unavailable"
    assert stored.disposition_reason == "runtime_not_allocated"

    log_entry = (
        db_session.query(LogEntry)
        .filter(LogEntry.task_execution_id == execution.id)
        .filter(LogEntry.level == "WARNING")
        .one()
    )
    assert "runtime_not_allocated" in log_entry.log_metadata


def test_record_change_set_unavailable_returned_by_lookup(db_session, tmp_path):
    project, task, execution, _project_dir = _seed_task(db_session, tmp_path)
    service = ChangesetService(db_session)
    service.record_task_execution_change_set_unavailable(
        project,
        task,
        session_id=None,
        task_execution_id=execution.id,
        snapshot_key="task-1-execution-1-pre-run",
        reason="snapshot_missing",
    )

    change_set = service.get_task_execution_change_set(task_execution_id=execution.id)
    assert change_set is not None
    assert change_set["status"] == "snapshot_missing"
    assert change_set["disposition"] == "unavailable"
    assert change_set["added_files"] == []
    assert change_set["changed_count"] == 0

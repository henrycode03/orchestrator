from __future__ import annotations

import json

from app.models import Project, Session as SessionModel, Task, TaskStatus
from app.services.orchestration.persistence import (
    append_orchestration_event,
    record_validation_verdict,
    save_orchestration_checkpoint,
)
from app.services.orchestration.types import ValidationVerdict
from app.services.prompt_templates import OrchestrationState
from app.services.tool_tracking_service import ToolTrackingService


def test_append_orchestration_event_writes_append_only_jsonl(tmp_path):
    project_dir = tmp_path / "journal-project"
    project_dir.mkdir()

    append_orchestration_event(
        project_dir=project_dir,
        session_id=7,
        task_id=13,
        event_type="phase_started",
        details={"phase": "planning"},
    )
    append_orchestration_event(
        project_dir=project_dir,
        session_id=7,
        task_id=13,
        event_type="phase_finished",
        details={"phase": "planning", "status": "accepted"},
    )

    log_path = project_dir / ".openclaw" / "events" / "session_7_task_13.jsonl"
    lines = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [line["event_type"] for line in lines] == [
        "phase_started",
        "phase_finished",
    ]
    assert lines[0]["details"]["phase"] == "planning"


def test_validation_verdict_also_persists_event(db_session, tmp_path):
    project_dir = tmp_path / "validation-journal-project"
    project_dir.mkdir()
    state = OrchestrationState(
        session_id="9",
        task_description="Validate plan",
        project_name="Validation Journal",
        task_id=5,
    )
    state._project_dir_override = str(project_dir)

    verdict = ValidationVerdict(
        stage="plan",
        status="warning",
        profile="implementation",
        reasons=["Naming mismatch"],
    )

    record_validation_verdict(
        db_session,
        session_id=9,
        task_id=5,
        orchestration_state=state,
        verdict=verdict,
    )

    log_path = project_dir / ".openclaw" / "events" / "session_9_task_5.jsonl"
    lines = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert lines[-1]["event_type"] == "validation_result"
    assert lines[-1]["details"]["stage"] == "plan"
    assert lines[-1]["details"]["status"] == "warning"


def test_checkpoint_save_also_persists_checkpoint_saved_event(db_session, tmp_path):
    project_dir = tmp_path / "checkpoint-journal-project"
    project_dir.mkdir()
    state = OrchestrationState(
        session_id="11",
        task_description="Checkpoint event",
        project_name="Checkpoint Journal",
        task_id=8,
    )
    state._project_dir_override = str(project_dir)

    save_orchestration_checkpoint(
        db_session,
        session_id=11,
        task_id=8,
        prompt="Checkpoint event prompt",
        orchestration_state=state,
        checkpoint_name="autosave_latest",
    )

    log_path = project_dir / ".openclaw" / "events" / "session_11_task_8.jsonl"
    lines = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert lines[-1]["event_type"] == "checkpoint_saved"
    assert lines[-1]["details"]["checkpoint_name"] == "autosave_latest"


def test_tool_tracking_also_persists_tool_events(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.workspace.project_isolation_service.get_effective_workspace_root",
        lambda db=None: tmp_path,
    )

    project = Project(name="Tool Events", workspace_path="tool-events")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    task = Task(
        project_id=project.id,
        title="Track tool events",
        status=TaskStatus.RUNNING,
        task_subfolder="task-21",
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    session = SessionModel(
        project_id=project.id,
        name="Tool Session",
        description="track tools",
        status="running",
        is_active=True,
        execution_mode="manual",
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)

    tool_service = ToolTrackingService(db_session)
    tool_service.track(
        execution_id="exec-1",
        tool_name="rg",
        params={"pattern": "TODO"},
        result={"matches": 3},
        success=False,
        session_id=session.id,
        task_id=task.id,
        error_message="rg failed",
    )

    log_path = (
        tmp_path
        / "tool-events"
        / "task-21"
        / ".openclaw"
        / "events"
        / f"session_{session.id}_task_{task.id}.jsonl"
    )
    lines = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert lines[0]["event_type"] == "tool_invoked"
    assert lines[-1]["event_type"] == "tool_failed"
    assert lines[-1]["details"]["tool_name"] == "rg"

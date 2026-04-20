from __future__ import annotations

import json

from app.services.orchestration.persistence import (
    append_orchestration_event,
    record_validation_verdict,
)
from app.services.orchestration.types import ValidationVerdict
from app.services.prompt_templates import OrchestrationState


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

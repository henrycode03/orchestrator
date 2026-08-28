from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.models import LogEntry
from app.services.orchestration.coordinators.completion_coordinator import (
    _publication_validation_log_metadata,
    _resolve_change_set_id,
)
from app.services.orchestration.state.persistence import record_live_log
from app.services.orchestration.validation.accepted_path_authority import (
    accepted_plan_identity,
)
from app.services.orchestration.validation.candidate_checks import (
    candidate_delta_identity,
)
from app.services.orchestration.validation.path_authority import (
    AcceptedPathAuthority,
    GrantClass,
    GrantProvenance,
    PathGrant,
    declare,
)
from app.services.orchestration.validation.validator import ValidatorService


def _plan() -> list[dict]:
    return [
        {
            "step_number": 1,
            "description": "Replace the grounded guide",
            "commands": ["test -f guide.md"],
            "verification": "test -f guide.md",
            "expected_files": ["guide.md"],
            "ops": [{"op": "replace_in_file", "path": "guide.md"}],
        }
    ]


def _authority(runtime: Path) -> AcceptedPathAuthority:
    return AcceptedPathAuthority.create(
        accepted_plan_identity=accepted_plan_identity(_plan()),
        workspace_identity=str(runtime.resolve()),
        maximum_scope_digest="0" * 64,
        grants=[
            PathGrant(
                path=declare("guide.md"),
                grant_class=GrantClass.EXISTING_MUTABLE,
                provenance=GrantProvenance.ACCEPTED_PLAN,
                baseline_content_hash="0" * 64,
            )
        ],
    )


def _publish_verdict(runtime: Path, baseline: Path, validated: str):
    change_set = {
        "target_path": str(runtime),
        "added_files": [],
        "modified_files": ["guide.md"],
        "deleted_files": [],
    }
    return ValidatorService.validate_baseline_publish(
        validation_profile="mutation",
        baseline_path=str(baseline),
        baseline_file_count=1,
        missing_task_expected_files=[],
        missing_prior_expected_files=[],
        candidate_change_set=change_set,
        accepted_path_authority=_authority(runtime),
        require_accepted_path_authority=True,
        validated_candidate_identity=validated,
    )


def test_publication_identity_mismatch_rejects_and_same_identity_accepts(tmp_path):
    baseline = tmp_path / "baseline"
    runtime = tmp_path / "runtime"
    baseline.mkdir()
    runtime.mkdir()
    (baseline / "guide.md").write_text("before\n", encoding="utf-8")
    (runtime / "guide.md").write_text("after\n", encoding="utf-8")
    change_set = {
        "target_path": str(runtime),
        "added_files": [],
        "modified_files": ["guide.md"],
        "deleted_files": [],
    }
    validated = candidate_delta_identity(change_set, project_dir=runtime)

    same = _publish_verdict(runtime, baseline, validated)
    assert same.accepted
    assert "publication_candidate_identity" not in same.details

    (runtime / "guide.md").write_text("different after\n", encoding="utf-8")
    mismatch = _publish_verdict(runtime, baseline, validated)
    assert mismatch.rejected
    assert mismatch.details["publication_candidate_identity"] == {
        "validated": validated,
        "observed": candidate_delta_identity(change_set),
    }


def test_publication_identity_diagnostic_is_retained_without_file_contents(
    db_session, tmp_path
):
    baseline = tmp_path / "baseline"
    runtime = tmp_path / "runtime"
    baseline.mkdir()
    runtime.mkdir()
    (baseline / "guide.md").write_text("before\n", encoding="utf-8")
    (runtime / "guide.md").write_text("after\n", encoding="utf-8")
    change_set = {
        "target_path": str(runtime),
        "added_files": [],
        "modified_files": ["guide.md"],
        "deleted_files": [],
    }
    validated = candidate_delta_identity(change_set, project_dir=runtime)
    (runtime / "guide.md").write_text("different after\n", encoding="utf-8")
    verdict = _publish_verdict(runtime, baseline, validated)

    metadata = _publication_validation_log_metadata(
        verdict,
        task_execution_id=314,
        change_set_id=_resolve_change_set_id(
            SimpleNamespace(
                get_task_execution_change_set=lambda **_: {"change_set_id": 243}
            ),
            change_set,
            314,
        ),
        preflight=True,
    )
    record_live_log(
        db_session,
        session_id=175,
        task_id=228,
        level="ERROR",
        message="[ORCHESTRATION] Baseline publish preflight failed validation",
        metadata=metadata,
        task_execution_id=314,
    )

    entry = (
        db_session.query(LogEntry)
        .filter(LogEntry.task_execution_id == 314)
        .order_by(LogEntry.id.desc())
        .first()
    )
    retained = json.loads(entry.log_metadata)
    assert retained["phase"] == "baseline_publish"
    assert retained["validation_status"] == "rejected"
    assert retained["reasons"] == [
        "Publication candidate identity does not match the validated candidate"
    ]
    assert retained["task_execution_id"] == 314
    assert retained["change_set_id"] == 243
    assert retained["publication_candidate_identity"] == {
        "validated": validated,
        "observed": candidate_delta_identity(change_set),
    }
    serialized = json.dumps(retained)
    assert "different after" not in serialized
    assert "after\n" not in serialized

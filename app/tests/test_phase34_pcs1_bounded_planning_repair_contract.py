"""PHASE34-PCS1 bounded planning repair contract tests.

Provider-free proofs that a narrow Plan repair may only change the invalid
portion of a Plan: replace operations that were already valid keep their
operation mode, and everything that cannot be preserved deterministically
still fails closed inside the existing arbitration fence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.orchestration.operations.file_ops_contract import (
    ReplaceOperationMode,
    classify_replace_operation,
    preserve_replace_operation_modes,
    replace_mode_transitions,
)
from app.services.orchestration.operations.source_region_identity import (
    SourceRegionIdentity,
)
from app.services.orchestration.phases.planning_repair_arbitration_control import (
    arbitrate_planning_repair_candidate,
    preserve_repair_replace_operation_modes,
)
from app.services.orchestration.phases.planning_support import _PlanningRetryState
from app.services.orchestration.planning.planner import (
    PlanningRepairOutputContractViolation,
)
from app.services.orchestration.planning.source_materialization import (
    current_source_version_identity,
    materialize_planner_source_context,
)
from app.services.orchestration.validation.accepted_path_authority import (
    accepted_plan_identity,
    plan_identity_text,
)
from app.services.orchestration.validation.validator import ValidatorService

FORMATTER_PATH = "src/greeting/formatting.py"
TEST_PATH = "tests/test_formatting.py"
FORMATTER_SOURCE = "def normalize_name(value):\n" '    return " ".join(value.split())\n'


def _seed_case_b_workspace(root: Path) -> None:
    """Seed the PBC1 Case B ProductRoot shape: one existing formatter."""

    (root / "src" / "greeting").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / FORMATTER_PATH).write_text(FORMATTER_SOURCE, encoding="utf-8")


def _selector(root: Path, path: str) -> dict:
    source = (root / path).read_bytes()
    return SourceRegionIdentity.from_region(
        canonical_path=path,
        expected_source_version=current_source_version_identity(root / path),
        start_byte=0,
        end_byte=len(source),
        selected_region_sha256=hashlib.sha256(source).hexdigest(),
    ).to_dict()


def _semantic_operation(root: Path, path: str = FORMATTER_PATH) -> dict:
    return {
        "op": "replace_in_file",
        "path": path,
        "selector": _selector(root, path),
        "new": (
            "def normalize_name(value):\n"
            '    return " ".join(value.split()).title()\n'
        ),
    }


def _legacy_operation(path: str = FORMATTER_PATH) -> dict:
    return {
        "op": "replace_in_file",
        "path": path,
        "old": '    return " ".join(value.split())\n',
        "new": '    return " ".join(value.split()).title()\n',
    }


def _primary_step(operation: dict) -> dict:
    return {
        "step_number": 1,
        "description": "Normalize customer-facing names",
        "commands": [],
        "verification": f"test -f {FORMATTER_PATH}",
        "rollback": None,
        "expected_files": [FORMATTER_PATH],
        "ops": [operation],
    }


def _test_target_step() -> dict:
    return {
        "step_number": 2,
        "description": "Add regression coverage for the formatter",
        "commands": ["python -m pytest tests/test_formatting.py"],
        "verification": "python -m pytest tests/test_formatting.py",
        "rollback": None,
        "expected_files": [TEST_PATH],
        "ops": [
            {
                "op": "write_file",
                "path": TEST_PATH,
                "content": (
                    "from src.greeting.formatting import normalize_name\n"
                    "\n"
                    "def test_repeated_spaces():\n"
                    '    assert normalize_name("  ada  lovelace ") == "Ada Lovelace"\n'
                    "\n"
                    "def test_blank_input():\n"
                    '    assert normalize_name("   ") == ""\n'
                ),
            }
        ],
    }


def _ctx(root: Path, plan):
    """A context carrying no runtime service, so any provider use would fail."""

    return SimpleNamespace(
        orchestration_state=SimpleNamespace(
            project_dir=root,
            plan=plan,
            phase_history=[],
        ),
        prompt="normalize customer-facing names",
        task=None,
        logger=logging.getLogger("pcs1-test"),
        emit_live=lambda *_args, **_kwargs: None,
    )


def _arbitrate(ctx, previous_plan):
    return arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=_PlanningRetryState(),
        previous_plan=previous_plan,
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text=json.dumps(ctx.orchestration_state.plan),
        planning_timeout_seconds=1,
        prompt_profile=None,
        repair_planning_output=lambda **_kwargs: None,
    )


# 1 — a valid initial Plan passes through the seam unchanged.
def test_valid_initial_plan_is_untouched_by_the_preservation_seam(tmp_path):
    _seed_case_b_workspace(tmp_path)
    plan = [_primary_step(_semantic_operation(tmp_path)), _test_target_step()]
    original = copy.deepcopy(plan)
    ctx = _ctx(tmp_path, plan)

    result = preserve_repair_replace_operation_modes(ctx=ctx, previous_plan=plan)

    assert result == {"preserved": [], "unpreserved": []}
    assert ctx.orchestration_state.plan is plan
    assert plan == original


# 2 + 3 + 4 — the PBC1 Case B regression: repairing the missing secondary test
# target must not downgrade the already-valid primary semantic replace.
def test_case_b_missing_test_target_repair_preserves_primary_semantic_replace(
    tmp_path,
):
    _seed_case_b_workspace(tmp_path)
    semantic_operation = _semantic_operation(tmp_path)
    previous_plan = [_primary_step(semantic_operation)]
    repaired_plan = [_primary_step(_legacy_operation()), _test_target_step()]
    ctx = _ctx(tmp_path, repaired_plan)

    assert replace_mode_transitions(previous_plan, repaired_plan)

    result = preserve_repair_replace_operation_modes(
        ctx=ctx, previous_plan=previous_plan
    )

    assert result["unpreserved"] == []
    assert result["preserved"] == [
        {
            "step_number": 1,
            "operation_index": 1,
            "from": "SEMANTIC_REPLACE",
            "to": "LEGACY_REPLACE",
        }
    ]
    preserved_plan = ctx.orchestration_state.plan
    # The already-valid primary target and its operation mode survived.
    assert preserved_plan[0]["ops"][0] == semantic_operation
    assert (
        classify_replace_operation(preserved_plan[0]["ops"][0])
        is ReplaceOperationMode.SEMANTIC_REPLACE
    )
    assert preserved_plan[0]["expected_files"] == [FORMATTER_PATH]
    # Only the invalid portion — the missing test target — was repaired.
    assert preserved_plan[1] == _test_target_step()
    # The arbitration fence now has nothing left to reject.
    assert replace_mode_transitions(previous_plan, preserved_plan) == ()
    assert _arbitrate(ctx, previous_plan)["action"] == "none"


# 2 + 3 + 4 through the production path: arbitration itself owns the seam and
# hands the restored Plan back through the existing replace contract.
def test_case_b_repair_is_preserved_by_arbitration_itself(tmp_path):
    _seed_case_b_workspace(tmp_path)
    semantic_operation = _semantic_operation(tmp_path)
    previous_plan = [_primary_step(semantic_operation)]
    repaired_plan = [_primary_step(_legacy_operation()), _test_target_step()]
    ctx = _ctx(tmp_path, repaired_plan)

    result = _arbitrate(ctx, previous_plan)

    assert result["action"] == "replace"
    assert result["plan"] is ctx.orchestration_state.plan
    assert result["plan"][0]["ops"][0] == semantic_operation
    assert result["plan"][1] == _test_target_step()
    assert replace_mode_transitions(previous_plan, result["plan"]) == ()


# 3 — an operation mode that was never invalid is not rewritten when the repair
# changes only the operation content.
def test_repair_content_change_keeps_operation_mode(tmp_path):
    _seed_case_b_workspace(tmp_path)
    previous_plan = [_primary_step(_semantic_operation(tmp_path))]
    repaired_operation = _semantic_operation(tmp_path)
    repaired_operation["new"] = "def normalize_name(value):\n    return value\n"
    repaired_plan = [_primary_step(repaired_operation)]
    ctx = _ctx(tmp_path, repaired_plan)

    result = preserve_repair_replace_operation_modes(
        ctx=ctx, previous_plan=previous_plan
    )

    assert result == {"preserved": [], "unpreserved": []}
    assert ctx.orchestration_state.plan[0]["ops"][0] == repaired_operation


# 4 — a silent SEMANTIC_REPLACE -> LEGACY_REPLACE downgrade never reaches the
# arbitrated Plan, and the fence itself is unchanged for anything else.
def test_legacy_to_semantic_upgrade_still_fails_closed(tmp_path):
    _seed_case_b_workspace(tmp_path)
    previous_plan = [_primary_step(_legacy_operation())]
    repaired_plan = [_primary_step(_semantic_operation(tmp_path))]
    ctx = _ctx(tmp_path, repaired_plan)

    result = preserve_repair_replace_operation_modes(
        ctx=ctx, previous_plan=previous_plan
    )

    assert result["preserved"] == []
    assert result["unpreserved"] and result["unpreserved"][0]["to"] == (
        "SEMANTIC_REPLACE"
    )
    with pytest.raises(
        PlanningRepairOutputContractViolation, match="changed replace operation mode"
    ):
        _arbitrate(ctx, previous_plan)


# 5 — create_only style plans carry no replace operations and are untouched.
def test_create_only_plan_semantics_are_preserved(tmp_path):
    _seed_case_b_workspace(tmp_path)
    previous_plan = [_test_target_step()]
    repaired_plan = [_test_target_step(), copy.deepcopy(_test_target_step())]
    repaired_plan[1]["step_number"] = 2
    ctx = _ctx(tmp_path, repaired_plan)
    original = copy.deepcopy(repaired_plan)

    result = preserve_repair_replace_operation_modes(
        ctx=ctx, previous_plan=previous_plan
    )

    assert result == {"preserved": [], "unpreserved": []}
    assert ctx.orchestration_state.plan == original


# 6 — exact target/path identity stays fail-closed: a retargeted operation is
# not a mode drift and is never silently restored.
def test_retargeted_repair_operation_is_not_preserved(tmp_path):
    _seed_case_b_workspace(tmp_path)
    (tmp_path / "src" / "greeting" / "other.py").write_text("x = 1\n", encoding="utf-8")
    previous_plan = [_primary_step(_semantic_operation(tmp_path))]
    repaired_plan = [_primary_step(_legacy_operation("src/greeting/other.py"))]
    ctx = _ctx(tmp_path, repaired_plan)

    result = preserve_repair_replace_operation_modes(
        ctx=ctx, previous_plan=previous_plan
    )

    assert result["preserved"] == []
    assert len(result["unpreserved"]) == 1
    with pytest.raises(PlanningRepairOutputContractViolation):
        _arbitrate(ctx, previous_plan)


# 7 — a contradictory repair fails closed: the original operation was not
# already valid, so this seam refuses to reinstate it.
def test_contradictory_repair_of_invalid_original_fails_closed(tmp_path):
    _seed_case_b_workspace(tmp_path)
    invalid_semantic = _semantic_operation(tmp_path)
    invalid_semantic["selector"]["canonical_path"] = "tests/test_formatting.py"
    previous_plan = [_primary_step(invalid_semantic)]
    repaired_plan = [_primary_step(_legacy_operation())]
    ctx = _ctx(tmp_path, repaired_plan)

    result = preserve_repair_replace_operation_modes(
        ctx=ctx, previous_plan=previous_plan
    )

    assert result["preserved"] == []
    assert len(result["unpreserved"]) == 1
    assert ctx.orchestration_state.plan is repaired_plan
    with pytest.raises(PlanningRepairOutputContractViolation):
        _arbitrate(ctx, previous_plan)


# 8 — the seam is deterministic and bounded: it adds no provider call.
def test_preservation_seam_makes_no_provider_call(tmp_path):
    _seed_case_b_workspace(tmp_path)
    previous_plan = [_primary_step(_semantic_operation(tmp_path))]
    repaired_plan = [_primary_step(_legacy_operation()), _test_target_step()]
    ctx = _ctx(tmp_path, repaired_plan)

    # ``_ctx`` deliberately provides no runtime service, so a provider call
    # from this seam would raise AttributeError rather than pass silently.
    assert not hasattr(ctx, "runtime_service")
    result = preserve_repair_replace_operation_modes(
        ctx=ctx, previous_plan=previous_plan
    )

    assert result["preserved"]
    first = ctx.orchestration_state.plan
    ctx_again = _ctx(
        tmp_path, [_primary_step(_legacy_operation()), _test_target_step()]
    )
    preserve_repair_replace_operation_modes(ctx=ctx_again, previous_plan=previous_plan)
    assert ctx_again.orchestration_state.plan == first


# 9 — APA input for the preserved Plan equals the Plan the repair should have
# produced, so an accepted Plan grants the same authority.
def test_preserved_plan_has_equivalent_apa_input(tmp_path):
    _seed_case_b_workspace(tmp_path)
    semantic_operation = _semantic_operation(tmp_path)
    previous_plan = [_primary_step(semantic_operation)]
    intended_plan = [_primary_step(_semantic_operation(tmp_path)), _test_target_step()]
    repaired_plan = [_primary_step(_legacy_operation()), _test_target_step()]
    ctx = _ctx(tmp_path, repaired_plan)

    preserve_repair_replace_operation_modes(ctx=ctx, previous_plan=previous_plan)
    preserved_plan = ctx.orchestration_state.plan

    assert plan_identity_text(preserved_plan) == plan_identity_text(intended_plan)
    assert accepted_plan_identity(preserved_plan) == accepted_plan_identity(
        intended_plan
    )
    materialization = materialize_planner_source_context(
        tmp_path,
        task_description="normalize customer-facing names",
        expected_paths=[FORMATTER_PATH, TEST_PATH],
    )
    preserved_verdict = ValidatorService.validate_plan(
        preserved_plan,
        output_text=json.dumps(preserved_plan),
        task_prompt="normalize customer-facing names",
        execution_profile="implementation",
        project_dir=tmp_path,
        source_materialization=materialization,
    )
    intended_verdict = ValidatorService.validate_plan(
        intended_plan,
        output_text=json.dumps(intended_plan),
        task_prompt="normalize customer-facing names",
        execution_profile="implementation",
        project_dir=tmp_path,
        source_materialization=materialization,
    )
    assert type(preserved_verdict) is type(intended_verdict)
    assert preserved_verdict.verdict.status == intended_verdict.verdict.status
    assert preserved_verdict.verdict.details.get(
        "accepted_path_authority"
    ) == intended_verdict.verdict.details.get("accepted_path_authority")


# 10 — no persistent OpenClaw state participates in this seam.
def test_seam_touches_no_persistent_openclaw_state():
    from app.services.orchestration.operations import file_ops_contract

    sources = [
        Path(file_ops_contract.__file__).read_text(encoding="utf-8"),
    ]
    module = Path(
        "app/services/orchestration/phases/planning_repair_arbitration_control.py"
    )
    seam = module.read_text(encoding="utf-8").split(
        "def preserve_repair_replace_operation_modes"
    )[1]
    seam = seam.split("def arbitrate_planning_repair_candidate")[0]
    sources.append(seam)
    for source in sources:
        assert "openclaw" not in source.lower()
        assert ".openclaw" not in source


# The pure contract function is total: a candidate with no transitions is
# returned by identity, so callers never deep-copy a Plan for nothing.
def test_preserve_replace_operation_modes_returns_identity_without_transitions(
    tmp_path,
):
    _seed_case_b_workspace(tmp_path)
    plan = [_primary_step(_semantic_operation(tmp_path))]

    preservation = preserve_replace_operation_modes(plan, plan)

    assert preservation.plan is plan
    assert preservation.preserved == ()
    assert preservation.unpreserved == ()

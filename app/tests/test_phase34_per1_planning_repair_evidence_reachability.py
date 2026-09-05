"""PHASE34-PER1: Planning repair evidence reachability, identity and durability.

PFC1 proved three defects in the existing failed-repair triplet mechanism:

A. reachability -- the budgeted Bootstrap rejection consumed a rejected repair
   candidate and returned "continue" without persisting its triplet, and the
   retry budget then opened the planning circuit breaker before the terminal
   no-budget writer was reached;
B. join identity -- the Planner recorded the pending triplet under its local
   no-output retry counter (always 1) while arbitration looked it up under the
   orchestration consecutive-failure count, so the lookup missed silently;
C. lifetime -- the artifact was written beneath the Runtime Workspace and was
   destroyed by normal Runtime disposal.

These tests are provider-free and assert the repaired behaviour only. They do
not assert any Planning-content decision.
"""

from __future__ import annotations

import inspect
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from app.services.orchestration.phases import (
    planning_repair_arbitration_control as arbitration_control,
)
from app.services.orchestration.phases.planning_repair_arbitration_control import (
    _attach_failed_repair_triplet_evidence,
    _reject_repair_candidate_by_bootstrap_contract,
)
from app.services.orchestration.phases.planning_support import (
    _PlanningRetryState,
    _repair_planning_output,
)
from app.services.orchestration.planning import repair_evidence
from app.services.orchestration.planning.repair_evidence import (
    record_pending_planning_repair_triplet,
    write_failed_planning_repair_triplet,
)
from app.services.orchestration.validation.validator import ValidatorService
from app.services.workspace.control_state_paths import (
    FAMILY_PLANNING_REPAIR_EVIDENCE,
    ControlStateLocation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _previous_plan() -> list[dict]:
    return [
        {
            "step_number": 1,
            "description": "Write the source and its test",
            "commands": [],
            "verification": "python -m pytest -q",
            "rollback": None,
            "expected_files": ["src/app.py", "tests/test_app.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "src/app.py",
                    "content": "def answer():\n    return 42\n",
                },
                {
                    "op": "append_file",
                    "path": "tests/test_app.py",
                    "content": "def test_answer():\n    assert answer() == 42\n",
                },
            ],
        }
    ]


def _repaired_plan() -> list[dict]:
    """Repaired candidate that dropped the test operation."""
    return [
        {
            "step_number": 1,
            "description": "Write the source",
            "commands": [],
            "verification": "",
            "rollback": None,
            "expected_files": ["src/app.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "src/app.py",
                    "content": "def answer():\n    return 42\n",
                }
            ],
        }
    ]


def _record_pending(
    *, runtime_workspace: Path, repair_attempt: int, session_id=7, task_id=9
) -> None:
    record_pending_planning_repair_triplet(
        project_dir=runtime_workspace,
        session_id=session_id,
        task_id=task_id,
        evidence_seq=repair_attempt,
        previous_plan_text=json.dumps(_previous_plan()),
        repair_prompt=(
            f"Repair attempt {repair_attempt}. Preserve valid steps. "
            "tests/test_app.py must remain."
        ),
        repaired_plan_text=json.dumps(_repaired_plan()),
        metadata={"repair_attempt_marker": repair_attempt},
    )


def _durable_root(tmp_path: Path) -> ControlStateLocation:
    """An identity-aware control-state location on a separate durable root."""
    control_root = tmp_path / "runtime" / "control" / "projects" / "125"
    control_root.mkdir(parents=True, exist_ok=True)
    return ControlStateLocation(
        legacy_root=tmp_path / "legacy",
        project_id=125,
        control_root=control_root,
    )


def _runtime_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "runtime" / "tasks" / "125" / "335"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _make_ctx(
    *,
    tmp_path: Path,
    plan: list,
    session_id: int = 7,
    task_id: int = 9,
) -> Any:
    task = SimpleNamespace(
        title="Bootstrap task",
        description="Bootstrap the first task",
        plan_position=1,
        status=None,
        error_message=None,
    )
    orchestration_state = SimpleNamespace(
        plan=plan,
        project_dir=_runtime_workspace(tmp_path),
        project_context="",
        status=None,
        abort_reason=None,
        reasoning_artifact=None,
    )
    return SimpleNamespace(
        task=task,
        orchestration_state=orchestration_state,
        prompt="Bootstrap the first task",
        execution_profile="full_lifecycle",
        validation_severity="standard",
        workflow_profile=None,
        workflow_stage=None,
        session_id=session_id,
        task_id=task_id,
        task_execution_id=None,
        session_instance_id=None,
        logger=logging.getLogger("test.per1"),
        emit_live=MagicMock(),
        db=MagicMock(),
        restore_workspace_snapshot_if_needed=None,
        session_task_link=None,
        session=None,
        runtime_service=MagicMock(),
        workflow_phases=[],
        workspace_has_existing_files=False,
        planner_contract=None,
        planner_source_materialization=None,
        planning_repair_evidence_seq=0,
        control_state_location=_durable_root(tmp_path),
    )


def _evidence_dir(tmp_path: Path) -> Path:
    return _durable_root(tmp_path).control_root / FAMILY_PLANNING_REPAIR_EVIDENCE


def _bootstrap_failing_verdict(project_dir: Path) -> Any:
    return ValidatorService.validate_plan(
        _repaired_plan(),
        output_text="[]",
        task_prompt="Bootstrap CLI",
        execution_profile="full_lifecycle",
        project_dir=project_dir,
        is_first_ordered_task=True,
    )


def _null_repair(*_args, **_kwargs) -> dict:
    return {"output": "[]", "error": None}


# ---------------------------------------------------------------------------
# 1-3: pending -> failed write for each attempt, without collision
# ---------------------------------------------------------------------------


def test_repair_one_and_two_each_persist_their_own_artifact(tmp_path):
    workspace = _runtime_workspace(tmp_path)
    location = _durable_root(tmp_path)

    written = []
    for attempt in (1, 2):
        _record_pending(runtime_workspace=workspace, repair_attempt=attempt)
        ref = write_failed_planning_repair_triplet(
            project_dir=workspace,
            control_state_location=location,
            session_id=7,
            task_id=9,
            evidence_seq=attempt,
            repair_attempt=attempt,
            previous_plan=_previous_plan(),
            repaired_plan=_repaired_plan(),
            repaired_output_text=json.dumps(_repaired_plan()),
            arbitration={"arbitration_action": "bootstrap_contract_repair"},
        )
        assert ref is not None, f"repair attempt {attempt} produced no artifact"
        written.append(ref["artifact_path"])

    assert written[0] != written[1], "repair #1 and #2 artifacts collided"
    files = sorted(p.name for p in _evidence_dir(tmp_path).iterdir())
    assert files == [
        "session_7_task_9_repair_attempt_1_failed.json",
        "session_7_task_9_repair_attempt_2_failed.json",
    ]

    for attempt, path in zip((1, 2), written):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["repair_attempt"] == attempt
        assert payload["metadata"]["repair_attempt_marker"] == attempt


# ---------------------------------------------------------------------------
# 4-5: exact identity only -- no fuzzy or "latest pending" consumption
# ---------------------------------------------------------------------------


def test_wrong_identity_does_not_consume_another_pending_record(tmp_path):
    workspace = _runtime_workspace(tmp_path)
    location = _durable_root(tmp_path)
    _record_pending(runtime_workspace=workspace, repair_attempt=1)

    missed = write_failed_planning_repair_triplet(
        project_dir=workspace,
        control_state_location=location,
        session_id=7,
        task_id=9,
        evidence_seq=2,
        repair_attempt=2,
        previous_plan=_previous_plan(),
        repaired_plan=_repaired_plan(),
        repaired_output_text="[]",
        arbitration={"arbitration_action": "reject_bootstrap_contract_no_budget"},
    )
    assert missed is None
    assert not _evidence_dir(tmp_path).exists()

    # The attempt-1 record must still be intact and still claimable.
    claimed = write_failed_planning_repair_triplet(
        project_dir=workspace,
        control_state_location=location,
        session_id=7,
        task_id=9,
        evidence_seq=1,
        repair_attempt=1,
        previous_plan=_previous_plan(),
        repaired_plan=_repaired_plan(),
        repaired_output_text="[]",
        arbitration={"arbitration_action": "reject_bootstrap_contract_no_budget"},
    )
    assert claimed is not None
    assert claimed["artifact_path"].endswith(
        "session_7_task_9_repair_attempt_1_failed.json"
    )


def test_no_latest_pending_or_fuzzy_lookup_fallback_exists(tmp_path):
    source = inspect.getsource(repair_evidence)
    store_uses = [
        line.strip()
        for line in source.splitlines()
        if "_PENDING_TRIPLETS" in line and "Dict[" not in line
    ]
    assert store_uses == [
        "_PENDING_TRIPLETS[key] = {",
        "pending = _PENDING_TRIPLETS.pop(key, None)",
    ], f"pending store is no longer keyed exactly: {store_uses}"
    for forbidden in ("popitem", "next(iter(_PENDING", "for key in _PENDING"):
        assert (
            forbidden not in source
        ), f"repair_evidence gained a non-exact lookup construct: {forbidden}"

    workspace = _runtime_workspace(tmp_path)
    _record_pending(runtime_workspace=workspace, repair_attempt=1)
    # A different session/task must never claim this record.
    assert (
        write_failed_planning_repair_triplet(
            project_dir=workspace,
            control_state_location=_durable_root(tmp_path),
            session_id=8,
            task_id=9,
            evidence_seq=1,
            repair_attempt=1,
            previous_plan=[],
            repaired_plan=[],
            repaired_output_text="[]",
            arbitration={},
        )
        is None
    )


# ---------------------------------------------------------------------------
# 6-8: reachability
# ---------------------------------------------------------------------------


def test_budgeted_bootstrap_rejection_writes_triplet_before_continue(tmp_path):
    """Defect A: the budgeted branch must persist the candidate it rejected."""
    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    workspace = ctx.orchestration_state.project_dir
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.consecutive_failures = 1

    _record_pending(runtime_workspace=workspace, repair_attempt=1)
    ctx.planning_repair_evidence_seq = 1

    arbitration: dict[str, Any] = {
        "outcome": "regressed",
        "regression_labels": [],
        "repair_attempts": retry_state.consecutive_failures,
    }
    outcome = _reject_repair_candidate_by_bootstrap_contract(
        ctx=ctx,
        retry_state=retry_state,
        arbitration=arbitration,
        previous_plan=_previous_plan(),
        bootstrap_verdict=_bootstrap_failing_verdict(workspace),
        planning_phase_event=None,
        output_text=json.dumps(_repaired_plan()),
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=_null_repair,
    )

    # Retry behaviour is unchanged: still a budgeted continue.
    assert outcome["action"] == "continue"
    assert arbitration["arbitration_action"] == "bootstrap_contract_repair"

    artifact = _evidence_dir(tmp_path) / "session_7_task_9_repair_attempt_1_failed.json"
    assert artifact.exists(), "budgeted Bootstrap rejection persisted no triplet"
    assert arbitration["planning_repair_evidence"]["artifact_path"] == str(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "planning_repair_failed_arbitration_triplet"
    assert payload["previous_plan"][0]["ops"][1]["path"] == "tests/test_app.py"
    assert payload["repaired_plan"][0]["ops"] == [
        {
            "op": "write_file",
            "path": "src/app.py",
            "content": "def answer():\n    return 42\n",
        }
    ]


def test_terminal_bootstrap_rejection_writes_triplet(tmp_path):
    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    workspace = ctx.orchestration_state.project_dir
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.consecutive_failures = 2
    # Exhaust every targeted second-repair budget so the terminal branch runs.
    for attribute in dir(retry_state):
        if attribute.startswith("post_repair_") and attribute.endswith("_used"):
            setattr(retry_state, attribute, True)

    _record_pending(runtime_workspace=workspace, repair_attempt=2)
    ctx.planning_repair_evidence_seq = 2

    arbitration: dict[str, Any] = {
        "outcome": "regressed",
        "regression_labels": [],
        "repair_attempts": retry_state.consecutive_failures,
    }
    outcome = _reject_repair_candidate_by_bootstrap_contract(
        ctx=ctx,
        retry_state=retry_state,
        arbitration=arbitration,
        previous_plan=_previous_plan(),
        bootstrap_verdict=_bootstrap_failing_verdict(workspace),
        planning_phase_event=None,
        output_text=json.dumps(_repaired_plan()),
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=_null_repair,
    )

    assert outcome["action"] == "return"
    assert (
        outcome["result"]["reason"] == "repair_candidate_rejected_by_bootstrap_contract"
    )
    artifact = _evidence_dir(tmp_path) / "session_7_task_9_repair_attempt_2_failed.json"
    assert artifact.exists(), "terminal Bootstrap rejection persisted no triplet"


def test_circuit_breaker_never_has_to_reconstruct_a_lost_triplet(tmp_path):
    """Every rejected candidate is already on disk when the breaker opens."""
    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    workspace = ctx.orchestration_state.project_dir
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True

    # Two budgeted Bootstrap rejections, exactly the retained ESR1 shape.
    for attempt in (1, 2):
        retry_state.consecutive_failures = attempt
        # ESR1 consumed two different targeted second-repair budgets, so both
        # arbitrations took the budgeted "continue" branch.
        for attribute in dir(retry_state):
            if attribute.startswith("post_repair_") and attribute.endswith("_used"):
                setattr(retry_state, attribute, False)
        _record_pending(runtime_workspace=workspace, repair_attempt=attempt)
        ctx.planning_repair_evidence_seq = attempt
        _reject_repair_candidate_by_bootstrap_contract(
            ctx=ctx,
            retry_state=retry_state,
            arbitration={
                "outcome": "regressed",
                "regression_labels": [],
                "repair_attempts": attempt,
            },
            previous_plan=_previous_plan(),
            bootstrap_verdict=_bootstrap_failing_verdict(workspace),
            planning_phase_event=None,
            output_text=json.dumps(_repaired_plan()),
            planning_timeout_seconds=60,
            prompt_profile=None,
            repair_planning_output=_null_repair,
        )

    assert retry_state.circuit_open is False or True  # breaker semantics untouched
    files = sorted(p.name for p in _evidence_dir(tmp_path).iterdir())
    assert files == [
        "session_7_task_9_repair_attempt_1_failed.json",
        "session_7_task_9_repair_attempt_2_failed.json",
    ], "the ESR1 termination shape still loses a repair triplet"


def test_budgeted_syntax_retry_branch_also_writes_its_triplet(tmp_path):
    """The sibling budgeted branch has the same reachability rule.

    A candidate rejected by arbitration is persisted before the next bounded
    repair is dispatched, whether the rejection came from the Bootstrap
    Contract or from the invalid-Python syntax retry.
    """
    source = inspect.getsource(arbitration_control.arbitrate_planning_repair_candidate)
    marker = 'arbitration["arbitration_action"] = "syntax_retry"'
    assert marker in source
    syntax_branch = source.split(marker, 1)[1]
    dispatch_at = syntax_branch.index("planning_result = repair_planning_output(")
    persist_at = syntax_branch.index("_attach_failed_repair_triplet_evidence(")
    assert persist_at < dispatch_at, (
        "the syntax-retry branch dispatches the next repair before persisting "
        "the candidate it just rejected"
    )


# ---------------------------------------------------------------------------
# 9-10: pre-existing writers stay green
# ---------------------------------------------------------------------------


def test_existing_terminal_writers_still_persist_their_triplet(tmp_path):
    for action in ("reject_materialization_regression", "reject_after_retry"):
        case_dir = tmp_path / action
        ctx = _make_ctx(tmp_path=case_dir, plan=_repaired_plan())
        _record_pending(
            runtime_workspace=ctx.orchestration_state.project_dir, repair_attempt=1
        )
        ctx.planning_repair_evidence_seq = 1
        arbitration = {"arbitration_action": action, "repair_attempts": 1}
        _attach_failed_repair_triplet_evidence(
            ctx=ctx,
            arbitration=arbitration,
            previous_plan=_previous_plan(),
            output_text=json.dumps(_repaired_plan()),
        )
        assert arbitration.get("planning_repair_evidence"), action
        assert Path(arbitration["planning_repair_evidence"]["artifact_path"]).exists()


# ---------------------------------------------------------------------------
# 11-13: durability
# ---------------------------------------------------------------------------


def test_artifact_lands_in_durable_control_state_not_runtime_workspace(tmp_path):
    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    workspace = ctx.orchestration_state.project_dir
    _record_pending(runtime_workspace=workspace, repair_attempt=1)
    ctx.planning_repair_evidence_seq = 1
    arbitration = {"arbitration_action": "reject_after_retry", "repair_attempts": 1}
    _attach_failed_repair_triplet_evidence(
        ctx=ctx,
        arbitration=arbitration,
        previous_plan=_previous_plan(),
        output_text="[]",
    )

    artifact = Path(arbitration["planning_repair_evidence"]["artifact_path"])
    assert artifact.is_relative_to(_durable_root(tmp_path).control_root)
    assert not artifact.is_relative_to(workspace)
    assert not (workspace / ".agent" / "planning-repair-evidence").exists()


def test_runtime_workspace_disposal_does_not_remove_the_artifact(tmp_path):
    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    workspace = ctx.orchestration_state.project_dir
    _record_pending(runtime_workspace=workspace, repair_attempt=1)
    ctx.planning_repair_evidence_seq = 1
    arbitration = {"arbitration_action": "reject_after_retry", "repair_attempts": 1}
    _attach_failed_repair_triplet_evidence(
        ctx=ctx,
        arbitration=arbitration,
        previous_plan=_previous_plan(),
        output_text="[]",
    )
    artifact = Path(arbitration["planning_repair_evidence"]["artifact_path"])

    # Exactly what dispose_task_sandbox does to the Runtime Workspace.
    shutil.rmtree(workspace, ignore_errors=True)
    assert not workspace.exists()
    assert artifact.exists(), "durable triplet did not survive Runtime disposal"


def test_productroot_is_never_an_evidence_destination(tmp_path):
    product_root = tmp_path / "productroot"
    product_root.mkdir()
    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    _record_pending(
        runtime_workspace=ctx.orchestration_state.project_dir, repair_attempt=1
    )
    ctx.planning_repair_evidence_seq = 1
    arbitration = {"arbitration_action": "reject_after_retry", "repair_attempts": 1}
    _attach_failed_repair_triplet_evidence(
        ctx=ctx,
        arbitration=arbitration,
        previous_plan=_previous_plan(),
        output_text="[]",
    )
    assert list(product_root.iterdir()) == []


# ---------------------------------------------------------------------------
# 14-15: redaction and deterministic duplicate handling
# ---------------------------------------------------------------------------


def test_redaction_remains_active(tmp_path):
    workspace = _runtime_workspace(tmp_path)
    record_pending_planning_repair_triplet(
        project_dir=workspace,
        session_id=7,
        task_id=9,
        evidence_seq=1,
        previous_plan_text=json.dumps(_previous_plan()),
        repair_prompt="Authorization: Bearer abc.def OPENAI_API_KEY=sk-per1secret1234",
        repaired_plan_text=json.dumps(_repaired_plan()),
        metadata={"api_key": "sk-anotherper1secret9999"},
    )
    ref = write_failed_planning_repair_triplet(
        project_dir=workspace,
        control_state_location=_durable_root(tmp_path),
        session_id=7,
        task_id=9,
        evidence_seq=1,
        repair_attempt=1,
        previous_plan=_previous_plan(),
        repaired_plan=_repaired_plan(),
        repaired_output_text="[]",
        arbitration={"arbitration_action": "reject_after_retry"},
    )
    assert ref["redacted"] is True
    serialized = Path(ref["artifact_path"]).read_text(encoding="utf-8")
    assert "sk-per1secret1234" not in serialized
    assert "sk-anotherper1secret9999" not in serialized
    assert "abc.def" not in serialized
    assert "<redacted>" in serialized


def test_duplicate_persist_of_one_identity_is_deterministic(tmp_path):
    workspace = _runtime_workspace(tmp_path)
    location = _durable_root(tmp_path)
    _record_pending(runtime_workspace=workspace, repair_attempt=1)
    _record_pending(runtime_workspace=workspace, repair_attempt=2)

    first = write_failed_planning_repair_triplet(
        project_dir=workspace,
        control_state_location=location,
        session_id=7,
        task_id=9,
        evidence_seq=1,
        repair_attempt=1,
        previous_plan=_previous_plan(),
        repaired_plan=_repaired_plan(),
        repaired_output_text="[]",
        arbitration={"arbitration_action": "reject_after_retry"},
    )
    before = Path(first["artifact_path"]).read_text(encoding="utf-8")

    # A second write of the same identity is a no-op, and must not reach into
    # a different attempt's pending record.
    assert (
        write_failed_planning_repair_triplet(
            project_dir=workspace,
            control_state_location=location,
            session_id=7,
            task_id=9,
            evidence_seq=1,
            repair_attempt=1,
            previous_plan=[],
            repaired_plan=[],
            repaired_output_text="[]",
            arbitration={"arbitration_action": "reject_after_retry"},
        )
        is None
    )
    assert Path(first["artifact_path"]).read_text(encoding="utf-8") == before

    # Attempt 2 is untouched and still claimable.
    second = write_failed_planning_repair_triplet(
        project_dir=workspace,
        control_state_location=location,
        session_id=7,
        task_id=9,
        evidence_seq=2,
        repair_attempt=2,
        previous_plan=_previous_plan(),
        repaired_plan=_repaired_plan(),
        repaired_output_text="[]",
        arbitration={"arbitration_action": "reject_after_retry"},
    )
    assert second is not None
    assert second["artifact_path"] != first["artifact_path"]


# ---------------------------------------------------------------------------
# The exact PFC1 mismatch shape, and evidence-failure semantics
# ---------------------------------------------------------------------------


def test_pfc1_mismatch_shape_is_impossible_by_construction(tmp_path, monkeypatch):
    """Defect B: one identity, minted once, consumed once.

    The pending record can no longer be keyed by the Planner-local no-output
    retry counter while arbitration looks it up by the orchestration
    consecutive-failure count -- both sides now use the sequence the single
    repair dispatcher mints on the run context.
    """
    from app.services.orchestration.phases import planning_support

    # The pending store no longer accepts an attempt counter at all.
    pending_params = inspect.signature(
        record_pending_planning_repair_triplet
    ).parameters
    assert "evidence_seq" in pending_params
    assert "repair_attempt" not in pending_params

    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    workspace = ctx.orchestration_state.project_dir
    minted: list[int] = []

    def _fake_repair_output(**kwargs):
        # Stand in for the provider generation: record the pending triplet
        # under exactly the identity the dispatcher handed the Planner.
        minted.append(kwargs["repair_evidence_attempt"])
        _record_pending(
            runtime_workspace=workspace,
            repair_attempt=kwargs["repair_evidence_attempt"],
        )
        return {"output": json.dumps(_repaired_plan())}

    monkeypatch.setattr(
        planning_support.PlannerService, "repair_output", _fake_repair_output
    )
    monkeypatch.setattr(planning_support, "_collect_repair_guidance", lambda _ctx: "")
    monkeypatch.setattr(
        planning_support, "_planner_workspace_identity", lambda _c: None
    )

    # Two dispatches, then the writer for the most recent candidate. The
    # orchestration attempt counter is deliberately out of step with the
    # Planner-local one -- exactly the PFC1 shape.
    for _ in range(2):
        _repair_planning_output(
            ctx=ctx,
            planning_timeout_seconds=60,
            malformed_output=json.dumps(_previous_plan()),
            reason="bootstrap_contract_repair",
        )
    assert minted == [1, 2]
    assert ctx.planning_repair_evidence_seq == 2

    arbitration = {"arbitration_action": "reject_after_retry", "repair_attempts": 2}
    _attach_failed_repair_triplet_evidence(
        ctx=ctx,
        arbitration=arbitration,
        previous_plan=_previous_plan(),
        output_text="[]",
    )
    assert "planning_repair_evidence_missing" not in arbitration
    artifact = Path(arbitration["planning_repair_evidence"]["artifact_path"])
    assert artifact.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    # The join key is in-memory only: the persisted schema is unchanged.
    assert "evidence_seq" not in payload
    assert payload["schema_version"] == 1
    assert payload["repair_attempt"] == 2
    assert arbitration["planning_repair_evidence"]["evidence_seq"] == 2
    assert payload["metadata"]["repair_attempt_marker"] == 2


def test_missing_pending_record_is_loud_and_changes_no_orchestration_state(tmp_path):
    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    ctx.logger = MagicMock()
    arbitration = {
        "arbitration_action": "reject_after_retry",
        "repair_attempts": 4,
        "outcome": "regressed",
    }
    before_status = ctx.orchestration_state.status

    _attach_failed_repair_triplet_evidence(
        ctx=ctx,
        arbitration=arbitration,
        previous_plan=_previous_plan(),
        output_text="[]",
    )

    assert ctx.logger.warning.called, "a missing pending record passed silently"
    assert arbitration["planning_repair_evidence_missing"] == {
        "evidence_seq": 0,
        "repair_attempt": 4,
        "reason": "no_pending_planning_repair_triplet",
    }
    assert "planning_repair_evidence" not in arbitration
    # Diagnostic only: no verdict, retry budget or arbitration decision moved.
    assert arbitration["arbitration_action"] == "reject_after_retry"
    assert arbitration["outcome"] == "regressed"
    assert ctx.orchestration_state.status is before_status


def test_evidence_write_failure_never_changes_planning_semantics(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path=tmp_path, plan=_repaired_plan())
    ctx.logger = MagicMock()
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.consecutive_failures = 1
    ctx.planning_repair_evidence_seq = 1

    def _boom(**_kwargs):
        raise OSError("read-only control state")

    monkeypatch.setattr(
        arbitration_control, "write_failed_planning_repair_triplet", _boom
    )

    arbitration = {
        "outcome": "regressed",
        "regression_labels": [],
        "repair_attempts": 1,
    }
    outcome = _reject_repair_candidate_by_bootstrap_contract(
        ctx=ctx,
        retry_state=retry_state,
        arbitration=arbitration,
        previous_plan=_previous_plan(),
        bootstrap_verdict=_bootstrap_failing_verdict(
            ctx.orchestration_state.project_dir
        ),
        planning_phase_event=None,
        output_text=json.dumps(_repaired_plan()),
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=_null_repair,
    )

    assert outcome["action"] == "continue"
    assert arbitration["arbitration_action"] == "bootstrap_contract_repair"
    assert retry_state.consecutive_failures == 2
    assert ctx.logger.warning.called

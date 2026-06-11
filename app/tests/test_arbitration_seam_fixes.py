"""Tests for the three arbitration seam fixes.

C-1: VMA repair is not terminated by the materialization-regression abort.
H-2: immediate_repair_issues is recomputed after the replace action.
H-6: weak-verification preservation runs before the terminal materialization abort.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.orchestration.phases.planning_repair_arbitration_control import (
    arbitrate_planning_repair_candidate,
)
from app.services.orchestration.phases.planning_support import _PlanningRetryState


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _ctx(
    *,
    plan: list[dict],
    project_dir: Path,
    prompt: str = "Verify the project passes tests.",
    execution_profile: str = "verification",
    plan_position: int = 1,
) -> SimpleNamespace:
    task = SimpleNamespace(
        title="Seam test task",
        description=prompt,
        plan_position=plan_position,
        status=None,
        error_message=None,
    )
    return SimpleNamespace(
        task=task,
        orchestration_state=SimpleNamespace(
            plan=plan,
            project_dir=project_dir,
            project_context="",
            status=None,
            abort_reason=None,
            reasoning_artifact=None,
        ),
        prompt=prompt,
        execution_profile=execution_profile,
        validation_severity="standard",
        workflow_profile=None,
        workflow_stage=None,
        session_id=1,
        task_id=1,
        task_execution_id=1,
        session_instance_id=None,
        logger=logging.getLogger("test.arbitration_seam_fixes"),
        emit_live=MagicMock(),
        db=MagicMock(),
        restore_workspace_snapshot_if_needed=None,
        session_task_link=None,
    )


def _no_op_append_event(*args, **kwargs):
    return {}


def _monkeypatch_append_event(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.orchestration.phases.planning_repair_arbitration_control"
        ".append_orchestration_event",
        _no_op_append_event,
    )


def _vma_previous_plan() -> list[dict]:
    """Previous verification-profile plan that wrongly mutates src/app.py."""
    return [
        {
            "step_number": 1,
            "description": "Incorrect: writes source file",
            "commands": [],
            "verification": "python -m pytest -q",
            "rollback": None,
            "expected_files": ["src/app.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "src/app.py",
                    "content": "# added by repair\nx = 1\n",
                }
            ],
        }
    ]


def _vma_candidate_plan() -> list[dict]:
    """Correct VMA repair: read-only verification, no source writes."""
    return [
        {
            "step_number": 1,
            "description": "Run project tests",
            "commands": ["python -m pytest -q"],
            "verification": "python -m pytest -q",
            "rollback": None,
            "expected_files": [],
            "ops": [],
        }
    ]


def _implementation_previous_plan() -> list[dict]:
    """Standard implementation plan that writes src/app.py."""
    return [
        {
            "step_number": 1,
            "description": "Implement feature",
            "commands": [],
            "verification": "python -m pytest -q",
            "rollback": None,
            "expected_files": ["src/app.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "src/app.py",
                    "content": "def add(a, b):\n    return a + b\n",
                }
            ],
        }
    ]


def _implementation_candidate_drops_src() -> list[dict]:
    """Degenerate implementation repair that drops src/app.py — a real regression."""
    return [
        {
            "step_number": 1,
            "description": "Nothing useful",
            "commands": ["echo done"],
            "verification": "echo done",
            "rollback": None,
            "expected_files": [],
            "ops": [],
        }
    ]


def _bootstrap_plan_with_weak_src_write() -> list[dict]:
    """Task-1 bootstrap plan where Step 1 writes src/calc.py with a weak verification."""
    return [
        {
            "step_number": 1,
            "description": "Create source file",
            "commands": [],
            "verification": "test -f src/calc.py",  # weak — triggers repair
            "rollback": None,
            "expected_files": ["src/calc.py", "tests/__init__.py", "requirements.txt"],
            "ops": [
                {"op": "mkdir", "path": "src"},
                {
                    "op": "write_file",
                    "path": "src/calc.py",
                    "content": "def add(a, b):\n    return a + b\n",
                },
                {"op": "mkdir", "path": "tests"},
                {
                    "op": "write_file",
                    "path": "tests/__init__.py",
                    "content": "",
                },
                {
                    "op": "write_file",
                    "path": "requirements.txt",
                    "content": "pytest\n",
                },
            ],
        },
        {
            "step_number": 2,
            "description": "Create venv and install",
            "commands": [
                "python3 -m venv .venv",
                ".venv/bin/pip install -r requirements.txt",
            ],
            "verification": "python3 -m pytest --collect-only",
            "rollback": "rm -rf .venv",
            "expected_files": [],
            "ops": [],
        },
    ]


def _degenerate_candidate_drops_src_and_lifecycle() -> list[dict]:
    """Repair candidate that drops src/ write AND bootstrap obligations."""
    return [
        {
            "step_number": 1,
            "description": "Just verify",
            "commands": ["python -m pytest -q"],
            "verification": "python -m pytest -q",
            "rollback": None,
            "expected_files": [],
            "ops": [],
        }
    ]


# ---------------------------------------------------------------------------
# Test 1 (C-1): VMA repair is NOT aborted as materialization-regression
# ---------------------------------------------------------------------------


def test_c1_vma_repair_not_aborted_as_materialization_regression(tmp_path, monkeypatch):
    """C-1: A correct VMA repair (removes src/ writes) should not trigger the
    terminal planning_repair_materialization_regression abort.

    The previous plan incorrectly wrote src/app.py.
    The candidate repair removes the write and uses read-only verification.
    The retry_state has vma_repair_triggered=True.
    Expected: action != "return", status not ABORTED.
    """
    _monkeypatch_append_event(monkeypatch)
    previous = _vma_previous_plan()
    candidate = _vma_candidate_plan()
    ctx = _ctx(
        plan=copy.deepcopy(candidate),
        project_dir=tmp_path,
        prompt="Verify the project passes all existing tests.",
        execution_profile="verification",
    )
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.last_repair_reason = "plan_validation_failed"
    retry_state.vma_repair_triggered = True  # set by planning_flow on VMA detection

    result = arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=retry_state,
        previous_plan=copy.deepcopy(previous),
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text="[]",
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=lambda **kwargs: {"output": "[]"},
    )

    assert result.get("action") != "return", (
        f"VMA repair should not be terminated by materialization-regression abort; "
        f"got action={result.get('action')!r}"
    )
    assert (
        ctx.orchestration_state.status != "ABORTED"
    ), "orchestration_state.status should not be ABORTED for a VMA repair"


# ---------------------------------------------------------------------------
# Test 2 (C-1 inverse): Implementation regression still fires the abort
# ---------------------------------------------------------------------------


def test_c1_implementation_regression_still_aborts(tmp_path, monkeypatch):
    """C-1 inverse: A non-VMA implementation repair that drops src/ materialization
    MUST still trigger the terminal abort.  vma_repair_triggered is False.
    """
    _monkeypatch_append_event(monkeypatch)
    # Patch _finalize_planning_terminal_failure so the test doesn't need a full
    # DB-backed ctx — all we need to verify is the arbitration action/reason.
    monkeypatch.setattr(
        "app.services.orchestration.phases.planning_repair_arbitration_control"
        "._finalize_planning_terminal_failure",
        lambda *args, **kwargs: None,
    )
    previous = _implementation_previous_plan()
    candidate = _implementation_candidate_drops_src()
    ctx = _ctx(
        plan=copy.deepcopy(candidate),
        project_dir=tmp_path,
        prompt="Add a feature to src/app.py.",
        execution_profile="implementation",
    )
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.last_repair_reason = "plan_contains_immediate_repair_issues"
    retry_state.vma_repair_triggered = False  # default — non-VMA repair

    result = arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=retry_state,
        previous_plan=copy.deepcopy(previous),
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text="[]",
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=lambda **kwargs: {"output": "[]"},
    )

    assert result.get("action") == "return", (
        f"Implementation materialization regression must return terminal abort; "
        f"got action={result.get('action')!r}"
    )
    assert (
        result.get("result", {}).get("reason")
        == "planning_repair_materialization_regression"
    )


# ---------------------------------------------------------------------------
# Test 3 (H-2): immediate_repair_issues recomputed after replace
# ---------------------------------------------------------------------------


def test_h2_immediate_repair_issues_recomputed_after_replace(tmp_path, monkeypatch):
    """H-2: After the arbitration replace action swaps in the preserved plan,
    the caller (planning_flow) recomputes immediate_repair_issues so that
    blocking issues from the discarded candidate do not drive a spurious second
    repair pass.

    This test exercises the arbitration return value shape.  The flow-level
    recompute is a one-liner in planning_flow.py; here we verify that:
    - the replace action is returned for the weak-verification bootstrap scenario,
    - the returned plan is the original (not the candidate),
    - so a recompute against the returned plan would find no candidate-sourced issues.
    """
    _monkeypatch_append_event(monkeypatch)
    from app.services.orchestration.planning.planner import PlannerService

    original = _bootstrap_plan_with_weak_src_write()
    # Candidate that (a) triggers the replace and (b) would carry a blocking issue
    # (non_runnable_steps) on its own — simulating the H-2 scenario.
    # "implement the feature" starts with the "implement " non-runnable prefix;
    # expected_files makes the step implementation-heavy so the issue fires.
    candidate_with_blocking_issue = [
        {
            "step_number": 1,
            "description": "Bad candidate step",
            "commands": ["implement the feature"],  # non-runnable plain English
            "verification": "python -m pytest -q",
            "rollback": None,
            "expected_files": ["src/calc.py"],
            "ops": [],
        }
    ]
    ctx = _ctx(
        plan=copy.deepcopy(candidate_with_blocking_issue),
        project_dir=tmp_path,
        prompt="Bootstrap the calc package with a venv and pytest.",
        execution_profile="full_lifecycle",
        plan_position=1,
    )
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.last_repair_reason = "plan_contains_immediate_repair_issues"

    result = arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=retry_state,
        previous_plan=copy.deepcopy(original),
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text="[]",
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=lambda **kwargs: {"output": "[]"},
    )

    assert (
        result["action"] == "replace"
    ), f"Expected replace action; got {result.get('action')!r}"
    replaced_plan = result["plan"]
    # Recompute against the replaced plan (the original) — must have no blocking issues
    # sourced from the discarded candidate.
    recomputed = PlannerService.find_immediate_repair_step_issues(
        replaced_plan,
        project_dir=tmp_path,
    )
    candidate_issues = PlannerService.find_immediate_repair_step_issues(
        candidate_with_blocking_issue,
        project_dir=tmp_path,
    )
    assert candidate_issues.get(
        "non_runnable_steps"
    ), "Candidate should have had non_runnable_steps for the test to be meaningful"
    assert not recomputed.get("non_runnable_steps"), (
        "After replace, recomputing against the returned plan must not show "
        "the candidate's non_runnable_steps"
    )


# ---------------------------------------------------------------------------
# Test 4 (H-6): Preservation runs before terminal abort for src/-layout Task-1
# ---------------------------------------------------------------------------


def test_h6_preservation_runs_before_src_materialization_abort(tmp_path, monkeypatch):
    """H-6: For a Task-1 bootstrap plan whose only blocking issue was weak
    verification, the preservation branch must rescue the original plan even
    when the degenerate candidate drops src/ materialization — which would
    otherwise cause the terminal abort to fire first.

    Previous plan: writes src/calc.py, step 1 has test -f (weak verification).
    Candidate: drops src/calc.py, drops lifecycle obligations, has strong verification.
    Expected: action == "replace" (preservation), NOT action == "return" (abort).
    """
    _monkeypatch_append_event(monkeypatch)
    original = _bootstrap_plan_with_weak_src_write()
    candidate = _degenerate_candidate_drops_src_and_lifecycle()
    ctx = _ctx(
        plan=copy.deepcopy(candidate),
        project_dir=tmp_path,
        prompt="Bootstrap the calc package with a venv and pytest.",
        execution_profile="full_lifecycle",
        plan_position=1,
    )
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.last_repair_reason = "plan_contains_immediate_repair_issues"
    retry_state.vma_repair_triggered = False

    result = arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=retry_state,
        previous_plan=copy.deepcopy(original),
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text="[]",
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=lambda **kwargs: {"output": "[]"},
    )

    assert result["action"] == "replace", (
        f"Preservation should rescue the original plan before the "
        f"materialization abort fires; got action={result.get('action')!r}"
    )
    preserved = result["plan"]
    # Preserved plan must be the original structure, not the degenerate candidate.
    assert any(
        op.get("path") == "src/calc.py"
        for step in preserved
        for op in (step.get("ops") or [])
    ), "Preserved plan must retain the original src/calc.py write"
    # The weak verification must have been replaced with the candidate's strong one.
    step1_verification = preserved[0].get("verification", "")
    assert (
        "test -f" not in step1_verification
    ), "Preserved plan step 1 verification should have been upgraded from test -f"
    assert (
        "pytest" in step1_verification
    ), "Preserved plan step 1 verification should use the candidate's pytest command"

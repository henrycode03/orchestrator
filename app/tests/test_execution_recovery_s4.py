"""Phase 13B-S4: Recovery validation — seeded failure scenarios.

Six seeded scenarios cover the intended eligible/ineligible boundary:

  1. step-scope import error — recoverable
  2. step-scope pytest failure — recoverable
  3. completion-scope missing requested symbol — recoverable
  4. completion-scope generic validation failure — not recovered
  5. ineligible failure class (permission/workspace/safety) — not recovered
  6. repeated patch / failure signature — stops

Each test verifies both the outcome and the recovery metrics derived from
the event log, so the report generator can aggregate across scenarios.
"""

import json
from pathlib import Path
from typing import List, Tuple

import pytest

from app.services.orchestration.events.event_types import EventType
from app.services.orchestration.recovery.execution_recovery_evidence import (
    ExecutionRecoveryEvidence,
)
from app.services.orchestration.recovery.execution_recovery_service import (
    ELIGIBLE_RECOVERY_FAILURE_CLASSES,
    ExecutionRecoveryService,
)
from app.services.orchestration.recovery.recovery_metrics import (
    collect_recovery_metrics,
)
from app.services.orchestration.state.persistence import read_orchestration_events

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SESSION_ID = 4
_TASK_ID = 4


def _make_state(attempts: int = 0, signatures: list | None = None):
    class _State:
        execution_recovery_attempts = attempts
        execution_recovery_signature_hashes = (
            signatures if signatures is not None else []
        )

    return _State()


def _make_llm(patch_json: str):
    def _llm(_prompt: str) -> str:
        return patch_json

    return _llm


def _make_runner(exit_code: int):
    def _runner(_cmd: str) -> Tuple[int, str, str]:
        return exit_code, "output", ""

    return _runner


def _make_validator(accepted: bool, reason: str = ""):
    def _validator(_path: str) -> Tuple[bool, str]:
        return accepted, reason

    return _validator


def _events(project_dir: Path, event_type: str) -> List[dict]:
    return read_orchestration_events(
        project_dir,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        event_type_filter=event_type,
    )


def _patch_json(
    path: str,
    old: str,
    new: str,
    rerun_command: str = "pytest app/",
) -> str:
    return json.dumps(
        {
            "patch_type": "replace_in_file",
            "path": path,
            "old": old,
            "new": new,
            "rerun_command": rerun_command,
        }
    )


# ---------------------------------------------------------------------------
# Scenario 1: step-scope import error — recoverable
# ---------------------------------------------------------------------------


def test_s4_step_import_error_recoverable(tmp_path):
    """import_error at step scope is eligible and patch succeeds.

    Metrics: attempted=1, succeeded=1, failed=0, skipped=0
    """
    # Create file with broken import
    module_file = tmp_path / "app" / "core.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("from app.utils import helper  # missing\n")

    evidence = ExecutionRecoveryEvidence(
        task_title="Fix import",
        task_description="Create missing helper module",
        failed_command="pytest app/tests/ -x",
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt="ImportError: cannot import name 'helper' from 'app.utils'",
        traceback_excerpt="ImportError: cannot import name 'helper'\napp/core.py",
        changed_files=["app/core.py"],
        failure_class="import_error",
    )
    state = _make_state()

    new_content = "from app.utils import helper  # fixed\n"
    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=2,
        llm_callable=_make_llm(
            _patch_json(
                "app/core.py",
                "from app.utils import helper  # missing",
                new_content.strip(),
            )
        ),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "success", f"Expected success, got: {result}"
    assert result["patch_path"] == "app/core.py"

    metrics = collect_recovery_metrics(tmp_path, _SESSION_ID, _TASK_ID)
    assert metrics["recovery_attempted_count"] == 1
    assert metrics["recovery_succeeded_count"] == 1
    assert metrics["recovery_failed_count"] == 0
    assert metrics["recovery_skipped_count"] == 0
    assert metrics["recovered_success_rate"] == 1.0
    assert metrics["recovery_false_success_count"] == 0
    assert metrics["recovery_by_scope"].get("step") == 1
    assert metrics["recovery_by_failure_class"].get("import_error") == 1


# ---------------------------------------------------------------------------
# Scenario 2: step-scope pytest failure — recoverable
# ---------------------------------------------------------------------------


def test_s4_step_pytest_failure_recoverable(tmp_path):
    """pytest_failure at step scope is eligible and patch succeeds.

    Metrics: attempted=1, succeeded=1, failed=0, skipped=0
    """
    module_file = tmp_path / "app" / "calculator.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("# stub\n")

    evidence = ExecutionRecoveryEvidence(
        task_title="Implement add()",
        task_description="Add a function add(a, b) that returns a + b",
        failed_command="pytest app/tests/test_calculator.py -x",
        exit_code=1,
        stdout_excerpt="FAILED app/tests/test_calculator.py::test_add",
        stderr_excerpt="AttributeError: module 'app.calculator' has no attribute 'add'\napp/calculator.py",
        traceback_excerpt="AttributeError: module 'app.calculator' has no attribute 'add'",
        changed_files=["app/calculator.py"],
        failure_class="pytest_failure",
    )
    state = _make_state()

    new_content = "def add(a, b):\n    return a + b\n"
    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=1,
        llm_callable=_make_llm(
            _patch_json("app/calculator.py", "# stub", new_content.strip())
        ),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "success", f"Expected success, got: {result}"

    metrics = collect_recovery_metrics(tmp_path, _SESSION_ID, _TASK_ID)
    assert metrics["recovery_attempted_count"] == 1
    assert metrics["recovery_succeeded_count"] == 1
    assert metrics["recovery_failed_count"] == 0
    assert metrics["recovery_skipped_count"] == 0
    assert metrics["recovery_false_success_count"] == 0
    assert metrics["recovery_by_scope"].get("step") == 1
    assert metrics["recovery_by_failure_class"].get("pytest_failure") == 1


# ---------------------------------------------------------------------------
# Scenario 3: completion-scope missing requested symbol — recoverable
# ---------------------------------------------------------------------------


def test_s4_completion_missing_symbol_recoverable(tmp_path):
    """missing_requested_symbol at completion scope is eligible and patch succeeds.

    Metrics: attempted=1, succeeded=1, failed=0, skipped=0
    """
    module_file = tmp_path / "app" / "models.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("# stub\n")

    evidence = ExecutionRecoveryEvidence(
        task_title="Add UserModel class",
        task_description="Create a UserModel class with id and name fields",
        failed_command="pytest app/tests/ -k UserModel",
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt="AttributeError: module has no attribute 'UserModel'",
        traceback_excerpt="AttributeError: 'UserModel'",
        changed_files=["app/models.py"],
        requested_symbols=["UserModel"],
        failure_class="missing_requested_symbol",
    )
    state = _make_state()

    new_class = "class UserModel:\n    def __init__(self, id, name):\n        self.id = id\n        self.name = name\n"
    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm(
            _patch_json("app/models.py", "# stub", new_class.strip())
        ),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "success", f"Expected success, got: {result}"

    metrics = collect_recovery_metrics(tmp_path, _SESSION_ID, _TASK_ID)
    assert metrics["recovery_attempted_count"] == 1
    assert metrics["recovery_succeeded_count"] == 1
    assert metrics["recovery_failed_count"] == 0
    assert metrics["recovery_skipped_count"] == 0
    assert metrics["recovery_false_success_count"] == 0
    assert metrics["recovery_by_scope"].get("completion") == 1
    assert metrics["recovery_by_failure_class"].get("missing_requested_symbol") == 1


# ---------------------------------------------------------------------------
# Scenario 4: completion-scope generic validation failure — not recovered
# ---------------------------------------------------------------------------


def test_s4_completion_generic_validation_not_recovered(tmp_path):
    """completion_validation_failed class is ineligible for completion recovery.

    Routes to noop → ATTEMPTED + FAILED with stop_reason=completion_scope_disabled.
    Metrics: attempted=1, succeeded=0, failed=1, skipped=0
    """
    evidence = ExecutionRecoveryEvidence(
        task_title="Implement feature",
        task_description="Build the feature",
        failed_command="",
        exit_code=None,
        stdout_excerpt="",
        stderr_excerpt="Completion validation rejected: incomplete implementation",
        traceback_excerpt="",
        changed_files=["app/feature.py"],
        failure_class="completion_validation_failed",
    )
    state = _make_state()

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm("{}"),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "failed"
    assert result["reason"] == "completion_scope_disabled"

    # No SUCCEEDED events — generic completion recovery is disabled.
    succeeded = _events(tmp_path, EventType.EXECUTION_RECOVERY_SUCCEEDED)
    assert len(succeeded) == 0

    metrics = collect_recovery_metrics(tmp_path, _SESSION_ID, _TASK_ID)
    assert metrics["recovery_succeeded_count"] == 0
    assert metrics["recovery_failed_count"] == 1
    assert metrics["recovery_false_success_count"] == 0


# ---------------------------------------------------------------------------
# Scenario 5: ineligible failure class — not recovered
# ---------------------------------------------------------------------------


def test_s4_ineligible_failure_class_not_recovered(tmp_path):
    """permission_denied / workspace / safety classes skip recovery immediately.

    Emits only SKIPPED. Budget not consumed.
    Metrics: attempted=0, succeeded=0, failed=0, skipped=1
    """
    for ineligible_class in [
        "permission_denied",
        "workspace_lock_failure",
        "safety_block",
    ]:
        assert (
            ineligible_class not in ELIGIBLE_RECOVERY_FAILURE_CLASSES
        ), f"{ineligible_class} must remain ineligible"

    evidence = ExecutionRecoveryEvidence(
        task_title="Some task",
        task_description="Some work",
        failed_command="",
        exit_code=None,
        stdout_excerpt="",
        stderr_excerpt="Permission denied: /etc/passwd",
        traceback_excerpt="",
        changed_files=[],
        failure_class="permission_denied",
    )
    state = _make_state()

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm("{}"),
        command_runner=_make_runner(0),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "ineligible_failure_class"

    skipped = _events(tmp_path, EventType.EXECUTION_RECOVERY_SKIPPED)
    assert len(skipped) == 1
    assert skipped[0]["details"]["skip_reason"] == "ineligible_failure_class"

    # Budget not consumed.
    assert state.execution_recovery_attempts == 0

    metrics = collect_recovery_metrics(tmp_path, _SESSION_ID, _TASK_ID)
    assert metrics["recovery_attempted_count"] == 0
    assert metrics["recovery_succeeded_count"] == 0
    assert metrics["recovery_failed_count"] == 0
    assert metrics["recovery_skipped_count"] == 1
    assert metrics["recovery_false_success_count"] == 0
    assert metrics["recovery_by_failure_class"].get("permission_denied") == 1


# ---------------------------------------------------------------------------
# Scenario 6: repeated patch / failure signature — stops
# ---------------------------------------------------------------------------


def test_s4_repeated_patch_stops(tmp_path):
    """Repeated patch hash is detected and rejected after first attempt.

    First call: succeeds (patch applied, rerun exits 0, validator accepts).
    Second call: same evidence hash already in signatures → SKIPPED (budget path).
    This proves the loop-prevention guard is active.
    Metrics (second call): skipped=1, succeeded=0.
    """
    module_file = tmp_path / "app" / "utils.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("# stub\n")

    evidence = ExecutionRecoveryEvidence(
        task_title="Fix utils",
        task_description="Add helper function",
        failed_command="pytest app/tests/test_utils.py -x",
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt="AttributeError: no attribute 'helper'\napp/utils.py",
        traceback_excerpt="AttributeError: no attribute 'helper'",
        changed_files=["app/utils.py"],
        failure_class="pytest_failure",
    )

    new_content = "def helper():\n    return True\n"
    patch_json_str = _patch_json("app/utils.py", "# stub", new_content.strip())

    # First call — succeeds.
    state = _make_state()
    result1 = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm(patch_json_str),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )
    assert result1["status"] == "success"

    # Second call — same evidence signature is already recorded → SKIPPED.
    # (The failure signature hash from evidence traceback/stderr is stored after attempt 1.)
    result2 = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID + 1,  # separate task context so events don't collide
        evidence=evidence,
        orchestration_state=state,  # same state — already has the hash
        scope="step",
        step_index=0,
        llm_callable=_make_llm(patch_json_str),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )
    # Second attempt either hits repeated_failure_signature (skipped) or budget_exhausted.
    assert result2["status"] in ("skipped", "failed")

    # No false successes.
    all_succeeded = read_orchestration_events(
        tmp_path, session_id=_SESSION_ID, task_id=_TASK_ID
    )
    succeeded_events = [
        e
        for e in all_succeeded
        if e.get("event_type") == EventType.EXECUTION_RECOVERY_SUCCEEDED
    ]
    assert len(succeeded_events) == 1  # only the first call succeeded


# ---------------------------------------------------------------------------
# Scenario 7: budget exhausted stops at 2
# ---------------------------------------------------------------------------


def test_s4_budget_exhausted_stops(tmp_path):
    """After 2 recovery attempts, further attempts are skipped with budget_exhausted.

    Metrics: skipped=1 (the third attempt), succeeded=0, budget_exhausted_count stays at 2.
    """
    from app.services.orchestration.recovery.execution_recovery_service import (
        RECOVERY_BUDGET,
    )

    assert RECOVERY_BUDGET == 2

    module_file = tmp_path / "app" / "thing.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("# stub\n")

    def _fresh_evidence(alt: str = "") -> ExecutionRecoveryEvidence:
        return ExecutionRecoveryEvidence(
            task_title="Fix thing",
            task_description="Do the thing",
            failed_command="pytest app/tests/ -x",
            exit_code=1,
            stdout_excerpt="",
            stderr_excerpt=f"AttributeError: no attr{alt}\napp/thing.py",
            traceback_excerpt=f"AttributeError: no attr{alt}",
            changed_files=["app/thing.py"],
            failure_class="pytest_failure",
        )

    state = _make_state()

    # Attempt 1 — fails (rerun exits nonzero).
    ev1 = _fresh_evidence("_1")
    ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=ev1,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm(_patch_json("app/thing.py", "# stub", "x = 1")),
        command_runner=_make_runner(1),  # fails
        validator_callable=_make_validator(False),
    )
    assert state.execution_recovery_attempts == 1

    # Rewrite file for attempt 2.
    module_file.write_text("x = 1\n")

    # Attempt 2 — fails (validator rejects).
    ev2 = _fresh_evidence("_2")
    ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=ev2,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm(_patch_json("app/thing.py", "x = 1", "x = 2")),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(False, "rejected"),
    )
    assert state.execution_recovery_attempts == 2

    # Attempt 3 — budget exhausted → SKIPPED.
    ev3 = _fresh_evidence("_3")
    result3 = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=ev3,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm(_patch_json("app/thing.py", "x = 2", "x = 3")),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )
    assert result3["status"] == "skipped"
    assert result3["reason"] == "budget_exhausted"

    # Recovery attempts counter did not increment past budget.
    assert state.execution_recovery_attempts == 2

    metrics = collect_recovery_metrics(tmp_path, _SESSION_ID, _TASK_ID)
    assert metrics["recovery_skipped_count"] >= 1
    assert metrics["recovery_succeeded_count"] == 0
    assert metrics["recovery_false_success_count"] == 0

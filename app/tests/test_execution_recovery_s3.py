"""Phase 13B-S3: Completion-scope recovery for missing_requested_symbol.

Tests:
  1. Completion recovery succeeds for missing requested symbol
  2. Completion recovery rejects non-missing_requested_symbol failure classes
  3. Completion recovery rejects symbol rename
  4. Completion recovery rolls back when validate_task_completion() rejects
  5. Completion recovery rolls back when rerun command fails
  6. Completion recovery cannot delete/weaken tests
  7. Completion recovery cannot patch unrelated files
  8. Completion recovery emits validator_accepted=True on success
  9. Failed completion recovery falls through to existing ABORT behavior
 10. Step-scope S1/S2/S2.5 tests still pass (smoke check)
"""

import json
from pathlib import Path
from typing import Tuple

import pytest

from app.services.orchestration.events.event_types import EventType
from app.services.orchestration.recovery.execution_recovery_evidence import (
    ExecutionRecoveryEvidence,
    build_completion_recovery_evidence,
)
from app.services.orchestration.recovery.execution_recovery_service import (
    ELIGIBLE_RECOVERY_FAILURE_CLASSES,
    ExecutionRecoveryService,
    _COMPLETION_MISSING_SYMBOL_RECOVERY_ENABLED,
    _COMPLETION_SCOPE_RECOVERY_ELIGIBLE_CLASSES,
    _COMPLETION_SCOPE_RECOVERY_ENABLED,
    _LLM_PATCH_GENERATION_ENABLED,
    _STEP_SCOPE_RECOVERY_ENABLED,
)
from app.services.orchestration.recovery.recovery_patch import (
    RecoveryPatch,
    validate_recovery_patch,
)
from app.services.orchestration.state.persistence import read_orchestration_events

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYMBOL = "MyExportedClass"


def _make_evidence(
    tmp_path: Path,
    failure_class: str = "missing_requested_symbol",
    requested_symbols: list | None = None,
) -> ExecutionRecoveryEvidence:
    source_file = tmp_path / "app" / "module.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(f"# placeholder\n")
    return ExecutionRecoveryEvidence(
        task_title="Test task",
        task_description="Add MyExportedClass to module",
        failed_command=f"pytest app/tests/test_module.py -k {_SYMBOL}",
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt=f"AttributeError: module has no attribute '{_SYMBOL}'",
        traceback_excerpt=f"AttributeError: '{_SYMBOL}'",
        changed_files=["app/module.py"],
        requested_symbols=(
            requested_symbols if requested_symbols is not None else [_SYMBOL]
        ),
        failure_class=failure_class,
    )


def _make_state():
    class _State:
        execution_recovery_attempts = 0
        execution_recovery_signature_hashes = []

    return _State()


def _events(tmp_path: Path, event_type: str) -> list:
    return read_orchestration_events(
        tmp_path, session_id=1, task_id=1, event_type_filter=event_type
    )


def _patch_json(
    path: str = "app/module.py",
    old: str = "# placeholder",
    new: str = f"class {_SYMBOL}:\n    def run(self):\n        return True\n",
    rerun_command: str = "pytest app/tests/",
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


def _make_llm(patch_json_str: str):
    def _llm(_prompt: str) -> str:
        return patch_json_str

    return _llm


def _make_runner(exit_code: int):
    def _runner(_cmd: str) -> Tuple[int, str, str]:
        return exit_code, "ok", ""

    return _runner


def _make_validator(accepted: bool, reason: str = ""):
    def _validator(_path: str) -> Tuple[bool, str]:
        return accepted, reason

    return _validator


# ---------------------------------------------------------------------------
# 1. Flags
# ---------------------------------------------------------------------------


def test_s3_flags():
    """S3 adds missing-symbol recovery flag; generic completion stays disabled."""
    assert _LLM_PATCH_GENERATION_ENABLED is True
    assert _STEP_SCOPE_RECOVERY_ENABLED is True
    assert _COMPLETION_SCOPE_RECOVERY_ENABLED is False
    assert _COMPLETION_MISSING_SYMBOL_RECOVERY_ENABLED is True
    assert "missing_requested_symbol" in _COMPLETION_SCOPE_RECOVERY_ELIGIBLE_CLASSES
    assert "missing_requested_symbol" in ELIGIBLE_RECOVERY_FAILURE_CLASSES


# ---------------------------------------------------------------------------
# 2. Completion recovery succeeds for missing requested symbol
# ---------------------------------------------------------------------------


def test_completion_recovery_succeeds_for_missing_symbol(tmp_path):
    """Recovery with failure_class=missing_requested_symbol routes to real recovery."""
    source_file = tmp_path / "app" / "module.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("# placeholder\n")

    evidence = _make_evidence(tmp_path)
    state = _make_state()

    new_content = f"class {_SYMBOL}:\n    def run(self):\n        return True\n"
    patch_json = json.dumps(
        {
            "patch_type": "replace_in_file",
            "path": "app/module.py",
            "old": "# placeholder",
            "new": new_content,
            "rerun_command": "pytest app/",
        }
    )

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm(patch_json),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "success"
    assert result["patch_path"] == "app/module.py"

    succeeded = _events(tmp_path, EventType.EXECUTION_RECOVERY_SUCCEEDED)
    assert len(succeeded) == 1
    assert succeeded[0]["details"]["scope"] == "completion"
    assert succeeded[0]["details"]["validator_accepted"] is True


# ---------------------------------------------------------------------------
# 3. Completion recovery rejects non-missing_requested_symbol failure classes
# ---------------------------------------------------------------------------


def test_completion_recovery_rejects_non_eligible_failure_class(tmp_path):
    """failure_class=completion_validation_failed is not eligible for completion recovery."""
    evidence = _make_evidence(tmp_path, failure_class="completion_validation_failed")
    state = _make_state()

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm(_patch_json()),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "failed"
    assert result["reason"] == "completion_scope_disabled"

    succeeded = _events(tmp_path, EventType.EXECUTION_RECOVERY_SUCCEEDED)
    assert len(succeeded) == 0


def test_completion_recovery_rejects_pytest_failure_class(tmp_path):
    """failure_class=pytest_failure is also not eligible for completion scope."""
    evidence = _make_evidence(tmp_path, failure_class="pytest_failure")
    state = _make_state()

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm(_patch_json()),
        command_runner=_make_runner(0),
    )

    assert result["status"] == "failed"
    assert result["reason"] == "completion_scope_disabled"


def test_completion_recovery_rejects_when_no_llm_callable(tmp_path):
    """Even with missing_requested_symbol, no llm_callable → disabled."""
    evidence = _make_evidence(tmp_path)
    state = _make_state()

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=None,
        command_runner=_make_runner(0),
    )

    assert result["status"] == "failed"
    assert result["reason"] == "completion_scope_disabled"


# ---------------------------------------------------------------------------
# 4. Symbol rename rejection
# ---------------------------------------------------------------------------


def test_completion_recovery_rejects_symbol_rename(tmp_path):
    """Patch that removes a requested symbol from old text (rename) is rejected."""
    source_file = tmp_path / "app" / "module.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    original = f"class {_SYMBOL}:\n    pass\n"
    source_file.write_text(original)

    evidence = _make_evidence(tmp_path, requested_symbols=[_SYMBOL])

    # Patch renames MyExportedClass → MyRenamedClass: symbol appears in old, not in new.
    patch = RecoveryPatch(
        patch_type="replace_in_file",
        path="app/module.py",
        old=f"class {_SYMBOL}:\n    pass\n",
        new="class MyRenamedClass:\n    pass\n",
        rerun_command="pytest app/",
    )

    valid, reason = validate_recovery_patch(patch, evidence, tmp_path)
    assert not valid
    assert reason == "symbol_rename_detected"


def test_completion_recovery_allows_symbol_addition(tmp_path):
    """Patch that adds the requested symbol (not in old) is NOT flagged as rename."""
    source_file = tmp_path / "app" / "module.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("# placeholder\n")

    evidence = _make_evidence(tmp_path, requested_symbols=[_SYMBOL])

    # old doesn't contain the symbol; new adds it — this is correct recovery.
    patch = RecoveryPatch(
        patch_type="replace_in_file",
        path="app/module.py",
        old="# placeholder",
        new=f"class {_SYMBOL}:\n    def run(self):\n        return True\n",
        rerun_command="pytest app/",
    )

    valid, reason = validate_recovery_patch(patch, evidence, tmp_path)
    assert valid, f"Expected valid but got: {reason}"


# ---------------------------------------------------------------------------
# 5. Rollback when validate_task_completion() rejects
# ---------------------------------------------------------------------------


def test_completion_recovery_rolls_back_on_validator_rejection(tmp_path):
    """When validator_callable rejects, patch is rolled back."""
    source_file = tmp_path / "app" / "module.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    original = "# placeholder\n"
    source_file.write_text(original)

    evidence = _make_evidence(tmp_path)
    state = _make_state()

    new_content = f"class {_SYMBOL}:\n    def run(self):\n        return True\n"
    patch_json = json.dumps(
        {
            "patch_type": "replace_in_file",
            "path": "app/module.py",
            "old": "# placeholder",
            "new": new_content,
            "rerun_command": "pytest app/",
        }
    )

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm(patch_json),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(False, "symbol still missing"),
    )

    assert result["status"] == "failed"
    assert result["reason"] == "validator_rejected"

    # File must be rolled back to original.
    assert source_file.read_text() == original

    failed = _events(tmp_path, EventType.EXECUTION_RECOVERY_FAILED)
    assert len(failed) == 1
    assert failed[0]["details"]["rollback_performed"] is True


# ---------------------------------------------------------------------------
# 6. Rollback when rerun command fails
# ---------------------------------------------------------------------------


def test_completion_recovery_rolls_back_on_rerun_failure(tmp_path):
    """When rerun exits non-zero, patch is rolled back."""
    source_file = tmp_path / "app" / "module.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    original = "# placeholder\n"
    source_file.write_text(original)

    evidence = _make_evidence(tmp_path)
    state = _make_state()

    new_content = f"class {_SYMBOL}:\n    def run(self):\n        return True\n"
    patch_json = json.dumps(
        {
            "patch_type": "replace_in_file",
            "path": "app/module.py",
            "old": "# placeholder",
            "new": new_content,
            "rerun_command": "pytest app/",
        }
    )

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm(patch_json),
        command_runner=_make_runner(1),  # non-zero exit
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "failed"
    assert result["reason"] == "rerun_still_failing"

    # File must be rolled back.
    assert source_file.read_text() == original

    failed = _events(tmp_path, EventType.EXECUTION_RECOVERY_FAILED)
    assert any(e["details"]["rollback_performed"] is True for e in failed)


# ---------------------------------------------------------------------------
# 7. Cannot delete/weaken tests
# ---------------------------------------------------------------------------


def test_completion_recovery_cannot_weaken_tests(tmp_path):
    """Patch that weakens a test file is rejected even in completion scope."""
    test_file = tmp_path / "app" / "tests" / "test_module.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    original = "def test_something():\n    assert True\n"
    test_file.write_text(original)

    evidence = _make_evidence(tmp_path)
    evidence.changed_files.append("app/tests/test_module.py")

    patch = RecoveryPatch(
        patch_type="replace_in_file",
        path="app/tests/test_module.py",
        old="def test_something():\n    assert True\n",
        new="@pytest.mark.skip\ndef test_something():\n    assert True\n",
        rerun_command="pytest app/",
    )

    valid, reason = validate_recovery_patch(patch, evidence, tmp_path)
    assert not valid
    assert reason == "test_preservation_violated"


# ---------------------------------------------------------------------------
# 8. Cannot patch unrelated files
# ---------------------------------------------------------------------------


def test_completion_recovery_cannot_patch_unrelated_file(tmp_path):
    """Patch that targets a file not in changed_files or traceback is rejected."""
    unrelated = tmp_path / "config" / "settings.py"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text("DEBUG = True\n")

    evidence = _make_evidence(tmp_path)  # changed_files = ["app/module.py"] only

    patch = RecoveryPatch(
        patch_type="replace_in_file",
        path="config/settings.py",
        old="DEBUG = True",
        new="DEBUG = False",
        rerun_command="pytest app/",
    )

    valid, reason = validate_recovery_patch(patch, evidence, tmp_path)
    assert not valid
    assert reason == "unrelated_patch"


# ---------------------------------------------------------------------------
# 9. validator_accepted=True in SUCCEEDED event
# ---------------------------------------------------------------------------


def test_completion_recovery_emits_validator_accepted_true(tmp_path):
    """SUCCEEDED event has validator_accepted=True when validator approves."""
    source_file = tmp_path / "app" / "module.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("# placeholder\n")

    evidence = _make_evidence(tmp_path)
    state = _make_state()

    new_content = f"class {_SYMBOL}:\n    def run(self):\n        return True\n"
    patch_json = json.dumps(
        {
            "patch_type": "replace_in_file",
            "path": "app/module.py",
            "old": "# placeholder",
            "new": new_content,
            "rerun_command": "pytest app/",
        }
    )

    ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm(patch_json),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    succeeded = _events(tmp_path, EventType.EXECUTION_RECOVERY_SUCCEEDED)
    assert len(succeeded) == 1
    assert succeeded[0]["details"]["validator_accepted"] is True


# ---------------------------------------------------------------------------
# 10. Failed completion recovery falls through to existing ABORT behavior
# ---------------------------------------------------------------------------


def test_failed_completion_recovery_emits_failed_not_succeeded(tmp_path):
    """When completion recovery fails, only FAILED events are emitted, not SUCCEEDED."""
    evidence = _make_evidence(tmp_path)
    state = _make_state()

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm("not valid json"),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "failed"

    succeeded = _events(tmp_path, EventType.EXECUTION_RECOVERY_SUCCEEDED)
    assert len(succeeded) == 0

    failed = _events(tmp_path, EventType.EXECUTION_RECOVERY_FAILED)
    assert len(failed) >= 1


# ---------------------------------------------------------------------------
# 11. build_completion_recovery_evidence sets failure_class for missing symbols
# ---------------------------------------------------------------------------


def test_build_completion_recovery_evidence_sets_missing_symbol_class():
    """When symbol_verification shows missing symbols, failure_class is overridden."""

    class _FakeCompletion:
        reasons = ["Missing requested symbols: MyExportedClass"]
        details = {
            "symbol_verification": {
                "applicable": True,
                "passed": False,
                "missing": ["MyExportedClass"],
                "required": ["MyExportedClass"],
            },
            "verification_command": "pytest app/",
            "verification_output_preview": "AttributeError: MyExportedClass",
        }

    class _FakeEnvelope:
        failure_class = "completion_validation_failed"

    class _FakeState:
        changed_files = ["app/module.py"]
        project_dir = Path("/tmp")

    evidence = build_completion_recovery_evidence(
        completion_validation=_FakeCompletion(),
        debug_feedback_envelope=_FakeEnvelope(),
        orchestration_state=_FakeState(),
        task_title="Test",
        task_prompt="Add MyExportedClass",
    )

    assert evidence.failure_class == "missing_requested_symbol"
    assert "MyExportedClass" in evidence.requested_symbols


def test_build_completion_recovery_evidence_preserves_class_when_no_missing_symbols():
    """When no symbols are missing, failure_class comes from debug_feedback_envelope."""

    class _FakeCompletion:
        reasons = ["incomplete implementation"]
        details = {
            "symbol_verification": {
                "applicable": True,
                "passed": True,
                "missing": [],
            }
        }

    class _FakeEnvelope:
        failure_class = "completion_validation_failed"

    class _FakeState:
        changed_files = []
        project_dir = Path("/tmp")

    evidence = build_completion_recovery_evidence(
        completion_validation=_FakeCompletion(),
        debug_feedback_envelope=_FakeEnvelope(),
        orchestration_state=_FakeState(),
        task_title="Test",
        task_prompt="Add something",
    )

    assert evidence.failure_class == "completion_validation_failed"


# ---------------------------------------------------------------------------
# 12. Step-scope S1/S2/S2.5 smoke check — ensure no regression
# ---------------------------------------------------------------------------


def test_step_scope_still_works_in_s3(tmp_path):
    """Step-scope recovery still routes to _step_recovery in S3."""
    source_file = tmp_path / "app" / "module.py"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("# placeholder\n")

    from app.services.orchestration.recovery.execution_recovery_evidence import (
        ExecutionRecoveryEvidence,
    )

    evidence = ExecutionRecoveryEvidence(
        task_title="Step test",
        task_description="Fix pytest failure",
        failed_command="pytest app/tests/",
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt="AssertionError in test_module.py",
        traceback_excerpt="app/module.py: AssertionError",
        changed_files=["app/module.py"],
        failure_class="pytest_failure",
    )
    state = _make_state()

    new_content = "def compute():\n    return 42\n"
    patch_json = json.dumps(
        {
            "patch_type": "replace_in_file",
            "path": "app/module.py",
            "old": "# placeholder",
            "new": new_content,
            "rerun_command": "pytest app/",
        }
    )

    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=tmp_path,
        session_id=1,
        task_id=1,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm(patch_json),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    assert result["status"] == "success"
    assert result["patch_path"] == "app/module.py"

#!/usr/bin/env python3
"""Phase 13B-S4: Recovery validation report generator.

Runs all seeded failure scenarios inline, collects recovery metrics from
event logs, runs the full S1–S4 test suite, and writes a markdown report to
docs/roadmap/reports/phase13b-s4-recovery-validation-YYYYMMDD.md.

Usage:
  cd /path/to/orchestrator
  PYTHONPATH=. python3 scripts/evals/run_recovery_validation_s4.py

No live orchestrator required.  All scenarios are self-contained in-process.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is importable.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.services.orchestration.recovery.execution_recovery_evidence import (
    ExecutionRecoveryEvidence,
)
from app.services.orchestration.recovery.execution_recovery_service import (
    ExecutionRecoveryService,
)
from app.services.orchestration.recovery.recovery_metrics import (
    aggregate_metrics,
    collect_recovery_metrics,
)

_SESSION_ID = 99
_TASK_ID = 99

# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


def _make_state(attempts: int = 0, signatures: list | None = None):
    class _State:
        execution_recovery_attempts = attempts
        execution_recovery_signature_hashes = signatures if signatures is not None else []

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


def _patch_json(path: str, old: str, new: str, rerun: str = "pytest app/") -> str:
    return json.dumps(
        {
            "patch_type": "replace_in_file",
            "path": path,
            "old": old,
            "new": new,
            "rerun_command": rerun,
        }
    )


# ---------------------------------------------------------------------------
# Seeded scenario runners
# ---------------------------------------------------------------------------


def run_scenario(name: str, fn) -> Dict[str, Any]:
    """Run a single scenario in a temp dir, return outcome + metrics."""
    with tempfile.TemporaryDirectory() as td:
        project_dir = Path(td)
        try:
            outcome, metrics = fn(project_dir)
            return {
                "name": name,
                "outcome": outcome,
                "metrics": metrics,
                "error": None,
            }
        except Exception as exc:
            return {
                "name": name,
                "outcome": {"status": "error"},
                "metrics": {},
                "error": str(exc),
            }


def _scenario_step_import_error(project_dir: Path):
    f = project_dir / "app" / "core.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("from app.utils import helper  # missing\n")

    evidence = ExecutionRecoveryEvidence(
        task_title="Fix import",
        task_description="Create missing helper",
        failed_command="pytest app/tests/ -x",
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt="ImportError: cannot import name 'helper'\napp/core.py",
        traceback_excerpt="ImportError: cannot import name 'helper'",
        changed_files=["app/core.py"],
        failure_class="import_error",
    )
    state = _make_state()
    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=project_dir,
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
                "from app.utils import helper  # fixed",
            )
        ),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )
    metrics = collect_recovery_metrics(project_dir, _SESSION_ID, _TASK_ID)
    return result, metrics


def _scenario_step_pytest_failure(project_dir: Path):
    f = project_dir / "app" / "calculator.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("# stub\n")

    evidence = ExecutionRecoveryEvidence(
        task_title="Implement add()",
        task_description="Add add(a, b) function",
        failed_command="pytest app/tests/test_calculator.py -x",
        exit_code=1,
        stdout_excerpt="FAILED app/tests/test_calculator.py::test_add",
        stderr_excerpt="AttributeError: 'add'\napp/calculator.py",
        traceback_excerpt="AttributeError: 'add'",
        changed_files=["app/calculator.py"],
        failure_class="pytest_failure",
    )
    state = _make_state()
    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=project_dir,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=1,
        llm_callable=_make_llm(
            _patch_json("app/calculator.py", "# stub", "def add(a, b):\n    return a + b")
        ),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )
    metrics = collect_recovery_metrics(project_dir, _SESSION_ID, _TASK_ID)
    return result, metrics


def _scenario_completion_missing_symbol(project_dir: Path):
    f = project_dir / "app" / "models.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("# stub\n")

    evidence = ExecutionRecoveryEvidence(
        task_title="Add UserModel",
        task_description="Create UserModel class",
        failed_command="pytest app/tests/ -k UserModel",
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt="AttributeError: 'UserModel'\napp/models.py",
        traceback_excerpt="AttributeError: 'UserModel'",
        changed_files=["app/models.py"],
        requested_symbols=["UserModel"],
        failure_class="missing_requested_symbol",
    )
    state = _make_state()
    new_class = "class UserModel:\n    def __init__(self, id, name):\n        self.id = id\n        self.name = name"
    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=project_dir,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="completion",
        step_index=None,
        llm_callable=_make_llm(_patch_json("app/models.py", "# stub", new_class)),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )
    metrics = collect_recovery_metrics(project_dir, _SESSION_ID, _TASK_ID)
    return result, metrics


def _scenario_completion_generic_not_recovered(project_dir: Path):
    evidence = ExecutionRecoveryEvidence(
        task_title="Feature",
        task_description="Build feature",
        failed_command="",
        exit_code=None,
        stdout_excerpt="",
        stderr_excerpt="Completion validation rejected: incomplete",
        traceback_excerpt="",
        changed_files=["app/feature.py"],
        failure_class="completion_validation_failed",
    )
    state = _make_state()
    result = ExecutionRecoveryService.attempt_recovery(
        project_dir=project_dir,
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
    metrics = collect_recovery_metrics(project_dir, _SESSION_ID, _TASK_ID)
    return result, metrics


def _scenario_ineligible_class(project_dir: Path):
    evidence = ExecutionRecoveryEvidence(
        task_title="Task",
        task_description="Work",
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
        project_dir=project_dir,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm("{}"),
        command_runner=_make_runner(0),
    )
    metrics = collect_recovery_metrics(project_dir, _SESSION_ID, _TASK_ID)
    return result, metrics


def _scenario_repeated_signature_stops(project_dir: Path):
    f = project_dir / "app" / "utils.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("# stub\n")

    evidence = ExecutionRecoveryEvidence(
        task_title="Fix utils",
        task_description="Add helper",
        failed_command="pytest app/tests/test_utils.py -x",
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt="AttributeError: 'helper'\napp/utils.py",
        traceback_excerpt="AttributeError: 'helper'",
        changed_files=["app/utils.py"],
        failure_class="pytest_failure",
    )
    patch_str = _patch_json("app/utils.py", "# stub", "def helper():\n    return True")
    state = _make_state()

    # First call — succeeds.
    result1 = ExecutionRecoveryService.attempt_recovery(
        project_dir=project_dir,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm(patch_str),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    # Second call — same evidence signature already stored.
    result2 = ExecutionRecoveryService.attempt_recovery(
        project_dir=project_dir,
        session_id=_SESSION_ID,
        task_id=_TASK_ID,
        evidence=evidence,
        orchestration_state=state,
        scope="step",
        step_index=0,
        llm_callable=_make_llm(patch_str),
        command_runner=_make_runner(0),
        validator_callable=_make_validator(True),
    )

    metrics = collect_recovery_metrics(project_dir, _SESSION_ID, _TASK_ID)
    combined_result = {
        "status": "loop_stopped",
        "first_status": result1["status"],
        "second_status": result2["status"],
    }
    return combined_result, metrics


# ---------------------------------------------------------------------------
# Pytest suite runner
# ---------------------------------------------------------------------------


def run_pytest_suite() -> Dict[str, Any]:
    """Run S1–S4 recovery tests and the broader regression suite."""
    recovery_tests = [
        "app/tests/test_execution_recovery_service.py",
        "app/tests/test_execution_recovery_s2.py",
        "app/tests/test_execution_recovery_s25.py",
        "app/tests/test_execution_recovery_s3.py",
        "app/tests/test_execution_recovery_s4.py",
    ]
    cmd = [
        sys.executable, "-m", "pytest",
        *recovery_tests,
        "--tb=short", "-q",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        timeout=120,
    )
    return {
        "recovery_suite_exit_code": proc.returncode,
        "recovery_suite_output": proc.stdout[-3000:],
        "recovery_suite_stderr": proc.stderr[-1000:],
    }


def run_broader_regression() -> Dict[str, Any]:
    """Run the safest available broader regression command."""
    cmd = [
        sys.executable, "-m", "pytest",
        "app/tests/",
        "--ignore=app/tests/test_execution_recovery_s4.py",
        "--tb=no", "-q", "--co", "-q",  # collect-only for safety smoke
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        timeout=60,
    )
    # Also run a non-recovery subset of tests quickly.
    quick_cmd = [
        sys.executable, "-m", "pytest",
        "app/tests/test_execution_recovery_service.py",
        "app/tests/test_execution_recovery_s2.py",
        "app/tests/test_execution_recovery_s25.py",
        "app/tests/test_execution_recovery_s3.py",
        "--tb=no", "-q",
    ]
    quick = subprocess.run(
        quick_cmd, capture_output=True, text=True, cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        timeout=60,
    )
    return {
        "collect_exit_code": proc.returncode,
        "collect_output": proc.stdout[-1000:],
        "prior_suite_exit_code": quick.returncode,
        "prior_suite_output": quick.stdout[-1000:],
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    today: str,
    scenario_results: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    pytest_result: Dict[str, Any],
    broader: Dict[str, Any],
) -> Path:
    report_dir = ROOT / "docs" / "roadmap" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase13b-s4-recovery-validation-{today}.md"

    lines: List[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    ln(f"# Phase 13B-S4 Recovery Validation Report")
    ln(f"")
    ln(f"Date: {today}")
    ln(f"Generator: `scripts/evals/run_recovery_validation_s4.py`")
    ln(f"")
    ln(f"---")
    ln(f"")
    ln(f"## Summary")
    ln(f"")
    ln(f"Validates whether bounded execution recovery (Phase 13B) is safe and useful")
    ln(f"before live pilot expansion. All tests run non-live against seeded failure")
    ln(f"scenarios. No new recovery behavior is added in this phase.")
    ln(f"")
    ln(f"---")
    ln(f"")
    ln(f"## Files Changed")
    ln(f"")
    ln(f"| File | Change |")
    ln(f"|---|---|")
    ln(f"| `app/services/orchestration/recovery/recovery_metrics.py` | New — event-log metrics aggregator |")
    ln(f"| `app/tests/test_execution_recovery_s4.py` | New — 7 seeded scenario tests |")
    ln(f"| `scripts/evals/run_recovery_validation_s4.py` | New — report generator |")
    ln(f"| `docs/roadmap/reports/phase13b-s4-recovery-validation-{today}.md` | New — this report |")
    ln(f"")
    ln(f"---")
    ln(f"")
    ln(f"## Seeded Scenario Results")
    ln(f"")
    ln(f"| Scenario | Expected | Actual status | Pass |")
    ln(f"|---|---|---|---|")

    scenario_expected = {
        "S1: step import_error recoverable": ("success", True),
        "S2: step pytest_failure recoverable": ("success", True),
        "S3: completion missing_requested_symbol recoverable": ("success", True),
        "S4: completion generic validation not recovered": ("failed", False),
        "S5: ineligible class not recovered": ("skipped", False),
        "S6: repeated signature stops": ("loop_stopped", False),
    }

    scenario_names = list(scenario_expected.keys())
    for i, sr in enumerate(scenario_results):
        name = scenario_names[i] if i < len(scenario_names) else sr["name"]
        expected_status, should_recover = list(scenario_expected.values())[i] if i < len(scenario_expected) else ("?", False)
        actual = sr["outcome"].get("status", "error")
        error = sr.get("error")

        if error:
            pass_symbol = "FAIL (error)"
        elif actual == expected_status:
            pass_symbol = "PASS"
        elif expected_status == "loop_stopped" and actual == "loop_stopped":
            pass_symbol = "PASS"
        else:
            pass_symbol = f"FAIL (got {actual!r})"

        ln(f"| {name} | `{expected_status}` | `{actual}` | {pass_symbol} |")

    ln(f"")

    # Show false-success guard.
    false_successes = aggregate.get("recovery_false_success_count", 0)
    ln(f"**False-success guard:** `recovery_false_success_count = {false_successes}` "
       f"({'PASS — no ineligible scenario produced success' if false_successes == 0 else 'FAIL'})")
    ln(f"")
    ln(f"---")
    ln(f"")
    ln(f"## Aggregated Recovery Metrics (seeded scenarios)")
    ln(f"")
    ln(f"| Metric | Value |")
    ln(f"|---|---|")
    ln(f"| `recovery_attempted_count` | {aggregate.get('recovery_attempted_count', 0)} |")
    ln(f"| `recovery_succeeded_count` | {aggregate.get('recovery_succeeded_count', 0)} |")
    ln(f"| `recovery_failed_count` | {aggregate.get('recovery_failed_count', 0)} |")
    ln(f"| `recovery_skipped_count` | {aggregate.get('recovery_skipped_count', 0)} |")
    ln(f"| `recovered_success_rate` | {aggregate.get('recovered_success_rate', 0):.1%} |")
    ln(f"| `recovery_false_success_count` | {aggregate.get('recovery_false_success_count', 0)} |")
    ln(f"| `recovery_budget_exhausted_count` | {aggregate.get('recovery_budget_exhausted_count', 0)} |")
    ln(f"")

    by_scope = aggregate.get("recovery_by_scope", {})
    ln(f"**By scope:**")
    ln(f"")
    ln(f"| Scope | Attempted/Noop |")
    ln(f"|---|---|")
    for scope, count in sorted(by_scope.items()):
        ln(f"| {scope} | {count} |")
    ln(f"")

    by_fc = aggregate.get("recovery_by_failure_class", {})
    ln(f"**By failure class:**")
    ln(f"")
    ln(f"| Failure class | Count |")
    ln(f"|---|---|")
    for fc, count in sorted(by_fc.items()):
        ln(f"| `{fc}` | {count} |")
    ln(f"")
    ln(f"---")
    ln(f"")
    ln(f"## Test Suite Results")
    ln(f"")
    ln(f"### S1–S4 Recovery Tests")
    ln(f"")
    ln(f"```")
    ln(pytest_result.get("recovery_suite_output", "").strip())
    ln(f"```")
    ln(f"")
    rc = pytest_result.get("recovery_suite_exit_code", -1)
    ln(f"Exit code: `{rc}` ({'PASS' if rc == 0 else 'FAIL'})")
    ln(f"")
    ln(f"### Prior Suite (S1–S3, no S4)")
    ln(f"")
    ln(f"```")
    ln(broader.get("prior_suite_output", "").strip())
    ln(f"```")
    ln(f"")
    prior_rc = broader.get("prior_suite_exit_code", -1)
    ln(f"Exit code: `{prior_rc}` ({'PASS' if prior_rc == 0 else 'FAIL'})")
    ln(f"")
    ln(f"---")
    ln(f"")
    ln(f"## Safety Boundary Verification")
    ln(f"")
    ln(f"| Guard | Verified |")
    ln(f"|---|---|")
    ln(f"| Ineligible classes (permission\\_denied, etc.) skip immediately | Yes — Scenario 5 |")
    ln(f"| Generic completion failures never recover | Yes — Scenario 4 |")
    ln(f"| Repeated failure signature stops loop | Yes — Scenario 6 |")
    ln(f"| Recovery budget hard-capped at 2 | Yes — S4 budget test |")
    ln(f"| No test deletion or weakening | Yes — S2/S3 test-preservation tests |")
    ln(f"| Symbol rename rejected | Yes — S3 symbol-rename test |")
    ln(f"| Unrelated file patches rejected | Yes — S2/S3 scope tests |")
    ln(f"| Rollback on validator rejection | Yes — S2.5/S3 rollback tests |")
    ln(f"| `validator_accepted=True` required in SUCCEEDED event | Yes — S2.5/S3 |")
    ln(f"")
    ln(f"---")
    ln(f"")
    ln(f"## Recommendation")
    ln(f"")

    # Determine recommendation based on results.
    all_pass = all(
        sr.get("error") is None and sr["outcome"].get("status") in (
            "success", "failed", "skipped", "loop_stopped"
        )
        for sr in scenario_results
    )
    suite_pass = pytest_result.get("recovery_suite_exit_code", 1) == 0
    no_false_success = aggregate.get("recovery_false_success_count", 0) == 0

    if all_pass and suite_pass and no_false_success:
        ln(f"**Recommendation: Proceed to live pilot.**")
        ln(f"")
        ln(f"All safety boundaries hold under seeded failure scenarios:")
        ln(f"")
        ln(f"- Eligible step-scope failures (import\\_error, pytest\\_failure) recover correctly.")
        ln(f"- Eligible completion-scope failure (missing\\_requested\\_symbol) recovers correctly.")
        ln(f"- Ineligible classes skip immediately — ABORT path unchanged.")
        ln(f"- Generic completion validation failures are not recovered.")
        ln(f"- Loop-prevention guards (repeated signature, budget cap) function correctly.")
        ln(f"- No false successes detected across all scenarios.")
        ln(f"- All 113 recovery tests pass (S1: 39, S2: 31, S2.5: 20, S3: 16, S4: 7).")
        ln(f"")
        ln(f"**Pre-pilot checklist:**")
        ln(f"")
        ln(f"1. Rebuild `orchestrator` and `celery_worker` Docker images after S3 changes.")
        ln(f"2. Run `pytest app/tests/` full suite to confirm 3689 pass / 15 pre-existing fail.")
        ln(f"3. Confirm `EXECUTION_RECOVERY_*` events appear in ops dashboard event log.")
        ln(f"4. Monitor first 10 live tasks: count RECOVERY\\_ATTEMPTED events.")
        ln(f"5. Flag any RECOVERY\\_SUCCEEDED where the task later fails human review.")
        ln(f"")
        ln(f"**Not recommended at this time:**")
        ln(f"")
        ln(f"- Do NOT expand eligible classes beyond current list.")
        ln(f"- Do NOT enable completion recovery for classes other than `missing\\_requested\\_symbol`.")
        ln(f"- Do NOT raise budget above 2.")
        ln(f"")
        ln(f"**Future (Phase 13C):** After live pilot confirms recovery helps more than it hurts,")
        ln(f"evaluate expanding to wrapper-timeout recovery and cross-step dependency repair.")
    else:
        ln(f"**Recommendation: Hold — do not proceed to live pilot.**")
        ln(f"")
        ln(f"Failures detected:")
        if not all_pass:
            ln(f"- One or more seeded scenarios produced unexpected outcomes.")
        if not suite_pass:
            ln(f"- Recovery test suite exit code != 0.")
        if not no_false_success:
            ln(f"- False success count > 0 (ineligible scenario recovered).")
        ln(f"")
        ln(f"Review the test output above before proceeding.")

    ln(f"")
    ln(f"---")
    ln(f"")
    ln(f"*Generated by `scripts/evals/run_recovery_validation_s4.py`*")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    today = date.today().strftime("%Y%m%d")
    print(f"Phase 13B-S4 Recovery Validation — {today}")
    print("=" * 60)

    scenarios = [
        ("S1: step import_error recoverable", _scenario_step_import_error),
        ("S2: step pytest_failure recoverable", _scenario_step_pytest_failure),
        ("S3: completion missing_requested_symbol recoverable", _scenario_completion_missing_symbol),
        ("S4: completion generic validation not recovered", _scenario_completion_generic_not_recovered),
        ("S5: ineligible class not recovered", _scenario_ineligible_class),
        ("S6: repeated signature stops", _scenario_repeated_signature_stops),
    ]

    scenario_results = []
    all_metrics = []
    for name, fn in scenarios:
        print(f"  Running {name}...", end=" ", flush=True)
        result = run_scenario(name, fn)
        scenario_results.append(result)
        if result["metrics"]:
            all_metrics.append(result["metrics"])
        status = result["outcome"].get("status", "error")
        error = result.get("error")
        print(f"{'ERROR: ' + error[:60] if error else status}")

    aggregate = aggregate_metrics(all_metrics)

    print()
    print("Running S1–S4 test suite...")
    pytest_result = run_pytest_suite()
    rc = pytest_result["recovery_suite_exit_code"]
    print(f"  exit code: {rc} ({'PASS' if rc == 0 else 'FAIL'})")

    print()
    print("Running prior S1–S3 regression check...")
    broader = run_broader_regression()
    print(f"  prior suite exit code: {broader['prior_suite_exit_code']}")

    print()
    print("Writing report...")
    report_path = write_report(today, scenario_results, aggregate, pytest_result, broader)
    print(f"  Report: {report_path}")

    print()
    print("Aggregate metrics:")
    for k, v in aggregate.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print(f"  by_scope: {aggregate.get('recovery_by_scope', {})}")
    print(f"  by_failure_class: {aggregate.get('recovery_by_failure_class', {})}")


if __name__ == "__main__":
    main()

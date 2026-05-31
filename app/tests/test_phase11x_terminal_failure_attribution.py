from types import SimpleNamespace

from app.services.orchestration.phases.planning_support import (
    _PlanningRetryState,
    _planning_root_cause_from_immediate_repair_issues,
    _planning_root_cause_from_plan_verdict,
    _record_planning_root_cause,
    _terminal_planning_root_cause,
)


def test_terminal_root_cause_prefers_latest_validator_blocker():
    retry_state = _PlanningRetryState()
    _record_planning_root_cause(retry_state, "invalid_python")

    verdict = SimpleNamespace(
        reasons=["Plan is missing verification commands for steps: [3]"],
        details={"missing_verification_steps": [3]},
    )
    _record_planning_root_cause(
        retry_state,
        _planning_root_cause_from_plan_verdict(verdict),
    )

    assert _terminal_planning_root_cause(retry_state) == "missing_verification"


def test_terminal_root_cause_preserves_existing_at_circuit_breaker():
    retry_state = _PlanningRetryState()
    _record_planning_root_cause(retry_state, "stale_replace")

    assert _terminal_planning_root_cause(retry_state) == "stale_replace"


def test_terminal_root_cause_uses_retry_exhausted_when_unknown():
    retry_state = _PlanningRetryState()

    assert _terminal_planning_root_cause(retry_state) == "retry_exhausted"


def test_root_cause_maps_immediate_repair_issue_buckets():
    assert (
        _planning_root_cause_from_immediate_repair_issues(
            {"stale_replace_ops_steps": [2]}
        )
        == "stale_replace"
    )
    assert (
        _planning_root_cause_from_immediate_repair_issues(
            {"weak_verification_steps": [3]}
        )
        == "missing_verification"
    )

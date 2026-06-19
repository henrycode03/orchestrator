"""Phase 13B-S4: Recovery metrics aggregation from orchestration event logs.

Reads EXECUTION_RECOVERY_* events from an event log and produces a structured
metrics dict suitable for reporting and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.orchestration.events.event_types import EventType

_RECOVERY_EVENT_TYPES = {
    EventType.EXECUTION_RECOVERY_ATTEMPTED,
    EventType.EXECUTION_RECOVERY_SUCCEEDED,
    EventType.EXECUTION_RECOVERY_FAILED,
    EventType.EXECUTION_RECOVERY_SKIPPED,
}


def collect_recovery_metrics(
    project_dir: Any,
    session_id: int,
    task_id: int,
) -> Dict[str, Any]:
    """Read event log for one run and return recovery metrics dict."""
    from app.services.orchestration.state.persistence import read_orchestration_events

    events = read_orchestration_events(
        project_dir, session_id=session_id, task_id=task_id
    )
    return _tally(events)


def aggregate_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge metrics from multiple runs into a single aggregate dict."""
    totals: Dict[str, Any] = {
        "recovery_attempted_count": 0,
        "recovery_succeeded_count": 0,
        "recovery_failed_count": 0,
        "recovery_skipped_count": 0,
        "recovery_budget_exhausted_count": 0,
        "recovery_false_success_count": 0,
        "recovery_by_scope": {},
        "recovery_by_failure_class": {},
    }
    for m in all_metrics:
        totals["recovery_attempted_count"] += m.get("recovery_attempted_count", 0)
        totals["recovery_succeeded_count"] += m.get("recovery_succeeded_count", 0)
        totals["recovery_failed_count"] += m.get("recovery_failed_count", 0)
        totals["recovery_skipped_count"] += m.get("recovery_skipped_count", 0)
        totals["recovery_budget_exhausted_count"] += m.get(
            "recovery_budget_exhausted_count", 0
        )
        totals["recovery_false_success_count"] += m.get(
            "recovery_false_success_count", 0
        )
        for scope, count in m.get("recovery_by_scope", {}).items():
            totals["recovery_by_scope"][scope] = (
                totals["recovery_by_scope"].get(scope, 0) + count
            )
        for fc, count in m.get("recovery_by_failure_class", {}).items():
            totals["recovery_by_failure_class"][fc] = (
                totals["recovery_by_failure_class"].get(fc, 0) + count
            )

    total_terminal = (
        totals["recovery_succeeded_count"]
        + totals["recovery_failed_count"]
        + totals["recovery_skipped_count"]
    )
    totals["recovered_success_rate"] = (
        round(totals["recovery_succeeded_count"] / total_terminal, 3)
        if total_terminal > 0
        else 0.0
    )
    totals["total_terminal_outcomes"] = total_terminal
    return totals


def _tally(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "recovery_attempted_count": 0,
        "recovery_succeeded_count": 0,
        "recovery_failed_count": 0,
        "recovery_skipped_count": 0,
        "recovery_budget_exhausted_count": 0,
        "recovery_false_success_count": 0,
        "recovery_by_scope": {},
        "recovery_by_failure_class": {},
    }
    for event in events:
        et = event.get("event_type", "")
        details = event.get("details", {})
        scope = details.get("scope", "unknown")
        fc = details.get("failure_class", "unknown")

        if et == EventType.EXECUTION_RECOVERY_ATTEMPTED:
            metrics["recovery_attempted_count"] += 1
            metrics["recovery_by_scope"][scope] = (
                metrics["recovery_by_scope"].get(scope, 0) + 1
            )
            metrics["recovery_by_failure_class"][fc] = (
                metrics["recovery_by_failure_class"].get(fc, 0) + 1
            )
        elif et == EventType.EXECUTION_RECOVERY_SUCCEEDED:
            metrics["recovery_succeeded_count"] += 1
        elif et == EventType.EXECUTION_RECOVERY_FAILED:
            metrics["recovery_failed_count"] += 1
            if details.get("budget_exhausted"):
                metrics["recovery_budget_exhausted_count"] += 1
        elif et == EventType.EXECUTION_RECOVERY_SKIPPED:
            metrics["recovery_skipped_count"] += 1
            metrics["recovery_by_scope"][scope] = (
                metrics["recovery_by_scope"].get(scope, 0) + 1
            )
            metrics["recovery_by_failure_class"][fc] = (
                metrics["recovery_by_failure_class"].get(fc, 0) + 1
            )

    total_terminal = (
        metrics["recovery_succeeded_count"]
        + metrics["recovery_failed_count"]
        + metrics["recovery_skipped_count"]
    )
    metrics["recovered_success_rate"] = (
        round(metrics["recovery_succeeded_count"] / total_terminal, 3)
        if total_terminal > 0
        else 0.0
    )
    metrics["total_terminal_outcomes"] = total_terminal
    return metrics

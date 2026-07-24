"""Event, telemetry, and observability helpers for orchestration."""

from __future__ import annotations

from typing import Any, Mapping

from .event_types import EventType, is_known_event_type
from .observability import build_trace_export
from .telemetry import emit_phase_event, record_phase_event


def report_candidate_recovery_event(
    request: Any,
    event_type: str,
    candidate: Any = None,
    details: Mapping[str, Any] | None = None,
) -> str:
    """Adapt planning recovery facts to the orchestration event journal."""

    from app.services.orchestration.state.persistence import append_orchestration_event

    payload: dict[str, Any] = dict(details or {})
    if candidate is not None:
        payload.update(candidate.to_dict())
    event = append_orchestration_event(
        project_dir=request.project_dir,
        session_id=request.session_id,
        task_id=request.task_id,
        event_type=event_type,
        parent_event_id=request.parent_event_id,
        details=payload,
    )
    return str(event.get("event_id") or "")


# The event package owns this compatibility registration.  Planning recovery
# remains importable without this package and accepts an explicit reporter.
from app.services.planning.candidate_recovery import (  # noqa: E402
    register_candidate_recovery_event_reporter,
)

register_candidate_recovery_event_reporter(report_candidate_recovery_event)

__all__ = [
    "EventType",
    "is_known_event_type",
    "build_trace_export",
    "emit_phase_event",
    "record_phase_event",
    "report_candidate_recovery_event",
]

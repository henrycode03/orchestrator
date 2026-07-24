"""Planning task-dispatch contract owned by the planning service boundary.

The planning service only publishes this protocol.  The Celery adapter is
registered by the task layer, keeping worker concerns out of planning-domain
modules while preserving the existing session-processing entry point.
"""

from __future__ import annotations

from typing import Protocol


class PlanningTaskDispatcher(Protocol):
    """Publish one fenced planning-session processing request."""

    def dispatch(
        self,
        *,
        session_id: int,
        generation_id: str,
        owner_token: str,
        task_id: str,
    ) -> None:
        """Publish the request without changing its worker semantics."""


_registered_dispatcher: PlanningTaskDispatcher | None = None


def register_planning_task_dispatcher(
    dispatcher: PlanningTaskDispatcher | None,
) -> None:
    """Install the application-composition dispatcher used by default.

    Tests and synchronous callers can pass a dispatcher directly to
    ``PlanningSessionService``.  The task layer registers the production
    adapter when the application imports its planning routes.
    """

    global _registered_dispatcher
    _registered_dispatcher = dispatcher


def get_planning_task_dispatcher() -> PlanningTaskDispatcher | None:
    """Return the currently registered task-layer adapter, if any."""

    return _registered_dispatcher


__all__ = [
    "PlanningTaskDispatcher",
    "get_planning_task_dispatcher",
    "register_planning_task_dispatcher",
]

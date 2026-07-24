"""Celery adapter for the planning-session dispatch contract."""

from __future__ import annotations

from app.celery_app import celery_app
from app.services.planning.planning_dispatch import (
    PlanningTaskDispatcher,
    register_planning_task_dispatcher,
)

PLANNING_SESSION_TASK_NAME = "app.tasks.planning_tasks.advance_planning_session"


class CeleryPlanningTaskDispatcher:
    """Publish the existing planning task without importing its task module."""

    def dispatch(
        self,
        *,
        session_id: int,
        generation_id: str,
        owner_token: str,
        task_id: str,
    ) -> None:
        celery_app.send_task(
            PLANNING_SESSION_TASK_NAME,
            args=(session_id, generation_id, owner_token),
            task_id=task_id,
        )


planning_task_dispatcher: PlanningTaskDispatcher = CeleryPlanningTaskDispatcher()


def ensure_planning_task_dispatcher() -> PlanningTaskDispatcher:
    """Register and return the application-level Celery adapter."""

    register_planning_task_dispatcher(planning_task_dispatcher)
    return planning_task_dispatcher


__all__ = [
    "CeleryPlanningTaskDispatcher",
    "PLANNING_SESSION_TASK_NAME",
    "ensure_planning_task_dispatcher",
    "planning_task_dispatcher",
]

"""Background tasks for interactive planning sessions."""

from __future__ import annotations

import logging
import time

from app.celery_app import celery_app
from app.database import get_db_session
from app.services.planning.planning_dispatch import register_planning_task_dispatcher
from app.services.planning.planning_session_service import PlanningSessionService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=15)
def advance_planning_session(
    self,
    session_id: int,
    generation_id: str | None = None,
    owner_token: str | None = None,
) -> dict[str, object]:
    started_at = time.monotonic()
    db = get_db_session()
    retrying = False
    task_status = "unknown"
    try:
        if not generation_id or not owner_token:
            task_status = "stale_owner"
            return {
                "status": "stale_owner",
                "session_id": session_id,
                "generation_id": generation_id or "",
                "reason": "legacy_task_arguments",
            }
        service = PlanningSessionService(db)
        session = service.process_session(
            session_id,
            generation_id,
            owner_token,
            processing_task_id=getattr(self.request, "id", None),
        )
        if isinstance(session, dict):
            task_status = str(session.get("status") or "unknown")
            return session
        if not session:
            task_status = "skipped"
            return {"status": "skipped", "session_id": session_id}
        task_status = str(session.status or "unknown")
        return {
            "status": session.status,
            "session_id": session.id,
            "project_id": session.project_id,
        }
    except Exception as exc:
        task_status = "exception"
        logger.exception("Planning background task failed for session %s", session_id)
        retrying = True
        raise self.retry(
            exc=exc,
            args=(session_id, generation_id, owner_token),
        )
    finally:
        if not retrying and generation_id and owner_token:
            PlanningSessionService(db).release_processing_task(
                session_id,
                generation_id,
                owner_token,
                getattr(self.request, "id", None),
            )
        logger.info(
            "[PHASE28RV_TIMING] celery_task=advance_planning_session session_id=%s "
            "status=%s total_celery_task_seconds=%.3f",
            session_id,
            task_status,
            time.monotonic() - started_at,
        )
        db.close()


class _RegisteredPlanningTaskDispatcher:
    """Compatibility adapter that preserves the task object's publish path."""

    def dispatch(
        self,
        *,
        session_id: int,
        generation_id: str,
        owner_token: str,
        task_id: str,
    ) -> None:
        advance_planning_session.apply_async(
            args=(session_id, generation_id, owner_token),
            task_id=task_id,
        )


# Importing this module still registers exactly one task.  It additionally
# supplies the task-layer adapter used by direct synchronous service tests.
register_planning_task_dispatcher(_RegisteredPlanningTaskDispatcher())

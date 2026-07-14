"""OpenClaw runtime adapter."""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from app.services.agents.openclaw_service import OpenClawSessionService
from app.services.agents.runtime_configuration import RuntimeConfiguration


def create_runtime(
    db: Session,
    session_id: Optional[int],
    task_id: Optional[int] = None,
    *,
    use_demo_mode: Optional[bool] = None,
    runtime_configuration: RuntimeConfiguration | None = None,
) -> OpenClawSessionService:
    """Instantiate the OpenClaw-backed orchestration runtime."""

    return OpenClawSessionService(
        db,
        session_id,
        task_id,
        use_demo_mode=use_demo_mode,
        runtime_configuration=runtime_configuration,
    )

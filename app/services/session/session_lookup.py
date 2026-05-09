"""Shared session lookup helpers."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session as DBSession

from app.models import Session as SessionModel


def get_session_or_404(
    db: DBSession,
    session_id: int,
    *,
    include_deleted: bool = False,
) -> SessionModel:
    query = db.query(SessionModel).filter(SessionModel.id == session_id)
    if not include_deleted:
        query = query.filter(SessionModel.deleted_at.is_(None))
    session = query.first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

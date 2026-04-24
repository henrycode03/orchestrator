"""Tests for GET /sessions/{id}/tasks/{task_id}/events — orchestration event journal endpoint."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.models import Project, Session as SessionModel


def _make_project(db, *, workspace_path="/tmp/evt_test"):
    project = Project(name="Events Test", workspace_path=workspace_path)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def _make_session(db, project, *, status="stopped"):
    session = SessionModel(
        project_id=project.id,
        name="Events Session",
        description="test",
        status=status,
        is_active=False,
        execution_mode="manual",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _write_events(
    workspace_path: str, session_id: int, task_id: int, events: list[dict]
) -> None:
    events_dir = Path(workspace_path) / ".openclaw" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    log_path = events_dir / f"session_{session_id}_task_{task_id}.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")


def test_events_endpoint_returns_empty_list_when_no_journal(
    authenticated_client, db_session
):
    with tempfile.TemporaryDirectory() as tmpdir:
        project = _make_project(db_session, workspace_path=tmpdir)
        session = _make_session(db_session, project)

        resp = authenticated_client.get(
            f"/api/v1/sessions/{session.id}/tasks/999/events"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["events"] == []
        assert body["session_id"] == session.id
        assert body["task_id"] == 999


def test_events_endpoint_returns_all_events(authenticated_client, db_session):
    with tempfile.TemporaryDirectory() as tmpdir:
        project = _make_project(db_session, workspace_path=tmpdir)
        session = _make_session(db_session, project)
        task_id = 7

        sample_events = [
            {
                "event_type": "phase_started",
                "session_id": session.id,
                "task_id": task_id,
                "details": {},
            },
            {
                "event_type": "step_finished",
                "session_id": session.id,
                "task_id": task_id,
                "details": {},
            },
            {
                "event_type": "task_completed",
                "session_id": session.id,
                "task_id": task_id,
                "details": {},
            },
        ]
        _write_events(tmpdir, session.id, task_id, sample_events)

        resp = authenticated_client.get(
            f"/api/v1/sessions/{session.id}/tasks/{task_id}/events"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["events"]) == 3
        types = [e["event_type"] for e in body["events"]]
        assert types == ["phase_started", "step_finished", "task_completed"]


def test_events_endpoint_filters_by_event_type(authenticated_client, db_session):
    with tempfile.TemporaryDirectory() as tmpdir:
        project = _make_project(db_session, workspace_path=tmpdir)
        session = _make_session(db_session, project)
        task_id = 3

        sample_events = [
            {
                "event_type": "phase_started",
                "session_id": session.id,
                "task_id": task_id,
                "details": {},
            },
            {
                "event_type": "validation_result",
                "session_id": session.id,
                "task_id": task_id,
                "details": {"passed": False},
            },
            {
                "event_type": "phase_finished",
                "session_id": session.id,
                "task_id": task_id,
                "details": {},
            },
            {
                "event_type": "validation_result",
                "session_id": session.id,
                "task_id": task_id,
                "details": {"passed": True},
            },
        ]
        _write_events(tmpdir, session.id, task_id, sample_events)

        resp = authenticated_client.get(
            f"/api/v1/sessions/{session.id}/tasks/{task_id}/events",
            params={"event_type": "validation_result"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["events"]) == 2
        assert all(e["event_type"] == "validation_result" for e in body["events"])


def test_events_endpoint_rejects_unknown_event_type(authenticated_client, db_session):
    with tempfile.TemporaryDirectory() as tmpdir:
        project = _make_project(db_session, workspace_path=tmpdir)
        session = _make_session(db_session, project)

        resp = authenticated_client.get(
            f"/api/v1/sessions/{session.id}/tasks/3/events",
            params={"event_type": "validation_reslt"},
        )

        assert resp.status_code == 400
        assert "Unknown event_type" in resp.json()["detail"]


def test_events_endpoint_returns_404_for_nonexistent_session(authenticated_client):
    resp = authenticated_client.get("/api/v1/sessions/99999/tasks/1/events")
    assert resp.status_code == 404


def test_events_endpoint_requires_auth(api_client, db_session):
    resp = api_client.get("/api/v1/sessions/1/tasks/1/events")
    assert resp.status_code == 401

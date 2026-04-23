"""Tests for admin diagnostics endpoint and session lifecycle audit logging."""

from __future__ import annotations

import json

import pytest

from app.models import LogEntry


def test_diagnostics_endpoint_returns_expected_shape(authenticated_client):
    resp = authenticated_client.get("/api/v1/admin/diagnostics")
    assert resp.status_code == 200
    body = resp.json()
    assert "overall_status" in body
    assert "backends" in body
    assert "queue" in body
    assert "sessions" in body
    assert "recent_audit_events" in body
    assert "checked_at" in body


def test_diagnostics_backends_list_includes_registered_providers(authenticated_client):
    resp = authenticated_client.get("/api/v1/admin/diagnostics")
    body = resp.json()
    names = [b["name"] for b in body["backends"]]
    assert "local_openclaw" in names
    assert "openai_responses_api" in names


def test_diagnostics_overall_status_is_valid_string(authenticated_client):
    resp = authenticated_client.get("/api/v1/admin/diagnostics")
    body = resp.json()
    assert body["overall_status"] in ("healthy", "warning", "degraded")


def test_diagnostics_queue_shape(authenticated_client):
    resp = authenticated_client.get("/api/v1/admin/diagnostics")
    body = resp.json()
    q = body["queue"]
    assert "status" in q
    assert "active_tasks" in q
    assert "worker_count" in q


def test_diagnostics_sessions_shape(authenticated_client):
    resp = authenticated_client.get("/api/v1/admin/diagnostics")
    body = resp.json()
    s = body["sessions"]
    assert "by_status" in s
    assert "failed_last_24h" in s
    assert "recent_failures" in s


def test_diagnostics_requires_auth(api_client):
    resp = api_client.get("/api/v1/admin/diagnostics")
    assert resp.status_code in (401, 403)


def test_diagnostics_recent_audit_events_only_returns_structured(
    db_session, authenticated_client
):
    db_session.add(
        LogEntry(
            level="INFO",
            message="unstructured log with no metadata",
            log_metadata=None,
        )
    )
    db_session.add(
        LogEntry(
            level="INFO",
            message="structured settings change",
            log_metadata=json.dumps(
                {
                    "event_type": "system_settings_updated",
                    "actor_email": "admin@test.com",
                    "changes": {},
                }
            ),
        )
    )
    db_session.commit()

    resp = authenticated_client.get("/api/v1/admin/diagnostics")
    body = resp.json()
    for event in body["recent_audit_events"]:
        assert "event_type" in event
        assert event["event_type"]


def test_session_lifecycle_audit_log_has_event_type(db_session):
    """Verify log entries written with event_type are stored and queryable."""
    for event_type in (
        "session_started",
        "session_stopped",
        "session_paused",
        "session_resumed",
        "session_start_failed",
        "session_stop_failed",
        "session_pause_failed",
        "session_resume_failed",
    ):
        db_session.add(
            LogEntry(
                level="INFO",
                message=f"test {event_type}",
                log_metadata=json.dumps({"event_type": event_type, "session_id": 1}),
            )
        )
    db_session.commit()

    rows = db_session.query(LogEntry).filter(LogEntry.log_metadata.isnot(None)).all()
    found_types = set()
    for row in rows:
        meta = json.loads(row.log_metadata)
        if "event_type" in meta:
            found_types.add(meta["event_type"])

    for expected in (
        "session_started",
        "session_stopped",
        "session_paused",
        "session_resumed",
    ):
        assert expected in found_types

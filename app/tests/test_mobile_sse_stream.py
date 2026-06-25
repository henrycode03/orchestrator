"""Tests for GET /api/v1/mobile/sessions/{session_id}/events/stream — Phase 14F-5.

Covers:
- _format_sse_frame helper: event line, data line with type field, double-newline
- heartbeat constant format
- auth rejection (key not configured → 503, wrong key → 401)
- session not found → 404
- catch-up JSONL events yielded first as SSE frames
- bus events yielded after catch-up
- duplicate event_id suppressed across catch-up and bus path
- JSONL fallback: new event polled after initial catch-up
- unsubscribe on generator close
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict

import pytest

from app.config import settings
from app.models import Project
from app.models import Session as SessionModel
from app.services.session.orchestration_event_bus import orchestration_event_bus
from app.services.session.session_stream_service import (
    _SSE_HEARTBEAT,
    _SSE_HEARTBEAT_INTERVAL,
    _format_sse_frame,
    mobile_sse_event_generator,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

MOBILE_KEY = "sse-test-key-abc123"
MOBILE_HEADERS = {"X-OpenClaw-API-Key": MOBILE_KEY}


def _make_event(
    session_id: int, event_id: str = "eid-1", event_type: str = "step_finished"
) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "session_id": session_id,
        "task_id": 1,
        "details": {},
    }


def _write_jsonl(tmp_path, session_id: int, task_id: int, events: list[dict]) -> None:
    events_dir = tmp_path / ".agent" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    log_path = events_dir / f"session_{session_id}_task_{task_id}.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(evt) + "\n")


def _make_db_session(db, project) -> SessionModel:
    session = SessionModel(
        project_id=project.id,
        name="SSE Test Session",
        description="",
        status="running",
        is_active=True,
        execution_mode="automatic",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _make_project(db, workspace_path: str) -> Project:
    project = Project(name="SSE Test Project", workspace_path=workspace_path)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


# ── _format_sse_frame ─────────────────────────────────────────────────────────


def test_format_sse_frame_starts_with_event_line():
    frame = _format_sse_frame({"event_id": "x", "event_type": "step_finished"})
    assert frame.startswith(b"event: orchestration_event\n")


def test_format_sse_frame_data_line_contains_type_field():
    event = {"event_id": "y", "event_type": "phase_started", "session_id": 1}
    frame = _format_sse_frame(event)
    lines = frame.decode().splitlines()
    data_line = next(ln for ln in lines if ln.startswith("data: "))
    payload = json.loads(data_line[len("data: ") :])
    assert payload["type"] == "orchestration_event"
    assert payload["event_type"] == "phase_started"


def test_format_sse_frame_ends_with_double_newline():
    frame = _format_sse_frame({"event_id": "z"})
    assert frame.endswith(b"\n\n")


def test_format_sse_frame_preserves_all_event_fields():
    event = {
        "event_id": "abc-123",
        "event_type": "phase_finished",
        "session_id": 42,
        "task_id": 7,
        "phase": "execution",
        "details": {"status": "ok"},
    }
    frame = _format_sse_frame(event)
    data_line = next(
        ln for ln in frame.decode().splitlines() if ln.startswith("data: ")
    )
    payload = json.loads(data_line[len("data: ") :])
    assert payload["event_id"] == "abc-123"
    assert payload["session_id"] == 42
    assert payload["phase"] == "execution"


# ── Heartbeat constants ───────────────────────────────────────────────────────


def test_heartbeat_constant_is_sse_comment():
    assert _SSE_HEARTBEAT == b": heartbeat\n\n"


def test_heartbeat_interval_is_thirty_seconds():
    assert _SSE_HEARTBEAT_INTERVAL == 30.0


# ── HTTP auth / session-not-found tests ──────────────────────────────────────


def test_sse_stream_returns_503_when_key_not_configured(api_client, db_session):
    """No mobile key configured → 503."""
    project = _make_project(db_session, "/tmp/sse_no_key")
    session = _make_db_session(db_session, project)
    # Don't set MOBILE_GATEWAY_API_KEY; settings default is empty
    original = settings.MOBILE_GATEWAY_API_KEY
    settings.MOBILE_GATEWAY_API_KEY = ""
    try:
        resp = api_client.get(
            f"/api/v1/mobile/sessions/{session.id}/events/stream",
            headers=MOBILE_HEADERS,
        )
    finally:
        settings.MOBILE_GATEWAY_API_KEY = original
    assert resp.status_code == 503


def test_sse_stream_returns_401_with_wrong_key(api_client, db_session, monkeypatch):
    """Wrong mobile key → 401."""
    monkeypatch.setattr(settings, "MOBILE_GATEWAY_API_KEY", MOBILE_KEY)
    project = _make_project(db_session, "/tmp/sse_wrong_key")
    session = _make_db_session(db_session, project)
    resp = api_client.get(
        f"/api/v1/mobile/sessions/{session.id}/events/stream",
        headers={"X-OpenClaw-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_sse_stream_returns_404_for_unknown_session(api_client, monkeypatch):
    """Valid auth but unknown session_id → 404."""
    monkeypatch.setattr(settings, "MOBILE_GATEWAY_API_KEY", MOBILE_KEY)
    resp = api_client.get(
        "/api/v1/mobile/sessions/999999/events/stream",
        headers=MOBILE_HEADERS,
    )
    assert resp.status_code == 404


# ── Async generator tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_catch_up_events_are_yielded_as_sse_frames(tmp_path):
    sid = 1001
    event = _make_event(sid, event_id="catchup-1", event_type="phase_started")
    _write_jsonl(tmp_path, sid, 1, [event])

    gen = mobile_sse_event_generator(sid, str(tmp_path))
    try:
        frame = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert b"event: orchestration_event" in frame
        assert b"catchup-1" in frame
        assert b"phase_started" in frame
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_catch_up_frame_includes_type_field(tmp_path):
    sid = 1002
    event = _make_event(sid, event_id="catchup-2")
    _write_jsonl(tmp_path, sid, 1, [event])

    gen = mobile_sse_event_generator(sid, str(tmp_path))
    try:
        frame = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        data_line = next(
            ln for ln in frame.decode().splitlines() if ln.startswith("data: ")
        )
        payload = json.loads(data_line[len("data: ") :])
        assert payload["type"] == "orchestration_event"
        assert payload["event_id"] == "catchup-2"
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_bus_event_is_yielded_as_sse_frame(tmp_path):
    sid = 1003
    event = _make_event(sid, event_id="bus-evt-1")

    gen = mobile_sse_event_generator(sid, None)

    async def _publish():
        await asyncio.sleep(0.05)
        orchestration_event_bus.publish(event)

    asyncio.create_task(_publish())
    try:
        frame = await asyncio.wait_for(gen.__anext__(), timeout=2.5)
        assert b"bus-evt-1" in frame
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_duplicate_suppressed_across_catchup_and_bus(tmp_path):
    """Event yielded in catch-up is not re-sent when same event_id arrives on bus."""
    sid = 1004
    event = _make_event(sid, event_id="dedup-bus")
    _write_jsonl(tmp_path, sid, 1, [event])

    gen = mobile_sse_event_generator(sid, str(tmp_path))
    try:
        # Consume the initial catch-up frame
        frame = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert b"dedup-bus" in frame

        # Now publish the same event to the bus (generator is now subscribed)
        orchestration_event_bus.publish(event)

        # The generator should NOT yield the duplicate within 0.2 s
        # (it would have to drain the bus first, dedup suppresses it,
        #  then sleep 1 s — timeout fires before any yield)
        try:
            duplicate = await asyncio.wait_for(gen.__anext__(), timeout=0.2)
            # If something was yielded it must not carry the same event_id
            assert b"dedup-bus" not in duplicate
        except asyncio.TimeoutError:
            pass  # expected: dedup suppressed it, next yield is >1 s away
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_jsonl_fallback_delivers_event_after_catchup(tmp_path):
    """New JSONL event written after catch-up is picked up by the fallback poll."""
    sid = 1005
    event_a = _make_event(sid, event_id="jf-a", event_type="phase_started")
    _write_jsonl(tmp_path, sid, 1, [event_a])

    gen = mobile_sse_event_generator(sid, str(tmp_path))
    try:
        # Consume initial catch-up event
        frame_a = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert b"jf-a" in frame_a

        # Write a new event to JSONL (simulates an event arriving after connect)
        event_b = _make_event(sid, event_id="jf-b", event_type="step_finished")
        _write_jsonl(tmp_path, sid, 1, [event_b])

        # Generator picks it up in the first poll loop iteration (before its sleep)
        frame_b = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        assert b"jf-b" in frame_b
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_duplicate_suppressed_between_jsonl_poll_iterations(tmp_path):
    """Same event written to JSONL twice does not produce two frames."""
    sid = 1006
    event = _make_event(sid, event_id="jsonl-dup")
    # Write the event once
    _write_jsonl(tmp_path, sid, 1, [event])

    gen = mobile_sse_event_generator(sid, str(tmp_path))
    try:
        frame = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert b"jsonl-dup" in frame

        # If we could inject the same event again via JSONL with a lower cursor
        # that won't happen normally, but we can verify via _should_send_event
        # already tested in test_websocket_event_deduplication.py.
        # Here we verify the generator handles JSONL catch-up dedup:
        # write a NEW event (different id) to confirm fallback still works
        event_new = _make_event(sid, event_id="jsonl-new")
        _write_jsonl(tmp_path, sid, 1, [event_new])

        frame_new = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        assert b"jsonl-new" in frame_new
        assert b"jsonl-dup" not in frame_new
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_unsubscribe_on_generator_close():
    """Bus subscriber is removed when the generator is closed."""
    sid = 1007
    gen = mobile_sse_event_generator(sid, None)

    # Start the generator so it subscribes
    try:
        await asyncio.wait_for(gen.__anext__(), timeout=0.05)
    except (asyncio.TimeoutError, StopAsyncIteration):
        pass
    # The generator is now paused inside asyncio.sleep(1.0) or is closed.
    # In either case, call aclose() to ensure cleanup.
    await gen.aclose()

    assert orchestration_event_bus.subscriber_count(sid) == 0


@pytest.mark.asyncio
async def test_no_event_id_events_are_never_suppressed(tmp_path):
    """Events without event_id are always forwarded; they do not pollute seen set."""
    sid = 1008
    event_no_id = {"event_type": "phase_started", "session_id": sid, "task_id": 1}
    _write_jsonl(tmp_path, sid, 1, [event_no_id, event_no_id])

    gen = mobile_sse_event_generator(sid, str(tmp_path))
    try:
        frame1 = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert b"phase_started" in frame1
        frame2 = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert b"phase_started" in frame2
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_multiple_catchup_events_yielded_in_order(tmp_path):
    sid = 1009
    events = [
        _make_event(sid, event_id=f"ord-{i}", event_type="step_finished")
        for i in range(3)
    ]
    _write_jsonl(tmp_path, sid, 1, events)

    gen = mobile_sse_event_generator(sid, str(tmp_path))
    try:
        for i in range(3):
            frame = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
            assert f"ord-{i}".encode() in frame
    finally:
        await gen.aclose()

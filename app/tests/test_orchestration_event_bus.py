"""Tests for OrchestrationEventBus (Phase 14F-3).

Covers: subscribe/publish/unsubscribe lifecycle, session isolation,
queue overflow, integration with append_orchestration_event, and
WebSocket stream service message shape preservation.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.services.session.orchestration_event_bus import (
    OrchestrationEventBus,
    orchestration_event_bus,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_event(session_id: int, event_type: str = "step_finished") -> dict:
    return {
        "event_id": "test-uuid",
        "session_id": session_id,
        "task_id": 1,
        "event_type": event_type,
        "details": {},
    }


# ── Subscriber receives published event ───────────────────────────────────────


@pytest.mark.asyncio
async def test_subscriber_receives_published_event():
    bus = OrchestrationEventBus()
    q = bus.subscribe(session_id=10)

    event = _make_event(10)
    bus.publish(event)

    received = await asyncio.wait_for(q.get(), timeout=1.0)
    assert received["event_type"] == "step_finished"
    assert received["session_id"] == 10


@pytest.mark.asyncio
async def test_multiple_subscribers_for_same_session_all_receive_event():
    bus = OrchestrationEventBus()
    q1 = bus.subscribe(session_id=20)
    q2 = bus.subscribe(session_id=20)

    event = _make_event(20)
    bus.publish(event)

    r1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    r2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert r1["session_id"] == 20
    assert r2["session_id"] == 20


# ── Session isolation ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscriber_for_other_session_does_not_receive_event():
    bus = OrchestrationEventBus()
    q_target = bus.subscribe(session_id=30)
    q_other = bus.subscribe(session_id=99)

    bus.publish(_make_event(30))

    # Target gets the event
    received = await asyncio.wait_for(q_target.get(), timeout=1.0)
    assert received["session_id"] == 30

    # Other session's queue stays empty
    with pytest.raises(asyncio.QueueEmpty):
        q_other.get_nowait()


# ── Unsubscribe ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = OrchestrationEventBus()
    q = bus.subscribe(session_id=40)

    bus.publish(_make_event(40, "phase_started"))
    await asyncio.wait_for(q.get(), timeout=1.0)  # consume first

    bus.unsubscribe(session_id=40, queue=q)
    bus.publish(_make_event(40, "phase_finished"))

    # Queue should be empty — no delivery after unsubscribe
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()


@pytest.mark.asyncio
async def test_unsubscribe_removes_session_when_no_subscribers_remain():
    bus = OrchestrationEventBus()
    q = bus.subscribe(session_id=41)
    assert bus.subscriber_count(41) == 1

    bus.unsubscribe(session_id=41, queue=q)
    assert bus.subscriber_count(41) == 0


@pytest.mark.asyncio
async def test_unsubscribe_unknown_queue_is_noop():
    bus = OrchestrationEventBus()
    q = bus.subscribe(session_id=42)
    other_q: asyncio.Queue = asyncio.Queue()

    bus.unsubscribe(session_id=42, queue=other_q)  # must not raise
    assert bus.subscriber_count(42) == 1

    bus.unsubscribe(session_id=42, queue=q)


# ── Queue overflow ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_overflow_does_not_crash_publisher():
    bus = OrchestrationEventBus(maxsize=2)
    q = bus.subscribe(session_id=50)

    # Fill the queue to capacity
    bus.publish(_make_event(50, "step_started"))
    bus.publish(_make_event(50, "step_finished"))
    await asyncio.sleep(0)  # yield so call_soon_threadsafe callbacks run

    # Overflow: must not raise
    bus.publish(_make_event(50, "retry_entered"))

    assert q.qsize() == 2  # only the first two fit


# ── Integration with append_orchestration_event ───────────────────────────────


@pytest.mark.asyncio
async def test_append_orchestration_event_writes_jsonl_when_no_subscribers(tmp_path):
    from app.services.orchestration.state.persistence import append_orchestration_event

    project_dir = tmp_path / "bus-no-sub"
    project_dir.mkdir()

    event = append_orchestration_event(
        project_dir=project_dir,
        session_id=60,
        task_id=1,
        event_type="phase_started",
        details={"phase": "planning"},
    )

    log_path = project_dir / ".agent" / "events" / "session_60_task_1.jsonl"
    assert log_path.exists()
    lines = [json.loads(ln) for ln in log_path.read_text().splitlines()]
    assert any(ln["event_type"] == "phase_started" for ln in lines)
    assert event["event_type"] == "phase_started"


@pytest.mark.asyncio
async def test_append_orchestration_event_publishes_after_successful_write(tmp_path):
    from app.services.orchestration.state.persistence import append_orchestration_event

    project_dir = tmp_path / "bus-pub"
    project_dir.mkdir()

    # Subscribe on the singleton bus for session 61
    q = orchestration_event_bus.subscribe(session_id=61)
    try:
        append_orchestration_event(
            project_dir=project_dir,
            session_id=61,
            task_id=2,
            event_type="step_finished",
            details={"step_index": 0},
        )
        # Allow call_soon_threadsafe callbacks to run
        await asyncio.sleep(0)

        bus_event = q.get_nowait()
        assert bus_event["event_type"] == "step_finished"
        assert bus_event["session_id"] == 61
    finally:
        orchestration_event_bus.unsubscribe(61, q)


@pytest.mark.asyncio
async def test_bus_publish_failure_does_not_fail_event_journaling(
    tmp_path, monkeypatch
):
    """Patching publish to raise must not propagate out of append_orchestration_event."""
    from app.services.orchestration.state import persistence

    project_dir = tmp_path / "bus-fail"
    project_dir.mkdir()

    import app.services.session.orchestration_event_bus as bus_module

    original_publish = bus_module.orchestration_event_bus.publish

    def _explode(event):
        raise RuntimeError("simulated bus failure")

    monkeypatch.setattr(bus_module.orchestration_event_bus, "publish", _explode)
    try:
        event = persistence.append_orchestration_event(
            project_dir=project_dir,
            session_id=62,
            task_id=3,
            event_type="task_completed",
            details={},
        )
    finally:
        monkeypatch.setattr(
            bus_module.orchestration_event_bus, "publish", original_publish
        )

    assert event["event_type"] == "task_completed"
    log_path = project_dir / ".agent" / "events" / "session_62_task_3.jsonl"
    assert log_path.exists()


# ── WebSocket message shape preservation ─────────────────────────────────────


@pytest.mark.asyncio
async def test_bus_event_preserves_existing_orchestration_event_shape():
    """Events delivered via bus carry the same fields as JSONL-polled events."""
    bus = OrchestrationEventBus()
    q = bus.subscribe(session_id=70)

    full_event = {
        "event_id": "abc-123",
        "timestamp": "2026-06-25T12:00:00Z",
        "event_type": "phase_finished",
        "session_id": 70,
        "task_id": 5,
        "parent_event_id": None,
        "phase": "task_summary",
        "coordinator": "CompletionCoordinator",
        "details": {"status": "done"},
    }
    bus.publish(full_event)

    received = await asyncio.wait_for(q.get(), timeout=1.0)

    # All envelope fields intact
    assert received["event_id"] == "abc-123"
    assert received["event_type"] == "phase_finished"
    assert received["session_id"] == 70
    assert received["task_id"] == 5
    assert received["phase"] == "task_summary"
    assert received["coordinator"] == "CompletionCoordinator"
    assert received["details"]["status"] == "done"

    # stream service wraps as {"type": "orchestration_event", **event}
    wrapped = {"type": "orchestration_event", **received}
    assert wrapped["type"] == "orchestration_event"
    assert wrapped["event_type"] == "phase_finished"


# ── subscriber_count ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscriber_count_tracks_subscribe_and_unsubscribe():
    bus = OrchestrationEventBus()
    assert bus.subscriber_count(80) == 0

    q1 = bus.subscribe(80)
    assert bus.subscriber_count(80) == 1

    q2 = bus.subscribe(80)
    assert bus.subscriber_count(80) == 2

    bus.unsubscribe(80, q1)
    assert bus.subscriber_count(80) == 1

    bus.unsubscribe(80, q2)
    assert bus.subscriber_count(80) == 0


# ── Publish with non-int session_id is silently ignored ─────────────────────


@pytest.mark.asyncio
async def test_publish_without_int_session_id_is_noop():
    bus = OrchestrationEventBus()
    q = bus.subscribe(90)

    bus.publish({"event_type": "step_finished", "session_id": "not-an-int"})
    bus.publish({"event_type": "step_finished"})  # no session_id key

    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()

    bus.unsubscribe(90, q)

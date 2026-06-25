"""Tests for WebSocket orchestration event deduplication (Phase 14F-4).

Covers:
- _should_send_event helper: pass-through for events without event_id
- first-seen returns True, duplicate returns False
- LRU eviction at capacity
- bus event followed by JSONL poll is sent once
- JSONL event followed by bus event is sent once
- initial catch-up events seed the seen set (not resent)
- existing _poll_new_orchestration_events regression coverage unaffected
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from app.services.session.session_stream_service import (
    _DEDUP_MAXSIZE,
    _should_send_event,
)


# ── Helper: build a minimal event ─────────────────────────────────────────────


def _event(event_id: str | None = "eid-1", event_type: str = "step_finished") -> dict:
    e: dict = {"event_type": event_type, "session_id": 1, "task_id": 1}
    if event_id is not None:
        e["event_id"] = event_id
    return e


# ── Events without event_id always pass through ───────────────────────────────


def test_no_event_id_always_passes():
    seen: OrderedDict = OrderedDict()
    e = _event(event_id=None)
    assert _should_send_event(e, seen) is True
    # A second call still passes — no state tracked for id-less events
    assert _should_send_event(e, seen) is True
    assert len(seen) == 0


def test_empty_string_event_id_treated_as_falsy_passes_through():
    seen: OrderedDict = OrderedDict()
    e = _event(event_id="")
    assert _should_send_event(e, seen) is True
    assert _should_send_event(e, seen) is True
    assert len(seen) == 0


# ── First occurrence is sent; duplicate is suppressed ─────────────────────────


def test_first_occurrence_returns_true_and_seeds_seen_set():
    seen: OrderedDict = OrderedDict()
    e = _event("uuid-1")
    result = _should_send_event(e, seen)
    assert result is True
    assert "uuid-1" in seen


def test_duplicate_event_id_returns_false():
    seen: OrderedDict = OrderedDict()
    e = _event("uuid-2")
    _should_send_event(e, seen)
    assert _should_send_event(e, seen) is False


def test_different_event_ids_both_pass():
    seen: OrderedDict = OrderedDict()
    assert _should_send_event(_event("id-a"), seen) is True
    assert _should_send_event(_event("id-b"), seen) is True
    assert len(seen) == 2


# ── LRU eviction at capacity ──────────────────────────────────────────────────


def test_lru_eviction_at_maxsize_allows_oldest_id_to_be_resent():
    seen: OrderedDict = OrderedDict()
    maxsize = 4

    for i in range(maxsize):
        _should_send_event(_event(f"id-{i}"), seen, maxsize=maxsize)

    assert len(seen) == maxsize

    # One more insertion evicts "id-0"
    _should_send_event(_event("id-new"), seen, maxsize=maxsize)
    assert len(seen) == maxsize
    assert "id-0" not in seen
    assert "id-new" in seen


def test_lru_eviction_then_old_id_returns_true():
    seen: OrderedDict = OrderedDict()
    maxsize = 2

    _should_send_event(_event("id-0"), seen, maxsize=maxsize)
    _should_send_event(_event("id-1"), seen, maxsize=maxsize)
    # Evict id-0 by adding id-2
    _should_send_event(_event("id-2"), seen, maxsize=maxsize)

    # id-0 was evicted — same event can be sent again
    assert _should_send_event(_event("id-0"), seen, maxsize=maxsize) is True


def test_default_maxsize_matches_module_constant():
    assert _DEDUP_MAXSIZE == 1024


# ── Bus then JSONL: same event_id is sent only once ───────────────────────────


def test_bus_event_then_jsonl_poll_sends_once():
    """Simulates: bus delivers event first, then JSONL poll finds the same event."""
    seen: OrderedDict = OrderedDict()
    event = _event("eid-bus-jsonl")

    # Bus path
    sent_via_bus = _should_send_event(event, seen)
    # JSONL poll path (same event_id)
    sent_via_jsonl = _should_send_event(event, seen)

    assert sent_via_bus is True
    assert sent_via_jsonl is False


# ── JSONL then bus: same event_id is sent only once ──────────────────────────


def test_jsonl_poll_then_bus_event_sends_once():
    """Simulates: JSONL poll delivers event first, then bus queue has the same event."""
    seen: OrderedDict = OrderedDict()
    event = _event("eid-jsonl-bus")

    # JSONL path first
    sent_via_jsonl = _should_send_event(event, seen)
    # Bus path arrives later
    sent_via_bus = _should_send_event(event, seen)

    assert sent_via_jsonl is True
    assert sent_via_bus is False


# ── Initial catch-up seeds the seen set ──────────────────────────────────────


def test_initial_catchup_seeds_seen_set_and_subsequent_bus_deduped():
    """Simulates: initial replay sends event, bus then delivers the same event."""
    seen: OrderedDict = OrderedDict()
    event = _event("eid-catchup")

    # Initial catch-up replay
    sent_initial = _should_send_event(event, seen)
    # Bus delivers same event in first poll cycle
    sent_bus = _should_send_event(event, seen)

    assert sent_initial is True
    assert sent_bus is False


def test_initial_catchup_with_mixed_events():
    """Catch-up events with event_id are seeded; ones without are not."""
    seen: OrderedDict = OrderedDict()
    event_with_id = _event("eid-has-id")
    event_without_id = _event(event_id=None)

    assert _should_send_event(event_with_id, seen) is True
    assert _should_send_event(event_without_id, seen) is True

    # The id'd event is now deduped; the id-less one is not
    assert _should_send_event(event_with_id, seen) is False
    assert _should_send_event(event_without_id, seen) is True


# ── Multiple distinct events with same type but different ids all pass ────────


def test_distinct_event_ids_same_type_all_pass():
    seen: OrderedDict = OrderedDict()
    events = [_event(f"unique-{i}") for i in range(10)]
    results = [_should_send_event(e, seen) for e in events]
    assert all(results)
    assert len(seen) == 10

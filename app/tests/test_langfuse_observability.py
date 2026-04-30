"""Langfuse tracing helpers should fail open and emit compact payloads."""

from __future__ import annotations

import types

from app.config import settings
from app.services.observability.langfuse import (
    build_text_trace_payload,
    flush_langfuse,
    reset_langfuse_client_for_tests,
    start_langfuse_observation,
    update_langfuse_observation,
)


def test_build_text_trace_payload_truncates_large_values():
    payload = build_text_trace_payload("x" * 700, max_preview_chars=20)

    assert payload == {
        "preview": ("x" * 20) + "...",
        "chars": 700,
        "lines": 1,
    }


def test_langfuse_helpers_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "ORCHESTRATOR_LANGFUSE_ENABLED", False)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "")
    reset_langfuse_client_for_tests()

    with start_langfuse_observation(name="disabled-span") as observation:
        assert observation is None
        update_langfuse_observation(observation, output={"status": "ok"})

    flush_langfuse()


def test_langfuse_helpers_emit_when_sdk_available(monkeypatch):
    captured = {
        "init": None,
        "start": None,
        "updates": [],
        "flushed": False,
    }

    class FakeObservation:
        def update(self, **kwargs):
            captured["updates"].append(kwargs)

    class FakeContextManager:
        def __enter__(self):
            return FakeObservation()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeLangfuse:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def start_as_current_observation(self, **kwargs):
            captured["start"] = kwargs
            return FakeContextManager()

        def flush(self):
            captured["flushed"] = True

    monkeypatch.setattr(settings, "ORCHESTRATOR_LANGFUSE_ENABLED", True)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(settings, "LANGFUSE_BASE_URL", "http://localhost:3001")
    monkeypatch.setattr(settings, "LANGFUSE_ENVIRONMENT", "test")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langfuse",
        types.SimpleNamespace(Langfuse=FakeLangfuse),
    )
    reset_langfuse_client_for_tests()

    with start_langfuse_observation(
        name="unit-test-span",
        as_type="generation",
        input={"preview": "hello"},
        metadata={"task_id": 7},
        model="gpt-test",
    ) as observation:
        update_langfuse_observation(
            observation,
            output={"status": "completed"},
            usage_details={"input": 10, "output": 20},
        )

    flush_langfuse()

    assert captured["init"]["public_key"] == "pk-test"
    assert captured["init"]["secret_key"] == "sk-test"
    assert captured["init"]["base_url"] == "http://localhost:3001"
    assert captured["start"]["name"] == "unit-test-span"
    assert captured["start"]["as_type"] == "generation"
    assert captured["start"]["model"] == "gpt-test"
    assert captured["updates"] == [
        {
            "output": {"status": "completed"},
            "usage_details": {"input": 10, "output": 20},
        }
    ]
    assert captured["flushed"] is True

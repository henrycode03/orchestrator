"""Focused provider-evidence contract tests for PHASE34-POB1."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from app.services.observability import planning_provider_evidence as evidence
from app.services.orchestration.planning.planner import PlannerService


def _begin(
    tmp_path: Path, monkeypatch, events: list[dict]
) -> evidence.PlanningProviderEvidence:
    def append_event(**kwargs):
        request_path = (
            Path(kwargs["details"]["evidence_artifact"]).parent / "planning-request.txt"
        )
        assert request_path.exists(), "started event must follow request retention"
        events.append(kwargs)
        return {"event_id": f"event-{len(events)}", **kwargs}

    monkeypatch.setattr(evidence, "append_orchestration_event", append_event)
    return evidence.begin_planning_provider_evidence(
        control_state_location=tmp_path / "control-root",
        project_id=114,
        task_id=235,
        session_id=183,
        task_execution_id=999,
        attempt=1,
        model="qwen-local",
        provider_endpoint_class="https://gateway.example/v1/chat/completions?secret=no",
        effective_timeout_seconds=300,
        transport_timeout_seconds=330,
        prompt="Return one bounded planning result.",
        invocation_kind="direct_chat_completions",
        provider_api_streaming=False,
    )


def test_started_event_precedes_provider_invocation_and_prompt_is_exact(
    tmp_path, monkeypatch
):
    events: list[dict] = []
    recorder = _begin(tmp_path, monkeypatch, events)

    request_path = recorder.artifact_directory / "planning-request.txt"
    assert (
        request_path.read_text(encoding="utf-8")
        == "Return one bounded planning result."
    )
    metadata = json.loads(
        (recorder.artifact_directory / "evidence.json").read_text(encoding="utf-8")
    )
    assert events[0]["event_type"] == "planning_provider_started"
    assert (
        metadata["prompt_sha256"]
        == hashlib.sha256(request_path.read_bytes()).hexdigest()
    )
    assert metadata["prompt_chars"] == len(request_path.read_text(encoding="utf-8"))
    assert metadata["prompt_token_estimate"] == 9
    assert metadata["effective_timeout_seconds"] == 300
    assert metadata["transport_timeout_seconds"] == 330
    assert (
        metadata["provider_endpoint_class"]
        == "https://gateway.example/v1/chat/completions"
    )
    assert "Authorization" not in json.dumps(metadata)
    assert "secret=no" not in json.dumps(metadata)


def test_completed_response_retains_bounded_visible_reasoning_usage_and_finish_reason(
    tmp_path, monkeypatch
):
    events: list[dict] = []
    recorder = _begin(tmp_path, monkeypatch, events)

    recorder.complete(
        visible_content='[{"step_number": 1}]',
        reasoning_content="bounded provider reasoning",
        usage={"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        finish_reason="stop",
        provider_request_correlation_id="req-114-235",
    )

    assert events[-1]["event_type"] == "planning_provider_completed"
    response = json.loads(
        (recorder.artifact_directory / "planning-response.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (recorder.artifact_directory / "evidence.json").read_text(encoding="utf-8")
    )
    assert response["visible_content"] == '[{"step_number": 1}]'
    assert response["reasoning_content"] == "bounded provider reasoning"
    assert response["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 4,
        "total_tokens": 13,
    }
    assert response["finish_reason"] == "stop"
    assert metadata["response_received"] is True
    assert metadata["visible_chars"] == len(response["visible_content"])
    assert metadata["reasoning_chars"] == len(response["reasoning_content"])
    assert metadata["provider_request_correlation_id"] == "req-114-235"
    assert metadata["elapsed_seconds"] >= 0


def test_failed_timeout_retains_exception_and_survives_exception_handling(
    tmp_path, monkeypatch
):
    events: list[dict] = []
    recorder = _begin(tmp_path, monkeypatch, events)

    recorder.fail(
        TimeoutError("provider deadline expired"),
        response_received=False,
        partial_content_snapshot="provider was still active",
    )

    assert events[-1]["event_type"] == "planning_provider_failed"
    metadata = json.loads(
        (recorder.artifact_directory / "evidence.json").read_text(encoding="utf-8")
    )
    response = json.loads(
        (recorder.artifact_directory / "planning-response.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["response_received"] is False
    assert metadata["exception_type"] == "TimeoutError"
    assert metadata["elapsed_seconds"] >= 0
    assert response["partial_content_snapshot"] == "provider was still active"


def test_each_provider_invocation_gets_a_distinct_attempt_id(tmp_path, monkeypatch):
    events: list[dict] = []
    first = _begin(tmp_path, monkeypatch, events)
    second = evidence.begin_planning_provider_evidence(
        control_state_location=tmp_path / "control-root",
        project_id=114,
        task_id=235,
        session_id=183,
        task_execution_id=1000,
        attempt=2,
        model="qwen-local",
        provider_endpoint_class="openclaw_cli",
        effective_timeout_seconds=300,
        transport_timeout_seconds=330,
        prompt="Return one bounded planning result.",
        invocation_kind="planning",
        provider_api_streaming=True,
    )
    assert first.attempt_id != second.attempt_id
    assert first.metadata["task_execution_id"] == 999
    assert second.metadata["task_execution_id"] == 1000
    assert first.metadata["attempt"] == 1
    assert second.metadata["attempt"] == 2


def test_chat_completion_response_shape_extracts_visible_reasoning_usage_and_finish():
    observed = evidence.inspect_chat_completion_response(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "plan",
                        "reasoning_content": "reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        }
    )
    assert observed == {
        "response_received": True,
        "visible_content": "plan",
        "reasoning_content": "reasoning",
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        "finish_reason": "stop",
        "provider_request_correlation_id": "chatcmpl-1",
    }


def test_runtime_evidence_stays_under_control_root_and_not_product_root(
    tmp_path, monkeypatch
):
    events: list[dict] = []
    recorder = _begin(tmp_path, monkeypatch, events)
    product_root = tmp_path / "product-root"
    product_root.mkdir()
    before = tuple(product_root.iterdir())

    recorder.complete(visible_content="[]", usage=None, finish_reason=None)

    assert recorder.artifact_directory.is_relative_to(tmp_path / "control-root")
    assert tuple(product_root.iterdir()) == before


def test_direct_planning_persists_started_event_before_nonstreaming_http(
    tmp_path, monkeypatch
):
    events: list[dict] = []
    recorder = _begin(tmp_path, monkeypatch, events)

    class Runtime:
        db = None
        project_id = 114
        task_id = 235
        session_id = 183
        task_execution_id = 999

        @staticmethod
        def get_backend_metadata():
            return {"backend": "openai_chat_completions", "model_family": "qwen-local"}

    class Response:
        headers = {"x-request-id": "chatcmpl-test"}

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "id": "chatcmpl-body-id",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "[]",
                            "reasoning": "short reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

    class AsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            assert events and events[0]["event_type"] == "planning_provider_started"
            assert kwargs["json"]["stream"] is False
            assert "Authorization" not in kwargs["headers"]
            return Response()

    monkeypatch.setattr(
        "app.services.orchestration.planning.planner.begin_planning_provider_evidence_from_runtime",
        lambda *args, **kwargs: recorder,
    )
    monkeypatch.setattr(
        "app.services.orchestration.planning.planner.httpx.AsyncClient", AsyncClient
    )
    monkeypatch.setattr(
        "app.services.orchestration.planning.planner.settings.PLANNING_REPAIR_BASE_URL",
        "https://gateway.example/v1",
    )
    monkeypatch.setattr(
        "app.services.orchestration.planning.planner.settings.PLANNING_REPAIR_MODEL",
        "qwen-local",
    )
    monkeypatch.setattr(
        "app.services.orchestration.planning.planner.settings.PLANNING_REPAIR_API_KEY",
        "",
    )
    monkeypatch.setattr(
        "app.services.orchestration.planning.planner.settings.PLANNING_REPAIR_TIMEOUT_SECONDS",
        300,
    )

    result = asyncio.run(
        PlannerService._invoke_direct_no_thinking_planning(
            Runtime(), "Return a small planning result."
        )
    )

    assert result["planning_direct"] is True
    assert events[-1]["event_type"] == "planning_provider_completed"
    metadata = json.loads(
        (recorder.artifact_directory / "evidence.json").read_text(encoding="utf-8")
    )
    assert metadata["provider_request_correlation_id"] == "chatcmpl-test"
    assert metadata["response_received"] is True
    assert metadata["visible_chars"] == 2
    assert metadata["reasoning_chars"] == len("short reasoning")
    assert metadata["usage"]["completion_tokens"] == 2

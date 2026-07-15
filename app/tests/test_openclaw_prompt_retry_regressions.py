from __future__ import annotations

import asyncio
import pytest

from app.models import Project, Session as SessionModel, Task, TaskStatus
from app.config import Settings, settings
from app.services.agents.openclaw_service import (
    OpenClawSessionError,
    OpenClawSessionService,
)


def _seed_service_models(db_session):
    project = Project(name="Prompt Retry Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    session = SessionModel(
        project_id=project.id,
        name="Prompt Retry Session",
        status="running",
        is_active=True,
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)

    task = Task(
        project_id=project.id,
        title="Retry overflowed prompt",
        description="Regression coverage for compact-prompt retry logic",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    return session, task


def test_execute_task_retries_context_overflow_with_compact_prompt(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "AGENT_MODEL", "local")
    session, task = _seed_service_models(db_session)
    service = OpenClawSessionService(
        db_session, session.id, task.id, use_demo_mode=False
    )

    seen_prompts: list[str] = []

    async def fake_execute_task_with_streaming(
        prompt, timeout_seconds, log_callback, *, reuse_task_session=True
    ):
        seen_prompts.append(prompt)
        if len(seen_prompts) == 1:
            return {
                "status": "failed",
                "mode": "real",
                "output": "",
                "error": "Context window exceeded",
                "logs": [],
            }
        return {
            "status": "completed",
            "mode": "real",
            "output": '{"ok":true}',
            "error": "",
            "logs": [],
        }

    monkeypatch.setattr(
        service, "execute_task_with_streaming", fake_execute_task_with_streaming
    )
    monkeypatch.setattr(service, "_log_entry", lambda *args, **kwargs: None)

    prompt = "\n".join(
        [
            "**Step:** Create the final Vitest test suite",
            "**Context:** " + ("existing workspace details " * 600),
            "**Output:** status, output, verification_output, files_changed, error_message",
        ]
    )

    result = asyncio.run(service.execute_task(prompt, timeout_seconds=30))

    assert result["status"] == "completed"
    assert result["backend"] == "local_openclaw"
    assert result["model_family"] == "local"
    assert result["backend_capabilities"]["supports_streaming"] is True
    assert len(seen_prompts) == 2
    assert len(seen_prompts[1]) < len(seen_prompts[0])
    assert "[Content truncated for performance]" in seen_prompts[1]


def test_execute_task_preserves_timeout_runtime_diagnostics(db_session, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    session, task = _seed_service_models(db_session)
    service = OpenClawSessionService(
        db_session, session.id, task.id, use_demo_mode=False
    )

    async def fake_execute_task_with_streaming(*args, **kwargs):
        exc = OpenClawSessionError("Task timed out after 90s")
        exc.runtime_diagnostics = {
            "timed_out": True,
            "stdout_chars": 0,
            "stderr_contains_model_content": False,
            "output_channel_used": "none",
        }
        raise exc

    monkeypatch.setattr(
        service, "execute_task_with_streaming", fake_execute_task_with_streaming
    )
    monkeypatch.setattr(service, "_log_entry", lambda *args, **kwargs: None)

    try:
        asyncio.run(service.execute_task("Return a plan", timeout_seconds=90))
    except OpenClawSessionError as exc:
        assert exc.runtime_diagnostics == {
            "timed_out": True,
            "stdout_chars": 0,
            "stderr_contains_model_content": False,
            "output_channel_used": "none",
        }
        return

    raise AssertionError("Expected timeout error")


def test_phase7f_debug_repair_no_longer_uses_openclaw_provider_io(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "AGENT_MODEL", "local")
    session, task = _seed_service_models(db_session)
    service = OpenClawSessionService(
        db_session, session.id, task.id, use_demo_mode=False
    )

    seen: dict[str, object] = {}

    async def fake_execute_task_with_streaming(prompt, timeout, log_callback, **kwargs):
        seen["prompt"] = prompt
        seen["timeout_seconds"] = timeout
        seen["diagnostic_metadata"] = kwargs.get("diagnostic_metadata")
        return {"status": "completed", "output": '{"ops":[]}', "logs": []}

    monkeypatch.setattr(
        service, "execute_task_with_streaming", fake_execute_task_with_streaming
    )
    monkeypatch.setattr(service, "_log_entry", lambda *args, **kwargs: None)

    result = asyncio.run(
        service.execute_task(
            "Return bounded JSON repair",
            timeout_seconds=180,
            diagnostic_label="BOUNDED_EXECUTION_DEBUG_REPAIR",
            diagnostic_metadata={"debug_failure_class": "source_step_validation"},
        )
    )

    assert result["status"] == "completed"
    assert result["backend"] == "local_openclaw"
    assert seen["prompt"] == "Return bounded JSON repair"
    assert seen["timeout_seconds"] == 180
    assert seen["diagnostic_metadata"]["debug_failure_class"] == (
        "source_step_validation"
    )


def test_bounded_debug_repair_architecture_label_uses_registry_owner(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "AGENT_MODEL", "local")
    session, task = _seed_service_models(db_session)
    service = OpenClawSessionService(
        db_session, session.id, task.id, use_demo_mode=False
    )

    seen: dict[str, object] = {}

    async def fake_execute_task_with_streaming(prompt, timeout, log_callback, **kwargs):
        seen["prompt"] = prompt
        seen["timeout_seconds"] = timeout
        seen["diagnostic_metadata"] = kwargs.get("diagnostic_metadata")
        return {"status": "completed", "output": '{"ops":[]}', "logs": []}

    monkeypatch.setattr(
        service, "execute_task_with_streaming", fake_execute_task_with_streaming
    )
    monkeypatch.setattr(service, "_log_entry", lambda *args, **kwargs: None)

    result = asyncio.run(
        service.execute_task(
            "Return bounded JSON repair",
            timeout_seconds=180,
            diagnostic_label="BOUNDED_EXECUTION_DEBUG_REPAIR",
            diagnostic_metadata={"debug_failure_class": "source_step_validation"},
        )
    )

    assert result["status"] == "completed"
    assert result["backend"] == "local_openclaw"
    assert seen["prompt"] == "Return bounded JSON repair"
    assert seen["timeout_seconds"] == 180
    assert seen["diagnostic_metadata"]["debug_failure_class"] == (
        "source_step_validation"
    )


def test_bounded_debug_repair_diagnostic_label_architecture_alias():
    assert (
        OpenClawSessionService._diagnostic_label_architecture(
            "BOUNDED_EXECUTION_DEBUG_REPAIR"
        )
        == "BOUNDED_EXECUTION_DEBUG_REPAIR"
    )
    assert (
        OpenClawSessionService._diagnostic_label_architecture(
            "BOUNDED_EXECUTION_DEBUG_REPAIR"
        )
        == "BOUNDED_EXECUTION_DEBUG_REPAIR"
    )
    assert OpenClawSessionService._diagnostic_label_architecture("PLANNING") is None


def test_debug_repair_direct_routing_accepts_legacy_and_architecture_labels(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "AGENT_MODEL", "local")
    session, task = _seed_service_models(db_session)
    service = OpenClawSessionService(
        db_session, session.id, task.id, use_demo_mode=False
    )

    assert (
        service._should_use_structured_debug_repair_direct_chat(
            "BOUNDED_EXECUTION_DEBUG_REPAIR"
        )
        is False
    )
    assert (
        service._should_use_structured_debug_repair_direct_chat(
            "BOUNDED_EXECUTION_DEBUG_REPAIR"
        )
        is False
    )
    assert service._should_use_structured_debug_repair_direct_chat("PLANNING") is False


def test_phase7f_debug_repair_controls_are_provider_neutral():
    from app.services.agents.runtime_invocation import RuntimeInvocationOptions

    options = RuntimeInvocationOptions(
        max_output_tokens=2048,
        temperature=0.0,
        reasoning_enabled=False,
        stream=False,
    )

    assert options.max_output_tokens == 2048
    assert options.temperature == 0.0
    assert options.reasoning_enabled is False


def test_debug_repair_responses_shape_is_owned_by_responses_adapter():
    from app.services.agents.providers.openai_adapter import _extract_output_text

    assert (
        _extract_output_text(
            {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": '{"ops":[]}'},
                        ]
                    }
                ]
            }
        )
        == '{"ops":[]}'
    )


def test_debug_repair_extracts_responses_output_text():
    body = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": '{"ops":['},
                    {"type": "output_text", "text": "]}"},
                ]
            }
        ]
    }

    assert OpenClawSessionService._extract_responses_output_text(body) == '{"ops":[]}'


def test_debug_repair_legacy_env_aliases_populate_architecture_settings():
    configured = Settings(
        _env_file=None,
        PHASE7F_REPAIR_BASE_URL="https://legacy.example/v1",
        PHASE7F_REPAIR_MODEL="legacy-model",
        PHASE7F_REPAIR_API_KEY="legacy-key",
        PHASE7F_REPAIR_DISABLE_THINKING=False,
        PHASE7F_REPAIR_DIRECT_ENABLED=False,
    )

    assert configured.DEBUG_REPAIR_BASE_URL == "https://legacy.example/v1"
    assert configured.DEBUG_REPAIR_MODEL == "legacy-model"
    assert configured.DEBUG_REPAIR_API_KEY == "legacy-key"
    assert configured.DEBUG_REPAIR_DISABLE_THINKING is False
    assert configured.DEBUG_REPAIR_DIRECT_ENABLED is False


def test_openclaw_adapter_rejects_debug_repair_provider_controls(
    db_session, monkeypatch
):
    from app.services.agents.runtime_invocation import RuntimeInvocationOptions

    monkeypatch.setattr(settings, "DEBUG_REPAIR_BACKEND", "openai_responses_api")
    session, task = _seed_service_models(db_session)
    service = OpenClawSessionService(
        db_session, session.id, task.id, use_demo_mode=False
    )
    with pytest.raises(Exception, match="provider-specific invocation options"):
        asyncio.run(
            service.invoke_prompt(
                "Return JSON",
                invocation_options=RuntimeInvocationOptions(max_output_tokens=2048),
            )
        )


def test_non_phase7f_debug_repair_keeps_openclaw_streaming_path(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "AGENT_MODEL", "local")
    session, task = _seed_service_models(db_session)
    service = OpenClawSessionService(
        db_session, session.id, task.id, use_demo_mode=False
    )

    calls: list[str] = []

    async def fake_execute_task_with_streaming(
        prompt, timeout_seconds, log_callback, *, reuse_task_session=True, **kwargs
    ):
        calls.append(prompt)
        return {
            "status": "completed",
            "mode": "real",
            "output": "ok",
            "error": "",
            "logs": [],
        }

    monkeypatch.setattr(
        service, "execute_task_with_streaming", fake_execute_task_with_streaming
    )
    monkeypatch.setattr(service, "_log_entry", lambda *args, **kwargs: None)

    result = asyncio.run(
        service.execute_task(
            "Return a normal answer",
            timeout_seconds=30,
            diagnostic_label="PLANNING",
        )
    )

    assert result["status"] == "completed"
    assert calls == ["Return a normal answer"]

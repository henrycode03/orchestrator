"""Provider-free Phase 34-C2 single-runtime/single-model contract tests."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import settings
from app.services.agents.agent_runtime import (
    BackendRole,
    RuntimeCapabilityError,
    low_resource_single_model_runtime_matrix,
    resolve_runtime_configuration,
    validate_runtime_provider_contract,
)
from app.services.agents.agent_backends import ExecutionTopology
from app.services.agents.providers.openai_chat_adapter import (
    OpenAIChatCompletionsRuntime,
)
from app.services.agents.runtime_invocation import RuntimeInvocationOptions
from app.services.orchestration.planning.planner import PlannerService
from app.services.planning.providers.direct_openai_compatible import (
    _load_configuration,
)


GENERATION_ROLES = (
    BackendRole.PLANNING,
    BackendRole.REPAIR,
    BackendRole.DEBUG_REPAIR,
    BackendRole.COMPLETION_REPAIR,
    BackendRole.EXECUTION,
)


def _set_low_resource_settings(monkeypatch, *, backend: str) -> None:
    monkeypatch.setattr(settings, "LOW_RESOURCE_SINGLE_MODEL", True, raising=False)
    monkeypatch.setattr(settings, "AGENT_BACKEND", backend)
    monkeypatch.setattr(settings, "AGENT_MODEL", "model-x")
    monkeypatch.setattr(settings, "PLANNING_BACKEND", backend)
    monkeypatch.setattr(settings, "EXECUTION_BACKEND", backend)
    monkeypatch.setattr(settings, "PLANNER_MODEL", "model-x")
    monkeypatch.setattr(settings, "EXECUTION_MODEL", "model-x")
    monkeypatch.setattr(settings, "OLLAMA_AGENT_MODEL", "model-x")
    monkeypatch.setattr(settings, "PLANNING_ADAPTATION_PROFILE", "ollama_default")
    monkeypatch.setattr(settings, "EXECUTION_ADAPTATION_PROFILE", "ollama_default")
    monkeypatch.setattr(settings, "REPAIR_ADAPTATION_PROFILE", "openai_default")
    monkeypatch.setattr(settings, "DEBUG_REPAIR_ADAPTATION_PROFILE", "openai_default")
    monkeypatch.setattr(
        settings, "COMPLETION_REPAIR_ADAPTATION_PROFILE", "openai_default"
    )
    monkeypatch.setattr(settings, "REPAIR_BACKEND", "openai_chat_completions")
    monkeypatch.setattr(settings, "DEBUG_REPAIR_BACKEND", "openai_chat_completions")
    monkeypatch.setattr(
        settings, "COMPLETION_REPAIR_BACKEND", "openai_chat_completions"
    )
    monkeypatch.setattr(settings, "PLANNING_REPAIR_MODEL", "legacy-repair-model")
    monkeypatch.setattr(settings, "DEBUG_REPAIR_MODEL", "legacy-debug-model")
    monkeypatch.setattr(settings, "COMPLETION_REPAIR_MODEL", "legacy-completion-model")
    monkeypatch.setattr(
        settings, "PLANNING_DIRECT_BASE_URL", "http://single-runtime:8001/v1"
    )
    monkeypatch.setattr(settings, "PLANNING_DIRECT_API_KEY", "canonical-key")
    monkeypatch.setattr(
        settings, "PLANNING_REPAIR_BASE_URL", "http://legacy-repair:8001/v1"
    )
    monkeypatch.setattr(settings, "PLANNING_REPAIR_API_KEY", "legacy-key")
    monkeypatch.setattr(
        settings, "DEBUG_REPAIR_BASE_URL", "http://legacy-debug:8001/v1"
    )
    monkeypatch.setattr(settings, "DEBUG_REPAIR_API_KEY", "legacy-debug-key")
    monkeypatch.setattr(settings, "EXECUTION_CONTEXT_TOKENS", 16000)
    monkeypatch.setattr(settings, "PLANNING_REPAIR_CONTEXT_TOKENS", 16000)
    monkeypatch.setattr(settings, "DEBUG_REPAIR_CONTEXT_TOKENS", 16000)
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://single-runtime:11434")
    monkeypatch.setattr(settings, "PLANNING_DIRECT_NO_THINKING_FOR_DIRECT_OLLAMA", True)


def test_low_resource_ollama_roles_inherit_one_canonical_identity(
    db_session, monkeypatch
):
    _set_low_resource_settings(monkeypatch, backend="direct_ollama")

    configurations = {
        role: resolve_runtime_configuration(db_session, role)
        for role in GENERATION_ROLES
    }

    assert {item.backend_name for item in configurations.values()} == {"direct_ollama"}
    assert {item.model_family for item in configurations.values()} == {"model-x"}
    assert {item.adaptation_profile for item in configurations.values()} == {
        "ollama_default"
    }
    matrix = low_resource_single_model_runtime_matrix(db_session)
    assert matrix["errors"] == []
    assert matrix["one_runtime"] is True
    assert matrix["one_generation_model"] is True
    assert matrix["one_profile"] is True
    assert matrix["one_endpoint"] is True
    assert (
        matrix["execution_topology"] == ExecutionTopology.STRUCTURED_ORCHESTRATOR.value
    )
    assert matrix["openclaw_required"] is False
    assert matrix["second_provider_required"] is False


def test_low_resource_openai_compatible_roles_inherit_one_canonical_identity(
    db_session, monkeypatch
):
    _set_low_resource_settings(monkeypatch, backend="openai_chat_completions")

    configurations = {
        role: resolve_runtime_configuration(db_session, role)
        for role in GENERATION_ROLES
    }

    assert {item.backend_name for item in configurations.values()} == {
        "openai_chat_completions"
    }
    assert {item.model_family for item in configurations.values()} == {"model-x"}
    assert {item.adaptation_profile for item in configurations.values()} == {
        "ollama_default"
    }
    matrix = low_resource_single_model_runtime_matrix(db_session)
    assert matrix["errors"] == []
    assert matrix["one_runtime"] is True
    assert matrix["one_generation_model"] is True
    assert matrix["one_profile"] is True
    assert matrix["one_endpoint"] is True
    assert (
        matrix["execution_topology"] == ExecutionTopology.STRUCTURED_ORCHESTRATOR.value
    )


def test_low_resource_configuration_fails_closed_on_core_model_mismatch(
    db_session, monkeypatch
):
    _set_low_resource_settings(monkeypatch, backend="direct_ollama")
    monkeypatch.setattr(settings, "EXECUTION_MODEL", "different-model")

    try:
        validate_runtime_provider_contract(
            db_session, BackendRole.PLANNING, dispatch=False
        )
    except RuntimeCapabilityError as exc:
        assert "same model" in str(exc)
    else:
        raise AssertionError("mismatched low-resource core models were accepted")


def test_low_resource_openai_repair_adapter_uses_canonical_endpoint(
    db_session, monkeypatch
):
    _set_low_resource_settings(monkeypatch, backend="openai_chat_completions")
    configuration = resolve_runtime_configuration(db_session, BackendRole.REPAIR)
    runtime = OpenAIChatCompletionsRuntime(
        db_session, session_id=None, runtime_configuration=configuration
    )

    assert runtime._base_url == "http://single-runtime:8001/v1"
    assert (
        runtime._invocation_base_url(RuntimeInvocationOptions())
        == "http://single-runtime:8001/v1"
    )
    assert runtime._invocation_api_key(RuntimeInvocationOptions()) == "canonical-key"


def test_low_resource_direct_planning_uses_canonical_model(monkeypatch):
    _set_low_resource_settings(monkeypatch, backend="direct_ollama")
    runtime_service = SimpleNamespace(
        get_backend_metadata=lambda: {
            "backend": "direct_ollama",
            "model_family": "model-x",
        }
    )

    assert (
        PlannerService._should_try_direct_no_thinking_planning(
            runtime_service, prompt_chars=100
        )
        is True
    )
    assert PlannerService._direct_no_thinking_model(runtime_service) == "model-x"


def test_low_resource_direct_planning_provider_uses_canonical_model_and_endpoint(
    monkeypatch,
):
    _set_low_resource_settings(monkeypatch, backend="direct_ollama")
    configuration = _load_configuration()

    assert configuration.model == "model-x"
    assert configuration.endpoint == ("http://single-runtime:11434/v1/chat/completions")


def test_gx10_hybrid_role_routing_remains_unchanged(db_session, monkeypatch):
    monkeypatch.setattr(settings, "LOW_RESOURCE_SINGLE_MODEL", False, raising=False)
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "PLANNING_BACKEND", "openai_chat_completions")
    monkeypatch.setattr(settings, "EXECUTION_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "REPAIR_BACKEND", "openai_chat_completions")
    monkeypatch.setattr(settings, "DEBUG_REPAIR_BACKEND", "openai_chat_completions")
    monkeypatch.setattr(settings, "COMPLETION_REPAIR_BACKEND", "direct_ollama")
    monkeypatch.setattr(settings, "PLANNER_MODEL", "planning-model")
    monkeypatch.setattr(settings, "EXECUTION_MODEL", "execution-model")
    monkeypatch.setattr(settings, "PLANNING_REPAIR_MODEL", "repair-model")
    monkeypatch.setattr(settings, "DEBUG_REPAIR_MODEL", "debug-model")
    monkeypatch.setattr(settings, "COMPLETION_REPAIR_MODEL", "completion-model")

    assert (
        resolve_runtime_configuration(db_session, BackendRole.PLANNING).backend_name
        == "openai_chat_completions"
    )
    assert (
        resolve_runtime_configuration(db_session, BackendRole.EXECUTION).backend_name
        == "local_openclaw"
    )
    assert (
        resolve_runtime_configuration(
            db_session, BackendRole.COMPLETION_REPAIR
        ).backend_name
        == "direct_ollama"
    )

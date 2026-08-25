"""POST33-EXEC1: execution capability decomposition and single-model lifecycle.

``supports_step_execution`` was a single boolean that the registry assigned as
if it meant "is a full agent runtime" (OpenClaw declared it alongside native
tools, streaming and checkpoint/resume) while ``validate_runtime_capabilities``
consumed it as "may own the EXECUTION role".  The production execution loop
does not need the former: it applies the accepted step's structured file
operations, portable commands and verification itself and calls the runtime
only for the residual reasoning turn.

These tests pin the decomposition: EXECUTION eligibility is derived from an
explicit capability set per execution topology, OpenClaw's agent topology is
unchanged, and no backend name is special-cased.

Every test here is provider-free: no runtime is instantiated and no HTTP call
is made.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import settings
from app.services.agents.agent_backends import (
    BackendHealth,
    EXECUTION_TOPOLOGY_REQUIRED_CAPABILITIES,
    ExecutionTopology,
    get_backend_descriptor,
    list_supported_backends,
)
from app.services.agents.agent_runtime import (
    RuntimeCapabilityError,
    resolve_runtime_configuration,
    validate_runtime_capabilities,
)
from app.services.agents.runtime_configuration import BackendRole

DIRECT_BACKENDS = ("direct_ollama", "openai_chat_completions")
AMPLE_CONTEXT = 200_000


def _validate(backend: str, *, dispatch: bool = False, **kwargs):
    return validate_runtime_capabilities(
        get_backend_descriptor(backend),
        BackendRole.EXECUTION,
        effective_context_tokens=AMPLE_CONTEXT,
        dispatch=dispatch,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _execution_model_configured(monkeypatch):
    # EXECUTION provider readiness is a separate, still fail-closed check; keep
    # these capability assertions independent of the CI execution-model value.
    monkeypatch.setattr(settings, "EXECUTION_MODEL", "capability-test-model")


# ---------------------------------------------------------------------------
# A. OpenClaw agent execution non-regression
# ---------------------------------------------------------------------------


def test_openclaw_still_qualifies_for_the_full_agent_execution_topology():
    capabilities = get_backend_descriptor("local_openclaw").capabilities
    assert capabilities.supports_step_reasoning is True
    assert capabilities.supports_step_execution is True
    assert capabilities.supports_tool_execution is True
    assert capabilities.supports_agent_workspace_binding is True
    assert capabilities.supports_streaming is True
    assert capabilities.supports_checkpoint_resume is True
    assert (
        capabilities.missing_execution_capabilities(ExecutionTopology.AGENT_RUNTIME)
        == []
    )

    for topology in ExecutionTopology:
        readiness = _validate("local_openclaw", execution_topology=topology)
        assert readiness["execution_topology"] == topology.value


# ---------------------------------------------------------------------------
# B/C. Direct backends qualify for the structured-orchestrator topology only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", DIRECT_BACKENDS)
def test_direct_backend_qualifies_for_structured_orchestrator_execution(backend):
    capabilities = get_backend_descriptor(backend).capabilities
    assert capabilities.supports_step_reasoning is True

    readiness = _validate(backend)
    assert readiness["execution_topology"] == (
        ExecutionTopology.STRUCTURED_ORCHESTRATOR.value
    )
    assert readiness["required_execution_capabilities"] == ["supports_step_reasoning"]


def test_structured_orchestrator_is_the_default_execution_topology():
    # The production execution loop never asks for the agent topology, so the
    # default must be the topology it actually runs.
    default = _validate("direct_ollama")
    explicit = _validate(
        "direct_ollama",
        execution_topology=ExecutionTopology.STRUCTURED_ORCHESTRATOR,
    )
    assert default["execution_topology"] == explicit["execution_topology"]


# ---------------------------------------------------------------------------
# D. Direct backends still fail closed on enhanced-agent requirements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", DIRECT_BACKENDS)
def test_direct_backend_rejected_for_agent_runtime_execution(backend):
    with pytest.raises(RuntimeCapabilityError) as excinfo:
        _validate(
            backend,
            dispatch=True,
            execution_topology=ExecutionTopology.AGENT_RUNTIME,
        )
    assert excinfo.value.code == "provider_endpoint_incompatible"
    message = str(excinfo.value)
    assert ExecutionTopology.AGENT_RUNTIME.value in message
    assert "supports_tool_execution" in message
    assert "supports_agent_workspace_binding" in message


@pytest.mark.parametrize("backend", DIRECT_BACKENDS)
def test_direct_backends_never_claim_unimplemented_agent_capabilities(backend):
    capabilities = get_backend_descriptor(backend).capabilities
    assert capabilities.supports_tool_execution is False
    assert capabilities.supports_agent_workspace_binding is False
    assert capabilities.supports_checkpoint_resume is False
    assert capabilities.supports_streaming is False
    assert capabilities.supports_step_execution is False


def test_openai_responses_api_has_no_execution_step_contract():
    # execute_task() delegates to invoke_prompt() with the generic contract.
    capabilities = get_backend_descriptor("openai_responses_api").capabilities
    assert capabilities.supports_step_reasoning is False
    with pytest.raises(RuntimeCapabilityError):
        _validate("openai_responses_api", dispatch=True)


# ---------------------------------------------------------------------------
# E/F. Planning and repair routing are untouched by the decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role,capability",
    [
        (BackendRole.PLANNING, "supports_planning"),
        (BackendRole.REPAIR, "supports_planning"),
        (BackendRole.COMPLETION_REPAIR, "supports_planning"),
        (BackendRole.DEBUG_REPAIR, "supports_debug_repair"),
    ],
)
def test_non_execution_roles_still_gate_on_their_original_capability(role, capability):
    for descriptor in list_supported_backends():
        if not descriptor.implemented:
            continue
        supported = getattr(descriptor.capabilities, capability)
        if supported:
            continue
        with pytest.raises(RuntimeCapabilityError) as excinfo:
            validate_runtime_capabilities(
                descriptor,
                role,
                effective_context_tokens=AMPLE_CONTEXT,
                dispatch=True,
            )
        assert excinfo.value.code in {
            "provider_endpoint_incompatible",
            "provider_model_unavailable",
        }


def test_non_execution_roles_report_no_execution_topology():
    readiness = validate_runtime_capabilities(
        get_backend_descriptor("openai_chat_completions"),
        BackendRole.PLANNING,
        dispatch=False,
    )
    assert readiness["execution_topology"] is None
    assert readiness["required_execution_capabilities"] is None


# ---------------------------------------------------------------------------
# G/H. One backend and one model may own several roles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend,model",
    [
        ("direct_ollama", "qwen2.5-coder:7b"),
        ("openai_chat_completions", "qwen-local"),
    ],
)
def test_single_model_deployment_resolves_planning_and_execution(
    db_session, monkeypatch, backend, model
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", backend)
    monkeypatch.setattr(settings, "PLANNING_BACKEND", backend)
    monkeypatch.setattr(settings, "EXECUTION_BACKEND", backend)
    monkeypatch.setattr(settings, "PLANNER_MODEL", model)
    monkeypatch.setattr(settings, "EXECUTION_MODEL", model)
    monkeypatch.setattr(settings, "AGENT_MODEL", model)
    monkeypatch.setattr(settings, "PLANNING_ADAPTATION_PROFILE", "ollama_default")
    monkeypatch.setattr(settings, "EXECUTION_ADAPTATION_PROFILE", "ollama_default")

    planning = resolve_runtime_configuration(db_session, BackendRole.PLANNING)
    execution = resolve_runtime_configuration(db_session, BackendRole.EXECUTION)

    assert planning.backend_name == execution.backend_name == backend
    # One physical model serves both roles: distinct roles never imply a
    # second model load.
    assert planning.model_family == execution.model_family == model
    assert planning.role is not execution.role
    assert planning.is_behaviorally_equivalent(execution)

    _validate(backend)


# ---------------------------------------------------------------------------
# I. No backend-name special case
# ---------------------------------------------------------------------------


def test_execution_eligibility_is_capability_derived_not_name_derived():
    for topology, required in EXECUTION_TOPOLOGY_REQUIRED_CAPABILITIES.items():
        assert required, f"{topology} must require at least one capability"
        for descriptor in list_supported_backends():
            expected = [
                name for name in required if not getattr(descriptor.capabilities, name)
            ]
            assert (
                descriptor.capabilities.missing_execution_capabilities(topology)
                == expected
            )


def test_capability_dict_shape_is_additive():
    payload = get_backend_descriptor("local_openclaw").capabilities.to_dict()
    # Existing operator-surface keys are retained verbatim.
    for key in (
        "supports_planning",
        "supports_step_execution",
        "supports_debug_repair",
        "supports_streaming",
        "supports_checkpoint_resume",
        "supports_tool_execution",
        "supports_json_mode",
    ):
        assert key in payload
    assert payload["supports_step_reasoning"] is True
    assert payload["supports_agent_workspace_binding"] is True


# ---------------------------------------------------------------------------
# Fail-closed defaults
# ---------------------------------------------------------------------------


def test_decomposed_capabilities_default_to_false():
    from app.services.agents.agent_backends import BackendCapabilities

    capabilities = BackendCapabilities(
        supports_planning=True,
        supports_step_execution=True,
        supports_debug_repair=True,
        supports_streaming=True,
        supports_checkpoint_resume=True,
        supports_tool_execution=True,
        supports_json_mode=True,
    )
    assert capabilities.supports_step_reasoning is False
    assert capabilities.supports_agent_workspace_binding is False
    assert capabilities.missing_execution_capabilities(
        ExecutionTopology.STRUCTURED_ORCHESTRATOR
    ) == ["supports_step_reasoning"]


def test_execution_context_floor_still_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "EXECUTION_CONTEXT_TOKENS", None)
    with pytest.raises(RuntimeCapabilityError) as excinfo:
        validate_runtime_capabilities(
            get_backend_descriptor("direct_ollama"),
            BackendRole.EXECUTION,
            dispatch=True,
        )
    assert excinfo.value.code == "provider_context_insufficient"


def test_execution_model_requirement_still_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "EXECUTION_MODEL", "")
    with pytest.raises(RuntimeCapabilityError) as excinfo:
        _validate("direct_ollama", dispatch=True)
    assert excinfo.value.code == "provider_model_unavailable"


def test_role_contract_errors_precede_unavailable_backend_health(monkeypatch):
    unavailable = BackendHealth(
        available=False,
        ready=False,
        status="degraded",
        errors=["synthetic backend health failure"],
        warnings=[],
    )
    descriptor = replace(get_backend_descriptor("direct_ollama"), health=unavailable)

    monkeypatch.setattr(settings, "EXECUTION_MODEL", "")
    with pytest.raises(RuntimeCapabilityError) as excinfo:
        validate_runtime_capabilities(
            descriptor,
            BackendRole.EXECUTION,
            effective_context_tokens=AMPLE_CONTEXT,
            dispatch=True,
        )
    assert excinfo.value.code == "provider_model_unavailable"

    monkeypatch.setattr(settings, "EXECUTION_MODEL", "capability-test-model")
    with pytest.raises(RuntimeCapabilityError) as excinfo:
        validate_runtime_capabilities(
            descriptor,
            BackendRole.EXECUTION,
            effective_context_tokens=AMPLE_CONTEXT,
            dispatch=True,
            execution_topology=ExecutionTopology.AGENT_RUNTIME,
        )
    assert excinfo.value.code == "provider_endpoint_incompatible"

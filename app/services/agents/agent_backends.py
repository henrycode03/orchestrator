"""Backend registry for orchestration model/runtime integrations."""

from __future__ import annotations

import enum
import shutil
import shlex
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services.agents.interfaces import BackendHealthCheck


class ExecutionTopology(str, enum.Enum):
    """How a runtime is expected to carry one accepted execution step.

    POST33-EXEC1 separated two responsibilities that ``supports_step_execution``
    had carried as one boolean.  The production execution loop applies the
    accepted step's structured file operations, portable commands, and
    verification itself (``ExecutorService.execute_file_ops`` /
    ``execute_verification_command``) and only calls the runtime for the
    residual reasoning turn.  That is ``STRUCTURED_ORCHESTRATOR``.  A runtime
    that additionally owns native tools and its own workspace binding offers
    ``AGENT_RUNTIME``.
    """

    STRUCTURED_ORCHESTRATOR = "structured_orchestrator_execution"
    AGENT_RUNTIME = "agent_runtime_execution"


# The capability set each execution topology requires of a backend, by
# ``BackendCapabilities`` field name.  Provider-neutral: eligibility is derived
# from declared capabilities only, never from a backend name.
EXECUTION_TOPOLOGY_REQUIRED_CAPABILITIES: Dict[ExecutionTopology, tuple[str, ...]] = {
    ExecutionTopology.STRUCTURED_ORCHESTRATOR: ("supports_step_reasoning",),
    ExecutionTopology.AGENT_RUNTIME: (
        "supports_step_reasoning",
        "supports_step_execution",
        "supports_tool_execution",
        "supports_agent_workspace_binding",
    ),
}


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared backend capabilities used for routing and operator visibility."""

    supports_planning: bool
    supports_step_execution: bool
    supports_debug_repair: bool
    supports_streaming: bool
    supports_checkpoint_resume: bool
    supports_tool_execution: bool
    supports_json_mode: bool
    mcp_capable: bool = False
    max_context_tokens: Optional[int] = None
    max_parallel_sessions: Optional[int] = None
    reliability_tier: str = "standard"
    latency_tier: str = "standard"
    # POST33-EXEC1 decomposition.  Both default to False so an unmigrated or
    # future descriptor fails closed on the execution boundary.
    #
    # supports_step_reasoning: the runtime accepts a rendered execution-step
    # prompt under an execution system contract and returns a bounded textual
    # step result within the caller's timeout.  This is the only capability
    # the structured-orchestrator execution topology requires.
    #
    # supports_step_execution: the runtime drives an agent-style step
    # lifecycle with provider-native tools.  Retained with its original
    # meaning and its original position in ``to_dict()``.
    #
    # supports_agent_workspace_binding: the runtime binds and mutates a
    # runtime workspace of its own (``bind_runtime_workspace`` /
    # ``execution_cwd_override``), rather than leaving every mutation to the
    # Orchestrator's own workspace-contained mechanics.
    supports_step_reasoning: bool = False
    supports_agent_workspace_binding: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def missing_execution_capabilities(
        self, topology: "ExecutionTopology"
    ) -> list[str]:
        """Capability names this backend lacks for ``topology``."""

        return [
            name
            for name in EXECUTION_TOPOLOGY_REQUIRED_CAPABILITIES[topology]
            if not getattr(self, name, False)
        ]


def resolve_execution_topology(
    capabilities: "BackendCapabilities",
) -> ExecutionTopology:
    """Return the execution topology a backend's declared capabilities support.

    Phase 34-A: the one place an ``ExecutionTopology`` is decided.  Derived from
    declared capabilities only -- never from a backend name -- and fail-closed:
    a backend that does not declare the full agent-runtime capability set is a
    ``STRUCTURED_ORCHESTRATOR`` deployment, in which the Orchestrator owns every
    workspace mutation and the runtime only carries the residual reasoning turn.
    """

    if not capabilities.missing_execution_capabilities(ExecutionTopology.AGENT_RUNTIME):
        return ExecutionTopology.AGENT_RUNTIME
    return ExecutionTopology.STRUCTURED_ORCHESTRATOR


@dataclass(frozen=True)
class BackendLaneTraits:
    """Provider-neutral planning/repair lane traits for governed escalation."""

    structured_output_reliability: str
    repair_convergence: str
    large_context_stability: str
    tool_discipline: str
    evidence_following: str
    latency_cost_class: str
    configured_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendConfigMetadata:
    """Static backend configuration metadata."""

    auth_mode: str
    transport_mode: str
    required_env_vars: List[str]
    supported_prompt_format: str
    prompt_dialect: str
    tool_call_shape: str
    streaming_mode: str
    adaptation_profiles: List[str]
    preferred_retry_strategy: str = "balanced"
    context_window_policy: str = "context_summary"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendHealth:
    """Runtime readiness state used by the operator UI and routing."""

    available: bool
    ready: bool
    status: str
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendDescriptor:
    """Backend metadata exposed to orchestration and operator surfaces."""

    name: str
    display_name: str
    implementation: str
    default_model_family: str
    implemented: bool
    capabilities: BackendCapabilities
    lane_traits: BackendLaneTraits
    config: BackendConfigMetadata
    health: BackendHealth

    @property
    def available(self) -> bool:
        return self.health.available

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["available"] = self.available
        payload["capabilities"] = self.capabilities.to_dict()
        payload["lane_traits"] = replace(
            self.lane_traits,
            configured_available=self.available,
        ).to_dict()
        payload["config"] = self.config.to_dict()
        payload["health"] = self.health.to_dict()
        return payload


class UnsupportedAgentBackendError(ValueError):
    """Raised when the configured backend is unknown or not implemented."""


@dataclass(frozen=True)
class _BackendRegistration:
    descriptor: BackendDescriptor
    health_check: BackendHealthCheck


def _resolve_openclaw_command_candidates() -> List[Path]:
    configured_path = (settings.OPENCLAW_CLI_PATH or "").strip()
    candidates: List[Path] = []
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    detected_path = shutil.which("openclaw")
    if detected_path:
        candidates.append(Path(detected_path))

    for known in (
        "/usr/local/bin/openclaw",
        "/usr/bin/openclaw",
        str(Path.home() / ".local" / "bin" / "openclaw"),
        "/root/.local/bin/openclaw",
        "/opt/openclaw/dist/index.js",
        "/root/.openclaw/app/dist/index.js",
    ):
        candidates.append(Path(known).expanduser())

    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _check_local_openclaw_health(descriptor: BackendDescriptor) -> BackendHealth:
    warnings: List[str] = []
    errors: List[str] = []
    cli_args = (settings.OPENCLAW_CLI_ARGS or "").strip()
    if cli_args:
        try:
            shlex.split(cli_args)
        except ValueError as exc:
            warnings.append(f"OPENCLAW_CLI_ARGS could not be parsed cleanly: {exc}")

    command_found = False
    for candidate in _resolve_openclaw_command_candidates():
        try:
            if candidate.exists():
                command_found = True
                break
        except OSError:
            continue
    if not command_found:
        errors.append(
            "OpenClaw CLI was not found in PATH, OPENCLAW_CLI_PATH, or known install locations."
        )

    return BackendHealth(
        available=command_found,
        ready=not errors,
        status="ready" if not errors else "degraded",
        errors=errors,
        warnings=warnings,
    )


def _check_planned_backend_health(descriptor: BackendDescriptor) -> BackendHealth:
    return BackendHealth(
        available=False,
        ready=False,
        status="not_implemented",
        errors=[
            (
                f"{descriptor.display_name} is registered for future expansion, "
                "but no runtime adapter is implemented yet."
            )
        ],
        warnings=[],
    )


def _check_openai_backend_health(descriptor: BackendDescriptor) -> BackendHealth:
    errors: List[str] = []
    warnings: List[str] = []

    if not (settings.OPENAI_API_KEY or "").strip():
        errors.append("OPENAI_API_KEY is not configured.")

    return BackendHealth(
        available=not errors,
        ready=not errors,
        status="ready" if not errors else "degraded",
        errors=errors,
        warnings=warnings,
    )


def _check_openai_chat_backend_health(descriptor: BackendDescriptor) -> BackendHealth:
    errors: List[str] = []
    warnings: List[str] = []

    if not (
        settings.OPENAI_CHAT_COMPLETIONS_BASE_URL or settings.OPENAI_BASE_URL
    ).strip():
        errors.append(
            "OPENAI_CHAT_COMPLETIONS_BASE_URL or OPENAI_BASE_URL is required."
        )
    if not (
        settings.OPENAI_CHAT_COMPLETIONS_MODEL
        or settings.PLANNER_MODEL
        or settings.AGENT_MODEL
    ).strip():
        errors.append(
            "OPENAI_CHAT_COMPLETIONS_MODEL, PLANNER_MODEL, or AGENT_MODEL is required."
        )
    if not (
        settings.OPENAI_CHAT_COMPLETIONS_API_KEY or settings.OPENAI_API_KEY
    ).strip():
        warnings.append(
            "No OpenAI-compatible chat API key is configured; this is acceptable for local llama.cpp endpoints."
        )

    return BackendHealth(
        available=not errors,
        ready=not errors,
        status="ready" if not errors else "degraded",
        errors=errors,
        warnings=warnings,
    )


def _check_direct_ollama_health(descriptor: BackendDescriptor) -> BackendHealth:
    errors: List[str] = []
    # Config-only check — does not verify Ollama is reachable or model is pulled.
    warnings: List[str] = [
        "Ollama reachability and model availability are not verified at startup; "
        "confirm 'ollama ps' shows the model before running tasks."
    ]

    if not (settings.OLLAMA_BASE_URL or "").strip():
        errors.append("OLLAMA_BASE_URL is not configured.")
    if not (settings.OLLAMA_AGENT_MODEL or "").strip():
        errors.append("OLLAMA_AGENT_MODEL is not configured.")

    return BackendHealth(
        available=not errors,
        ready=not errors,
        status="ready" if not errors else "degraded",
        errors=errors,
        warnings=warnings,
    )


def _default_ollama_model_family() -> str:
    return (settings.OLLAMA_AGENT_MODEL or "").strip() or "local"


def _base_descriptor(
    *,
    name: str,
    display_name: str,
    implementation: str,
    default_model_family: str,
    implemented: bool,
    capabilities: BackendCapabilities,
    lane_traits: BackendLaneTraits,
    config: BackendConfigMetadata,
) -> BackendDescriptor:
    return BackendDescriptor(
        name=name,
        display_name=display_name,
        implementation=implementation,
        default_model_family=default_model_family,
        implemented=implemented,
        capabilities=capabilities,
        lane_traits=lane_traits,
        config=config,
        health=BackendHealth(
            available=False,
            ready=False,
            status="unknown",
            errors=[],
            warnings=[],
        ),
    )


# BACKEND_COUPLING: This registry is OpenClaw-centric. Future transports register here.
# Each entry must supply a BackendHealthCheck callable and a BackendDescriptor.
# See app/services/agents/interfaces.py for the BackendHealthCheck protocol.
_BACKEND_REGISTRY: Dict[str, _BackendRegistration] = {
    "local_openclaw": _BackendRegistration(
        descriptor=_base_descriptor(
            name="local_openclaw",
            display_name="Local OpenClaw",
            implementation="app.services.agents.providers.openclaw_adapter.create_runtime",
            default_model_family="local",
            implemented=True,
            capabilities=BackendCapabilities(
                supports_planning=True,
                supports_step_execution=True,
                supports_debug_repair=True,
                supports_streaming=True,
                supports_checkpoint_resume=True,
                supports_tool_execution=True,
                supports_json_mode=False,
                mcp_capable=False,
                max_context_tokens=128000,
                max_parallel_sessions=1,
                reliability_tier="standard",
                latency_tier="local",
                # OpenClaw owns both the reasoning turn and the native-tool
                # step lifecycle, and binds its own runtime workspace.
                supports_step_reasoning=True,
                supports_agent_workspace_binding=True,
            ),
            lane_traits=BackendLaneTraits(
                structured_output_reliability="variable",
                repair_convergence="bounded",
                large_context_stability="strong",
                tool_discipline="native_tools",
                evidence_following="standard",
                latency_cost_class="local",
            ),
            config=BackendConfigMetadata(
                auth_mode="local_cli",
                transport_mode="cli",
                required_env_vars=[],
                supported_prompt_format="rendered_text_sections",
                prompt_dialect="openclaw_text_sections",
                tool_call_shape="native_cli_tools",
                streaming_mode="subprocess_jsonl",
                adaptation_profiles=["openclaw_default", "qwen_compact_json"],
                preferred_retry_strategy="compact_then_repair",
                context_window_policy="compress_then_retry",
            ),
        ),
        health_check=_check_local_openclaw_health,
    ),
    "remote_openclaw_gateway": _BackendRegistration(
        descriptor=_base_descriptor(
            name="remote_openclaw_gateway",
            display_name="Remote OpenClaw Gateway",
            implementation="app.services.agents.providers.remote_openclaw_adapter.create_runtime",
            default_model_family="gateway_default",
            implemented=False,
            capabilities=BackendCapabilities(
                supports_planning=True,
                supports_step_execution=True,
                supports_debug_repair=True,
                supports_streaming=True,
                supports_checkpoint_resume=False,
                supports_tool_execution=True,
                supports_json_mode=True,
                mcp_capable=False,
                max_context_tokens=None,
                reliability_tier="standard",
                latency_tier="network",
                supports_step_reasoning=True,
                supports_agent_workspace_binding=True,
            ),
            lane_traits=BackendLaneTraits(
                structured_output_reliability="standard",
                repair_convergence="bounded",
                large_context_stability="standard",
                tool_discipline="gateway_tools",
                evidence_following="standard",
                latency_cost_class="network",
            ),
            config=BackendConfigMetadata(
                auth_mode="api_key",
                transport_mode="api",
                required_env_vars=["OPENCLAW_GATEWAY_URL", "OPENCLAW_API_KEY"],
                supported_prompt_format="rendered_text_sections",
                prompt_dialect="openclaw_text_sections",
                tool_call_shape="gateway_tool_schema",
                streaming_mode="http_stream",
                adaptation_profiles=["openclaw_default", "claude_strict_tools"],
                preferred_retry_strategy="schema_first",
                context_window_policy="truncate_context",
            ),
        ),
        health_check=_check_planned_backend_health,
    ),
    "openai_responses_api": _BackendRegistration(
        descriptor=_base_descriptor(
            name="openai_responses_api",
            display_name="OpenAI Responses API",
            implementation="app.services.agents.providers.openai_adapter.create_runtime",
            default_model_family="gpt-5",
            implemented=True,
            capabilities=BackendCapabilities(
                supports_planning=True,
                supports_step_execution=False,
                supports_debug_repair=False,
                supports_streaming=True,
                supports_checkpoint_resume=False,
                supports_tool_execution=False,
                supports_json_mode=True,
                mcp_capable=True,
                max_context_tokens=None,
                reliability_tier="standard",
                latency_tier="network",
                # execute_task() delegates to invoke_prompt() with the generic
                # contract; no execution-step system contract is selected.
                supports_step_reasoning=False,
                supports_agent_workspace_binding=False,
            ),
            lane_traits=BackendLaneTraits(
                structured_output_reliability="high",
                repair_convergence="strong",
                large_context_stability="strong",
                tool_discipline="structured_no_execution",
                evidence_following="strong",
                latency_cost_class="network",
            ),
            config=BackendConfigMetadata(
                auth_mode="api_key",
                transport_mode="api",
                required_env_vars=["OPENAI_API_KEY"],
                supported_prompt_format="structured_prompt_envelope",
                prompt_dialect="responses_json",
                tool_call_shape="responses_tools",
                streaming_mode="responses_stream",
                adaptation_profiles=[
                    "openai_responses_default",
                    "openai_responses_structured",
                ],
                preferred_retry_strategy="structured_retry",
                context_window_policy="summarize_context",
            ),
        ),
        health_check=_check_openai_backend_health,
    ),
    "direct_ollama": _BackendRegistration(
        descriptor=_base_descriptor(
            name="direct_ollama",
            display_name="Direct Ollama",
            implementation="app.services.agents.providers.ollama_adapter.create_runtime",
            default_model_family=_default_ollama_model_family(),
            implemented=True,
            capabilities=BackendCapabilities(
                supports_planning=True,
                supports_step_execution=False,
                supports_debug_repair=False,
                supports_streaming=False,
                supports_checkpoint_resume=False,
                supports_tool_execution=False,
                supports_json_mode=False,
                mcp_capable=False,
                max_context_tokens=4096,
                reliability_tier="local",
                latency_tier="local",
                # execute_task() selects _STEP_SYSTEM and honours the caller's
                # timeout; no native tools and no workspace of its own.
                supports_step_reasoning=True,
                supports_agent_workspace_binding=False,
            ),
            lane_traits=BackendLaneTraits(
                structured_output_reliability="variable",
                repair_convergence="bounded",
                large_context_stability="bounded",
                tool_discipline="no_tools",
                evidence_following="standard",
                latency_cost_class="local",
            ),
            config=BackendConfigMetadata(
                auth_mode="none",
                transport_mode="local_http",
                required_env_vars=[],
                supported_prompt_format="plain_text",
                prompt_dialect="ollama_chat",
                tool_call_shape="none",
                streaming_mode="none",
                adaptation_profiles=["ollama_default"],
                preferred_retry_strategy="schema_first",
                context_window_policy="truncate_context",
            ),
        ),
        health_check=_check_direct_ollama_health,
    ),
    "openai_chat_completions": _BackendRegistration(
        descriptor=_base_descriptor(
            name="openai_chat_completions",
            display_name="OpenAI-Compatible Chat Completions",
            implementation="app.services.agents.providers.openai_chat_adapter.create_runtime",
            default_model_family="local",
            implemented=True,
            capabilities=BackendCapabilities(
                supports_planning=True,
                supports_step_execution=False,
                # The adapter is already used for planning repair and debug
                # repair; this declaration must describe the deployed role
                # path so capability validation does not reject it.
                supports_debug_repair=True,
                supports_streaming=False,
                supports_checkpoint_resume=False,
                supports_tool_execution=False,
                supports_json_mode=False,
                mcp_capable=False,
                max_context_tokens=None,
                reliability_tier="local",
                latency_tier="local",
                # execute_task() selects _STEP_SYSTEM via
                # _execute_task_system_prompt() and honours the caller's
                # timeout; no native tools and no workspace of its own.
                supports_step_reasoning=True,
                supports_agent_workspace_binding=False,
            ),
            lane_traits=BackendLaneTraits(
                structured_output_reliability="variable",
                repair_convergence="bounded",
                large_context_stability="bounded",
                tool_discipline="no_tools",
                evidence_following="standard",
                latency_cost_class="local",
            ),
            config=BackendConfigMetadata(
                auth_mode="optional_api_key",
                transport_mode="local_http",
                required_env_vars=[],
                supported_prompt_format="plain_text",
                prompt_dialect="openai_chat_completions",
                tool_call_shape="none",
                streaming_mode="none",
                adaptation_profiles=["ollama_default"],
                preferred_retry_strategy="schema_first",
                context_window_policy="truncate_context",
            ),
        ),
        health_check=_check_openai_chat_backend_health,
    ),
}


def _resolve_registration(name: Optional[str]) -> Optional[_BackendRegistration]:
    normalized = (name or "").strip().lower() or "local_openclaw"
    return _BACKEND_REGISTRY.get(normalized)


def list_supported_backends() -> List[BackendDescriptor]:
    """Return the currently registered orchestration backends."""

    descriptors: List[BackendDescriptor] = []
    for registration in _BACKEND_REGISTRY.values():
        health = registration.health_check(registration.descriptor)
        descriptor = registration.descriptor
        if descriptor.name == "direct_ollama":
            descriptor = replace(
                descriptor,
                default_model_family=_default_ollama_model_family(),
            )
        descriptors.append(replace(descriptor, health=health))
    return descriptors


def get_backend_descriptor(name: Optional[str]) -> BackendDescriptor:
    """Resolve a configured backend name to a concrete descriptor."""

    registration = _resolve_registration(name)
    if registration is None:
        raise UnsupportedAgentBackendError(
            f"Unsupported orchestration backend: {(name or '').strip() or '<empty>'}"
        )
    health = registration.health_check(registration.descriptor)
    descriptor = registration.descriptor
    if descriptor.name == "direct_ollama":
        descriptor = replace(
            descriptor,
            default_model_family=_default_ollama_model_family(),
        )
    return replace(descriptor, health=health)


def require_backend_descriptor(name: Optional[str]) -> BackendDescriptor:
    """Resolve a backend and reject known-but-unimplemented providers explicitly."""

    descriptor = get_backend_descriptor(name)
    if not descriptor.implemented:
        raise UnsupportedAgentBackendError(
            f"Backend '{descriptor.name}' is registered but not implemented yet."
        )
    return descriptor

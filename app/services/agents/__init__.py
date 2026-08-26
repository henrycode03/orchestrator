"""Agent backend and runtime integrations."""

from .agent_backends import (
    BackendCapabilities,
    BackendConfigMetadata,
    BackendDescriptor,
    BackendHealth,
    BackendLaneTraits,
    ExecutionTopology,
    resolve_execution_topology,
    UnsupportedAgentBackendError,
    get_backend_descriptor,
    list_supported_backends,
    require_backend_descriptor,
)
from .agent_runtime import (
    RuntimeCapabilityError,
    create_agent_runtime,
    invoke_runtime_prompt,
    runtime_reports_context_overflow,
    execution_topology_for_runtime,
    resolve_execution_topology_for_role,
    validate_runtime_capabilities,
)
from .interfaces import (
    AgentRuntime,
    AgentRuntimeError,
    RuntimeBackendResult,
    UnsupportedCapabilityError,
)

__all__ = [
    "BackendCapabilities",
    "ExecutionTopology",
    "resolve_execution_topology",
    "execution_topology_for_runtime",
    "resolve_execution_topology_for_role",
    "BackendConfigMetadata",
    "BackendDescriptor",
    "BackendHealth",
    "BackendLaneTraits",
    "UnsupportedAgentBackendError",
    "get_backend_descriptor",
    "list_supported_backends",
    "require_backend_descriptor",
    "AgentRuntime",
    "AgentRuntimeError",
    "RuntimeBackendResult",
    "UnsupportedCapabilityError",
    "create_agent_runtime",
    "invoke_runtime_prompt",
    "runtime_reports_context_overflow",
    "RuntimeCapabilityError",
    "validate_runtime_capabilities",
]

"""Factory helpers for orchestration runtime backends."""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.services.agents.agent_backends import (
    UnsupportedAgentBackendError,
    require_backend_descriptor,
)
from app.services.agents.interfaces import AgentRuntime
from app.services.agents.providers import get_runtime_factory
from app.services.agents.runtime_configuration import RuntimeConfiguration
from app.services.model_adaptation import (
    get_adaptation_profile,
    require_adaptation_profile,
)
from app.services.workspace.system_settings import get_effective_agent_backend
from app.services.workspace.system_settings import (
    PLANNING_ADAPTATION_PROFILE_KEY,
    get_effective_adaptation_profile,
    get_effective_agent_model_family,
    get_setting_value_runtime,
)

logger = logging.getLogger(__name__)


class BackendRole(str, enum.Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    DEBUG_REPAIR = "debug_repair"
    REPAIR = "repair"
    COMPLETION_REPAIR = "completion_repair"


_TEST_RUNTIME_BACKENDS = {"stub_success", "stub_capacity"}
_PROFILE_OVERRIDE_UNSET = object()


class UnsupportedRuntimeProfileError(ValueError):
    """Raised when an explicit role profile is not supported by its backend."""


def _test_runtime_backends_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_TEST_RUNTIME_BACKENDS", False))


def _is_test_runtime_backend(backend_name: str) -> bool:
    return backend_name in _TEST_RUNTIME_BACKENDS


def resolve_backend_name_for_role(db: Session, role: BackendRole) -> str:
    """Return the configured backend name for role, falling back to AGENT_BACKEND."""
    role_setting = {
        BackendRole.PLANNING: settings.PLANNING_BACKEND,
        BackendRole.EXECUTION: settings.EXECUTION_BACKEND,
        BackendRole.DEBUG_REPAIR: settings.DEBUG_REPAIR_BACKEND
        or settings.REPAIR_BACKEND,
        BackendRole.REPAIR: settings.REPAIR_BACKEND,
        BackendRole.COMPLETION_REPAIR: settings.COMPLETION_REPAIR_BACKEND,
    }.get(role)
    if role_setting:
        return role_setting.strip()
    return get_effective_agent_backend(settings.AGENT_BACKEND, db=db).strip()


def resolve_runtime_configuration(
    db: Session,
    role: BackendRole,
    *,
    backend_override: Optional[str] = None,
    adaptation_profile_override: object = _PROFILE_OVERRIDE_UNSET,
) -> RuntimeConfiguration:
    """Resolve provider-neutral ownership for a role runtime invocation."""

    backend_name = (
        str(backend_override).strip()
        if backend_override
        else resolve_backend_name_for_role(db, role)
    )
    if role is not BackendRole.PLANNING:
        return RuntimeConfiguration(role=role.value, backend_name=backend_name)

    descriptor = require_backend_descriptor(backend_name)
    model_family = (
        str(settings.PLANNER_MODEL or "").strip()
        or get_effective_agent_model_family(settings.AGENT_MODEL, db=db).strip()
        or descriptor.default_model_family
    )
    if adaptation_profile_override is _PROFILE_OVERRIDE_UNSET:
        explicit_profile = get_setting_value_runtime(
            PLANNING_ADAPTATION_PROFILE_KEY,
            settings.PLANNING_ADAPTATION_PROFILE,
            db=db,
        )
    else:
        explicit_profile = str(adaptation_profile_override or "").strip() or None

    if explicit_profile:
        profile = require_adaptation_profile(str(explicit_profile))
        if (
            profile.backend != "*"
            and profile.name not in descriptor.config.adaptation_profiles
        ):
            raise UnsupportedRuntimeProfileError(
                f"Adaptation profile '{profile.name}' is not supported by "
                f"planning backend '{descriptor.name}'."
            )
    else:
        profile = get_adaptation_profile(get_effective_adaptation_profile(db=db))

    return RuntimeConfiguration(
        role=role.value,
        backend_name=descriptor.name,
        model_family=model_family,
        adaptation_profile=profile.name,
    )


def resolve_planning_runtime_configuration(
    db: Session,
    *,
    adaptation_profile_override: object = _PROFILE_OVERRIDE_UNSET,
) -> RuntimeConfiguration:
    """Resolve the single configuration owner used by Planning Sessions."""

    return resolve_runtime_configuration(
        db,
        BackendRole.PLANNING,
        adaptation_profile_override=adaptation_profile_override,
    )


def create_agent_runtime(
    db: Session,
    session_id: Optional[int],
    task_id: Optional[int] = None,
    *,
    use_demo_mode: Optional[bool] = None,
    role: Optional[BackendRole] = None,
    backend_override: Optional[str] = None,
) -> AgentRuntime:
    """Instantiate the configured backend runtime for a session/task pair."""

    runtime_configuration: RuntimeConfiguration | None = None
    if role is BackendRole.PLANNING:
        runtime_configuration = resolve_runtime_configuration(
            db,
            role,
            backend_override=backend_override,
        )
        backend_name = runtime_configuration.backend_name
    elif backend_override:
        backend_name = str(backend_override).strip()
    elif role is not None:
        backend_name = resolve_backend_name_for_role(db, role)
    else:
        backend_name = get_effective_agent_backend(
            settings.AGENT_BACKEND, db=db
        ).strip()
    if _is_test_runtime_backend(backend_name):
        if not _test_runtime_backends_enabled():
            raise UnsupportedAgentBackendError(
                f"Backend '{backend_name}' is test-only and ENABLE_TEST_RUNTIME_BACKENDS is false."
            )
        from app.services.agents.stub_runtime import create_stub_runtime

        return create_stub_runtime(
            db,
            session_id,
            task_id,
            use_demo_mode=use_demo_mode,
            backend_id=backend_name,
        )
    descriptor = require_backend_descriptor(backend_name)
    runtime_factory = get_runtime_factory(descriptor.name)
    if runtime_factory is not None:
        runtime_kwargs: dict[str, Any] = {"use_demo_mode": use_demo_mode}
        if runtime_configuration is not None:
            runtime_kwargs["runtime_configuration"] = runtime_configuration
        runtime = runtime_factory(db, session_id, task_id, **runtime_kwargs)
        if role is not None and hasattr(runtime, "__dict__"):
            runtime.backend_role = role.value
        if runtime_configuration is not None and hasattr(runtime, "__dict__"):
            runtime.runtime_configuration = runtime_configuration
        return runtime

    raise UnsupportedAgentBackendError(
        f"Backend '{descriptor.name}' does not have a registered runtime adapter."
    )


def invoke_runtime_prompt(
    db: Session,
    prompt: str,
    *,
    session_id: Optional[int] = None,
    task_id: Optional[int] = None,
    task_execution_id: Optional[int] = None,
    project_id: Optional[int] = None,
    source_brain: str = "local",
    timeout_seconds: int = 180,
    session_prefix: str = "planning",
    role: Optional[BackendRole] = None,
) -> dict[str, Any]:
    """Execute a one-shot runtime prompt across local or remote backends."""

    runtime = create_agent_runtime(db, session_id, task_id, role=role)
    if project_id is not None and hasattr(runtime, "project_id"):
        runtime.project_id = project_id
    if task_execution_id is not None and hasattr(runtime, "task_execution_id"):
        runtime.task_execution_id = task_execution_id
    try:
        return asyncio.run(
            runtime.invoke_prompt(
                prompt,
                timeout_seconds=timeout_seconds,
                source_brain=source_brain,
                session_prefix=session_prefix,
            )
        )
    except Exception:
        logger.exception(
            "Runtime prompt invocation failed for session_id=%s task_id=%s "
            "task_execution_id=%s",
            session_id,
            task_id,
            task_execution_id,
        )
        raise


def runtime_reports_context_overflow(
    db: Session,
    result: Optional[dict[str, Any]],
    *,
    session_id: Optional[int] = None,
    task_id: Optional[int] = None,
    role: Optional[BackendRole] = None,
) -> bool:
    """Backend-neutral context overflow check for planning retries."""

    runtime = create_agent_runtime(
        db,
        session_id=session_id,
        task_id=task_id,
        role=role,
    )
    return runtime.reports_context_overflow(result)

"""Execution-time executor workspace binding layer (Phase 23D/ORS1).

Phase 23C confirmed a real architectural blocker: OpenClaw's fail-closed
agent-selection guard (Phase 22C-0) requires a configured agent whose
`workspace` field equals the execution cwd exactly, but a Task Execution
Sandbox path is unique per task execution -- no static `openclaw.json`
entry can ever match it, so every runtime-workspace dispatch raised
`OpenClawAgentSelectionError` by design.

This module closes that gap without rewriting the operator's persistent
`openclaw.json` and without creating or deleting any agent identity in it.
Normal Orchestrator dispatch reads the real config read-only, adds one
invocation-only synthetic agent to a private copy, and points that agent at
the current Runtime Workspace. The persistent runner-template selector below
remains available for explicit historical/maintenance callers, but is not a
normal lifecycle prerequisite.
The copy is consumed by exactly one dispatch (via `OPENCLAW_CONFIG_PATH`,
already an existing `OpenClawSessionService._openclaw_config_path()` seam)
and discarded on release -- the same ephemeral-artifact-per-invocation
pattern `git_containment_guard.build_git_containment_env` already uses for
the git shim.

Deliberately independent of OpenClaw's own service module (imports only
`RuntimeExecutorContext`) so a future executor with different workspace
binding semantics can add its own `bind_<executor>_workspace` function here
without this module growing OpenClaw-specific control flow (Goal 5:
Runtime Workspace ownership belongs to Orchestrator; an executor is only a
consumer of a bound context).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import socket
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from app.config import settings
from app.services.orchestration.execution.runtime_context import (
    RuntimeExecutorContext,
)

logger = logging.getLogger(__name__)

# Phase 37 Pre-B-F2B: ephemeral binding artifact ownership contract.
# Every binding lives in one Orchestrator-owned root (never loose in /tmp),
# carries non-sensitive ownership metadata, and holds an exclusive flock on
# its lock file for its whole lifetime. The kernel drops that lock when the
# last holder (worker or inherited OpenClaw child) exits -- including on
# SIGKILL -- so the lock, not PID or age, is the liveness authority used by
# `binding_reconciliation`.
BINDING_ROOT_ENV = "ORCHESTRATOR_OPENCLAW_BINDING_ROOT"
BINDING_DIR_PREFIX = "orchestrator-openclaw-binding-"
BINDING_METADATA_NAME = ".orchestrator-binding.json"
BINDING_LOCK_NAME = ".orchestrator-binding.lock"
BINDING_METADATA_SCHEMA = "orchestrator.openclaw_binding.v1"
BINDING_ARTIFACT_TYPE = "openclaw_runtime_binding"


def binding_artifact_root() -> Path:
    """Return the approved root for ephemeral OpenClaw binding artifacts."""

    configured = os.environ.get(BINDING_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / f"orchestrator-openclaw-bindings-{os.getuid()}"


def validate_binding_artifact_root(root: Path) -> None:
    """Fail closed unless ``root`` is a private directory owned by this uid."""

    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise ExecutorWorkspaceBindingError(
            f"Binding artifact root is unavailable: {root}: {exc}"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ExecutorWorkspaceBindingError(
            f"Binding artifact root is not a real directory: {root}"
        )
    if root_stat.st_uid != os.getuid() or root_stat.st_mode & 0o077:
        raise ExecutorWorkspaceBindingError(
            f"Binding artifact root {root} must be owned by uid {os.getuid()} "
            "with no group/other access"
        )


def _ensure_binding_artifact_root() -> Path:
    root = binding_artifact_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise ExecutorWorkspaceBindingError(
            f"Could not create binding artifact root {root}: {exc}"
        ) from exc
    validate_binding_artifact_root(root)
    return root


def read_boot_id() -> Optional[str]:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def read_pid_namespace() -> Optional[str]:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def read_process_start_ticks(pid: int) -> Optional[int]:
    """Return a PID's start time (clock ticks since boot) to detect PID reuse."""

    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        # Field 22; the comm field (2) may contain spaces, so split after ')'.
        return int(raw.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _binding_metadata(
    tmp_dir: Path, binding_id: str, context: RuntimeExecutorContext
) -> Dict[str, Any]:
    pid = os.getpid()
    return {
        "schema": BINDING_METADATA_SCHEMA,
        "artifact_type": BINDING_ARTIFACT_TYPE,
        "binding_id": binding_id,
        "artifact_dir_name": tmp_dir.name,
        "created_at_epoch": time.time(),
        "owner_service": "orchestrator.executor_workspace_binding",
        "owner_hostname": socket.gethostname(),
        "owner_uid": os.getuid(),
        "owner_pid": pid,
        "owner_pid_start_ticks": read_process_start_ticks(pid),
        "owner_pid_namespace": read_pid_namespace(),
        "boot_id": read_boot_id(),
        "task_execution_id": getattr(context, "task_execution_id", None),
        "liveness": "exclusive_flock:" + BINDING_LOCK_NAME,
        "cleanup_eligibility": (
            "only when the binding lock is free and metadata revalidates; "
            "operator --apply required"
        ),
    }


def _write_private_file(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


class ExecutorWorkspaceBindingError(Exception):
    """Raised when a Runtime Workspace cannot be bound to an executor.

    Callers must fail closed: never fall back to the Project Workspace,
    never fall back to an executor's default/static configuration, and
    never invent a new agent/executor identity to route around this.
    """


RUNNER_AGENT_ID_ENV = "OPENCLAW_RUNNER_AGENT_ID"
EPHEMERAL_AGENT_ID = "orchestrator-runtime"


@dataclass(frozen=True)
class OpenClawTemplateSelection:
    """The explicit template identity and its persistent workspace."""

    agent_id: str
    persistent_workspace: Path


def resolve_openclaw_runner_agent_id(
    configured_agent_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve the explicit runner identity without heuristic fallbacks."""

    explicit = str(configured_agent_id or "").strip()
    if explicit:
        return explicit
    environment_value = os.environ.get(RUNNER_AGENT_ID_ENV, "").strip()
    if environment_value:
        return environment_value
    configured = str(getattr(settings, "OPENCLAW_RUNNER_AGENT_ID", "") or "").strip()
    return configured or None


def _path_is_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def validate_runtime_workspace_context(context: RuntimeExecutorContext) -> None:
    """Validate the runtime/project relationship before creating a binding.

    This is the safety portion of the old runner admission contract. It does
    not inspect OpenClaw identities, so workspace containment remains enforced
    even when the persistent operator config contains only ``main``.
    """

    runtime_root_value = getattr(context, "runtime_root", None)
    if not runtime_root_value:
        raise ExecutorWorkspaceBindingError(
            "Runtime Workspace binding has no approved Orchestrator runtime root"
        )
    runtime_root = Path(runtime_root_value).expanduser().resolve(strict=False)
    project_workspace = (
        Path(context.project_workspace).expanduser().resolve(strict=False)
    )
    runtime_workspace = (
        Path(context.runtime_workspace).expanduser().resolve(strict=False)
    )
    if not runtime_root.exists() or not runtime_root.is_dir():
        raise ExecutorWorkspaceBindingError(
            f"Approved Orchestrator runtime root does not exist: {runtime_root}"
        )
    if _path_is_within(project_workspace, runtime_root) or _path_is_within(
        runtime_root, project_workspace
    ):
        raise ExecutorWorkspaceBindingError(
            "Project Workspace and approved Orchestrator runtime root overlap; "
            "refusing ambiguous workspace ownership"
        )
    if _path_is_within(runtime_workspace, project_workspace):
        raise ExecutorWorkspaceBindingError(
            f"Runtime Workspace {runtime_workspace} is inside Project Workspace "
            f"{project_workspace}"
        )
    if not _path_is_within(runtime_workspace, runtime_root):
        raise ExecutorWorkspaceBindingError(
            f"Runtime Workspace {runtime_workspace} is outside approved runtime "
            f"root {runtime_root}"
        )
    if not runtime_workspace.exists() or not runtime_workspace.is_dir():
        raise ExecutorWorkspaceBindingError(
            "Runtime Workspace does not exist as a directory: " f"{runtime_workspace}"
        )


def _find_template_agent_id(config: Dict[str, Any], workspace: Path) -> Optional[str]:
    """Preserve the legacy F12 audit matcher; never use it for dispatch.

    Phase 31's read-only launch-precondition report still needs to identify
    historical ProjectRoot registrations. Normal execution must use
    ``select_runtime_owned_openclaw_template`` and never call this helper.
    """

    project_root = Path(workspace).expanduser().resolve(strict=False)
    matches = [
        str(agent.get("id") or "").strip()
        for agent in (config.get("agents") or {}).get("list") or []
        if isinstance(agent, dict)
        and str(agent.get("id") or "").strip()
        and str(agent.get("workspace") or "").strip()
        and Path(str(agent.get("workspace") or "")).expanduser().resolve(strict=False)
        == project_root
    ]
    if len(matches) > 1:
        raise ExecutorWorkspaceBindingError(
            "Multiple OpenClaw agents are configured with a workspace matching "
            f"Project Workspace {project_root}: {matches}; refusing heuristic "
            "selection."
        )
    return matches[0] if matches else None


def validate_runtime_owned_openclaw_agent(
    config: Dict[str, Any],
    *,
    agent_id: Optional[str],
    project_workspace: Path,
    runtime_root: Path,
) -> OpenClawTemplateSelection:
    """Validate and return one explicit runner's persistent workspace."""

    resolved_id = str(agent_id or "").strip()
    if not resolved_id:
        raise ExecutorWorkspaceBindingError(
            "OpenClaw runner agent ID is not configured; refusing implicit "
            "main, ProjectRoot, nearest-workspace, or generic-workspace selection."
        )
    approved_root = Path(runtime_root).expanduser().resolve(strict=False)
    project_root = Path(project_workspace).expanduser().resolve(strict=False)
    if not approved_root.exists() or not approved_root.is_dir():
        raise ExecutorWorkspaceBindingError(
            f"Approved Orchestrator runtime root does not exist: {approved_root}"
        )
    if _path_is_within(project_root, approved_root) or _path_is_within(
        approved_root, project_root
    ):
        raise ExecutorWorkspaceBindingError(
            "Project Workspace and approved Orchestrator runtime root overlap; "
            "refusing ambiguous workspace ownership"
        )

    agents = (config.get("agents") or {}).get("list") or []
    matches = [
        agent
        for agent in agents
        if isinstance(agent, dict) and str(agent.get("id") or "").strip() == resolved_id
    ]
    if len(matches) != 1:
        if not matches:
            raise ExecutorWorkspaceBindingError(
                f"Configured OpenClaw runner agent {resolved_id!r} was not found; "
                "refusing fallback selection"
            )
        raise ExecutorWorkspaceBindingError(
            f"Multiple OpenClaw runner entries use ID {resolved_id!r}; refusing "
            "conflicting identity selection"
        )

    workspace_value = str(matches[0].get("workspace") or "").strip()
    if not workspace_value:
        raise ExecutorWorkspaceBindingError(
            f"OpenClaw runner agent {resolved_id!r} has no persistent workspace"
        )
    persistent_workspace = Path(workspace_value).expanduser().resolve(strict=False)
    if not persistent_workspace.exists() or not persistent_workspace.is_dir():
        raise ExecutorWorkspaceBindingError(
            "OpenClaw runner workspace does not exist as a directory: "
            f"{persistent_workspace}"
        )
    if not _path_is_within(persistent_workspace, approved_root):
        raise ExecutorWorkspaceBindingError(
            f"OpenClaw runner workspace {persistent_workspace} is outside approved "
            f"runtime root {approved_root}"
        )
    if _path_is_within(persistent_workspace, project_root) or _path_is_within(
        project_root, persistent_workspace
    ):
        raise ExecutorWorkspaceBindingError(
            f"OpenClaw runner workspace {persistent_workspace} overlaps Project "
            f"Workspace {project_root}"
        )
    return OpenClawTemplateSelection(
        agent_id=resolved_id,
        persistent_workspace=persistent_workspace,
    )


def select_runtime_owned_openclaw_template(
    config: Dict[str, Any],
    context: RuntimeExecutorContext,
    *,
    configured_agent_id: Optional[str] = None,
) -> OpenClawTemplateSelection:
    """Select one explicit runtime-owned OpenClaw template."""

    agent_id = resolve_openclaw_runner_agent_id(configured_agent_id)
    if not agent_id:
        raise ExecutorWorkspaceBindingError(
            "OpenClaw runner agent ID is not configured; refusing implicit "
            "main, ProjectRoot, nearest-workspace, or generic-workspace selection."
        )

    runtime_root_value = getattr(context, "runtime_root", None)
    if not runtime_root_value:
        raise ExecutorWorkspaceBindingError(
            "Runtime Workspace binding has no approved Orchestrator runtime root"
        )
    runtime_root = Path(runtime_root_value).expanduser().resolve(strict=False)
    project_workspace = (
        Path(context.project_workspace).expanduser().resolve(strict=False)
    )
    runtime_workspace = (
        Path(context.runtime_workspace).expanduser().resolve(strict=False)
    )
    if not runtime_root.exists() or not runtime_root.is_dir():
        raise ExecutorWorkspaceBindingError(
            f"Approved Orchestrator runtime root does not exist: {runtime_root}"
        )
    if _path_is_within(project_workspace, runtime_root) or _path_is_within(
        runtime_root, project_workspace
    ):
        raise ExecutorWorkspaceBindingError(
            "Project Workspace and approved Orchestrator runtime root overlap; "
            "refusing ambiguous workspace ownership"
        )
    if _path_is_within(runtime_workspace, project_workspace):
        raise ExecutorWorkspaceBindingError(
            f"Runtime Workspace {runtime_workspace} is inside Project Workspace "
            f"{project_workspace}"
        )
    if not _path_is_within(runtime_workspace, runtime_root):
        raise ExecutorWorkspaceBindingError(
            f"Runtime Workspace {runtime_workspace} is outside approved runtime "
            f"root {runtime_root}"
        )
    if not runtime_workspace.exists() or not runtime_workspace.is_dir():
        raise ExecutorWorkspaceBindingError(
            "Runtime Workspace does not exist as a directory: " f"{runtime_workspace}"
        )

    return validate_runtime_owned_openclaw_agent(
        config,
        agent_id=agent_id,
        project_workspace=project_workspace,
        runtime_root=runtime_root,
    )


@dataclass
class ExecutorWorkspaceBinding:
    """An active, per-invocation binding. Must be released via `release()`."""

    agent_id: str
    persistent_workspace: Optional[Path]
    config_path: Path
    _tmp_dir: Path
    environment: Dict[str, str]
    binding_id: Optional[str] = None
    _lock_fd: Optional[int] = None
    _unregister_forced_cleanup: Optional[Callable[[], None]] = field(
        default=None, repr=False
    )

    def subprocess_pass_fds(self) -> Tuple[int, ...]:
        """FDs an OpenClaw child must inherit so it keeps the binding live."""

        return (self._lock_fd,) if self._lock_fd is not None else ()

    def release(self) -> None:
        """Remove the ephemeral config copy, then drop the lock. Never raises.

        Idempotent. A failed removal is logged (path and binding id only) and
        the lock is still dropped, so the residue becomes discoverable by
        `binding_reconciliation` instead of being silently forgotten.
        """
        unregister = self._unregister_forced_cleanup
        self._unregister_forced_cleanup = None
        if unregister is not None:
            unregister()
        failures = []
        try:
            if os.path.lexists(self._tmp_dir):
                shutil.rmtree(
                    self._tmp_dir,
                    onerror=lambda _fn, path, exc_info: failures.append(
                        (path, exc_info[0].__name__)
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise
            failures.append((str(self._tmp_dir), type(exc).__name__))
        finally:
            lock_fd, self._lock_fd = self._lock_fd, None
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
        if failures:
            logger.warning(
                "[EXECUTOR_WORKSPACE_BINDING] binding_release_failed %s",
                json.dumps(
                    {
                        "binding_id": self.binding_id,
                        "artifact_dir": str(self._tmp_dir),
                        "failures": failures[:10],
                        "residue_discoverable_by": "binding_reconciliation",
                    },
                    sort_keys=True,
                ),
            )


def bind_openclaw_workspace(
    context: RuntimeExecutorContext,
    *,
    real_config_path: Path,
    runner_agent_id: Optional[str] = None,
    model_ref: Optional[str] = None,
) -> ExecutorWorkspaceBinding:
    """Bind an OpenClaw agent's workspace to `context.runtime_workspace`.

    Reads `real_config_path` (the real, persistent `openclaw.json`) once,
    read-only. Normal dispatch adds a synthetic invocation-only agent to a
    private temp copy and binds it to `context.runtime_workspace`.

    ``runner_agent_id`` is retained only for explicit historical callers that
    still need the old persistent-template adapter. Normal Orchestrator code
    deliberately leaves it unset and never consults environment/default
    runner identity.
    """

    try:
        real_config = json.loads(Path(real_config_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExecutorWorkspaceBindingError(
            f"Could not read OpenClaw config at {real_config_path}: {exc}"
        ) from exc

    validate_runtime_workspace_context(context)
    legacy_selection = None
    if runner_agent_id is not None:
        legacy_selection = select_runtime_owned_openclaw_template(
            real_config,
            context,
            configured_agent_id=runner_agent_id,
        )
        agent_id = legacy_selection.agent_id
    else:
        agent_id = EPHEMERAL_AGENT_ID
    resolved_model_ref = str(model_ref or "").strip() or None
    if resolved_model_ref is None and legacy_selection is None:
        raise ExecutorWorkspaceBindingError(
            "Explicit OpenClaw runtime model is required; refusing "
            "persistent/default model authority"
        )

    bound_config = json.loads(json.dumps(real_config))  # cheap deep copy
    agents = bound_config.setdefault("agents", {}).setdefault("list", [])
    if legacy_selection is None and any(
        isinstance(agent, dict) and str(agent.get("id") or "").strip() == agent_id
        for agent in agents
    ):
        raise ExecutorWorkspaceBindingError(
            f"Ephemeral OpenClaw agent ID {agent_id!r} collides with the "
            "operator config; refusing ambiguous identity selection"
        )

    tmp_dir = Path(
        tempfile.mkdtemp(prefix=BINDING_DIR_PREFIX, dir=_ensure_binding_artifact_root())
    )
    binding_id = uuid.uuid4().hex
    lock_fd: Optional[int] = None
    try:
        # Take the liveness lock before anything else exists in the directory
        # so a reconciler can never observe valid metadata with a free lock
        # while this process is alive.
        lock_fd = os.open(
            tmp_dir / BINDING_LOCK_NAME,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_private_file(
            tmp_dir / BINDING_METADATA_NAME,
            json.dumps(_binding_metadata(tmp_dir, binding_id, context), indent=2),
        )
        state_dir = tmp_dir / "state"
        agent_dir = tmp_dir / "agent"
        state_dir.mkdir(mode=0o700)
        agent_dir.mkdir(mode=0o700)
        main_agent = next(
            (
                agent
                for agent in (real_config.get("agents") or {}).get("list") or []
                if isinstance(agent, dict) and agent.get("id") == "main"
            ),
            None,
        )
        main_agent_dir = str((main_agent or {}).get("agentDir") or "").strip()
        auth_profiles = Path(main_agent_dir).expanduser() / "auth-profiles.json"
        if main_agent_dir and auth_profiles.is_file():
            # OpenClaw resolves provider credentials relative to agentDir. Copy
            # only the operator-owned auth profile into the invocation
            # directory; never point the ephemeral agent at persistent main
            # state and never modify the source file. The copy is owner-only
            # and lives exactly as long as this binding.
            copied_auth = agent_dir / "auth-profiles.json"
            shutil.copyfile(auth_profiles, copied_auth)
            os.chmod(copied_auth, 0o600)
        _populate_bound_config(
            bound_config,
            agents=agents,
            agent_id=agent_id,
            legacy_selection=legacy_selection,
            resolved_model_ref=resolved_model_ref,
            context=context,
            agent_dir=agent_dir,
            state_dir=state_dir,
        )
        config_path = tmp_dir / "openclaw.json"
        # The copy may carry provider settings from the operator config:
        # create it owner-only before writing any content.
        _write_private_file(config_path, "")
        config_path.write_text(json.dumps(bound_config, indent=2), encoding="utf-8")
    except BaseException:
        # Partial initialization: never leave a half-built (possibly
        # credential-bearing) directory behind for this process.
        ExecutorWorkspaceBinding(
            agent_id=agent_id,
            persistent_workspace=None,
            config_path=tmp_dir / "openclaw.json",
            _tmp_dir=tmp_dir,
            environment={},
            binding_id=binding_id,
            _lock_fd=lock_fd,
        ).release()
        raise
    environment = {
        "OPENCLAW_CONFIG_PATH": str(config_path),
        "OPENCLAW_STATE_DIR": str(state_dir),
    }

    logger.info(
        "[EXECUTOR_WORKSPACE_BINDING] Bound OpenClaw agent %s workspace "
        "%s -> %s for task_execution_id=%s (template workspace %s; ephemeral config at %s; "
        "persistent %s untouched; binding_id=%s)",
        agent_id,
        legacy_selection.persistent_workspace if legacy_selection else None,
        context.runtime_workspace,
        context.task_execution_id,
        legacy_selection.persistent_workspace if legacy_selection else None,
        config_path,
        real_config_path,
        binding_id,
    )
    binding = ExecutorWorkspaceBinding(
        agent_id=agent_id,
        persistent_workspace=(
            legacy_selection.persistent_workspace if legacy_selection else None
        ),
        config_path=config_path,
        _tmp_dir=tmp_dir,
        environment=environment,
        binding_id=binding_id,
        _lock_fd=lock_fd,
    )
    # A forced SIGTERM releases every binding this process still owns, not
    # only the worker-level ones (nested repair/fallback runtimes included).
    from app.services.agents.subprocess_lifecycle import (
        register_forced_termination_cleanup,
    )

    binding._unregister_forced_cleanup = register_forced_termination_cleanup(
        binding.release
    )
    return binding


def _populate_bound_config(
    bound_config: Dict[str, Any],
    *,
    agents: list,
    agent_id: str,
    legacy_selection: Optional[OpenClawTemplateSelection],
    resolved_model_ref: Optional[str],
    context: RuntimeExecutorContext,
    agent_dir: Path,
    state_dir: Path,
) -> None:
    if legacy_selection is None:
        agents.append(
            {
                "id": agent_id,
                "workspace": str(context.runtime_workspace),
                "agentDir": str(agent_dir),
                "model": {
                    "primary": resolved_model_ref,
                    "fallbacks": [],
                },
            }
        )
    else:
        for agent in agents:
            if (
                isinstance(agent, dict)
                and str(agent.get("id") or "").strip() == agent_id
            ):
                agent["workspace"] = str(context.runtime_workspace)
                agent["agentDir"] = str(agent_dir)
                if resolved_model_ref is not None:
                    agent["model"] = {
                        "primary": resolved_model_ref,
                        "fallbacks": [],
                    }

    defaults = (bound_config.setdefault("agents", {})).setdefault("defaults", {})
    # OpenClaw 2026.4.10 owns this control at agents.defaults.  Agent entries
    # are strict and reject the same key, so keep bootstrap suppression in the
    # defaults object while retaining the per-invocation state directory below.
    defaults["workspace"] = str(context.runtime_workspace)
    defaults["skipBootstrap"] = True
    session = bound_config.setdefault("session", {})
    session["store"] = str(state_dir / "sessions.json")

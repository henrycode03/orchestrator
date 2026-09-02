"""Execution-time executor workspace binding layer (Phase 23D).

Phase 23C confirmed a real architectural blocker: OpenClaw's fail-closed
agent-selection guard (Phase 22C-0) requires a configured agent whose
`workspace` field equals the execution cwd exactly, but a Task Execution
Sandbox path is unique per task execution -- no static `openclaw.json`
entry can ever match it, so every runtime-workspace dispatch raised
`OpenClawAgentSelectionError` by design.

This module closes that gap without rewriting the operator's persistent
`openclaw.json` and without creating or deleting any agent identity in it.
It reads the real config read-only, selects the explicitly configured
runtime-owned runner template, and writes a private, ephemeral copy of that
config with only that one agent's `workspace` field rewritten to the current
Runtime Workspace.
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

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings
from app.services.orchestration.execution.runtime_context import (
    RuntimeExecutorContext,
)

logger = logging.getLogger(__name__)


class ExecutorWorkspaceBindingError(Exception):
    """Raised when a Runtime Workspace cannot be bound to an executor.

    Callers must fail closed: never fall back to the Project Workspace,
    never fall back to an executor's default/static configuration, and
    never invent a new agent/executor identity to route around this.
    """


RUNNER_AGENT_ID_ENV = "OPENCLAW_RUNNER_AGENT_ID"


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
    persistent_workspace: Path
    config_path: Path
    _tmp_dir: Path
    environment: Dict[str, str]

    def release(self) -> None:
        """Best-effort cleanup of the ephemeral config copy. Never raises."""
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001 - cleanup must never raise
            logger.warning(
                "[EXECUTOR_WORKSPACE_BINDING] Failed to remove temp config " "dir %s",
                self._tmp_dir,
                exc_info=True,
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
    read-only. Selects the explicitly configured runtime-owned runner
    template and writes a private temp copy of the whole config with only
    that agent's `workspace` field rewritten to
    `context.runtime_workspace`.

    Raises `ExecutorWorkspaceBindingError` if the explicit runner is absent
    or unsafe. This never registers a new agent or falls back to a Project
    Workspace/default/generic match.
    """

    try:
        real_config = json.loads(Path(real_config_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExecutorWorkspaceBindingError(
            f"Could not read OpenClaw config at {real_config_path}: {exc}"
        ) from exc

    selection = select_runtime_owned_openclaw_template(
        real_config,
        context,
        configured_agent_id=runner_agent_id,
    )
    agent_id = selection.agent_id
    resolved_model_ref = None
    if model_ref is not None:
        resolved_model_ref = str(model_ref).strip()
        if not resolved_model_ref:
            raise ExecutorWorkspaceBindingError(
                "Explicit OpenClaw runtime model is empty; refusing "
                "persistent/default model authority"
            )

    bound_config = json.loads(json.dumps(real_config))  # cheap deep copy
    tmp_dir = Path(tempfile.mkdtemp(prefix="orchestrator-openclaw-binding-"))
    state_dir = tmp_dir / "state"
    agent_dir = tmp_dir / "agent"
    state_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    for agent in (bound_config.get("agents") or {}).get("list") or []:
        if isinstance(agent, dict) and str(agent.get("id") or "").strip() == agent_id:
            agent["workspace"] = str(context.runtime_workspace)
            agent["agentDir"] = str(agent_dir)
            if resolved_model_ref is not None:
                # OpenClaw 2026.4.10 resolves an explicit agent model before
                # agents.defaults.model. Empty fallbacks make this invocation
                # fail closed instead of silently changing provider/model.
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

    config_path = tmp_dir / "openclaw.json"
    config_path.write_text(json.dumps(bound_config, indent=2), encoding="utf-8")
    environment = {
        "OPENCLAW_CONFIG_PATH": str(config_path),
        "OPENCLAW_STATE_DIR": str(state_dir),
    }

    logger.info(
        "[EXECUTOR_WORKSPACE_BINDING] Bound OpenClaw agent %s workspace "
        "%s -> %s for task_execution_id=%s (template workspace %s; ephemeral config at %s; "
        "persistent %s untouched)",
        agent_id,
        selection.persistent_workspace,
        context.runtime_workspace,
        context.task_execution_id,
        selection.persistent_workspace,
        config_path,
        real_config_path,
    )
    return ExecutorWorkspaceBinding(
        agent_id=agent_id,
        persistent_workspace=selection.persistent_workspace,
        config_path=config_path,
        _tmp_dir=tmp_dir,
        environment=environment,
    )

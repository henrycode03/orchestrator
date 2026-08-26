"""Accepted-step execution capability core invariant (Phase 34-A).

One invariant: ``ACCEPTED_STEP_EXECUTION_CAPABILITY_COMPLETE``.  Every accepted
mutating step must have at least one execution channel that can actually
perform its mutation:

* **E1** -- a structured file operation the Orchestrator applies under APA;
* **E2** -- a shell command the local command policy can execute itself;
* **E5** -- an explicitly resolved ``AGENT_RUNTIME`` topology whose runtime owns
  native workspace side effects.

A residual text-only reasoning turn (**E4**) is not an execution channel.  Under
``STRUCTURED_ORCHESTRATOR`` a step that demands mutation through a shell command
E2 will refuse therefore has no channel at all, and the plan must be repaired
before the execution loop can dispatch it to a runtime that cannot mutate
anything.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.agents.agent_backends import ExecutionTopology
from app.services.orchestration.validation.local_command_policy import (
    local_shell_command_is_executable,
)

# The mutating shell forms plan admission recognizes.  Deliberately identical to
# the read-only-stage detector in ``core_file_ops`` so one plan cannot be
# "mutating" for one rule and "non-mutating" for the other.
MUTATING_COMMAND_PATTERNS = (
    re.compile(r"(^|[;&|]\s*)(mkdir|touch|cp|mv|rm)\b"),
    re.compile(r"\bsed\s+-i\b"),
    re.compile(r">\s*[^&\s]"),
    re.compile(r"\btee\s+"),
)


def command_requires_mutation(command: Any) -> bool:
    """True when a planned shell command is a workspace-mutating form."""

    command_text = str(command or "").strip()
    if not command_text:
        return False
    patterns = MUTATING_COMMAND_PATTERNS
    if command_text.startswith(("python -c ", "python3 -c ")):
        # Inline python verification scripts are read-only by contract; only the
        # unambiguous mutation forms apply.
        patterns = (
            MUTATING_COMMAND_PATTERNS[0],
            MUTATING_COMMAND_PATTERNS[1],
            MUTATING_COMMAND_PATTERNS[3],
        )
    return any(pattern.search(command_text) for pattern in patterns)


def step_unexecutable_mutating_commands(
    step: Dict[str, Any], project_dir: Path
) -> List[str]:
    """Mutating commands in one step that no Orchestrator-owned channel can run."""

    findings: List[str] = []
    for command in step.get("commands", []) or []:
        command_text = str(command or "").strip()
        if not command_text or not command_requires_mutation(command_text):
            continue
        if local_shell_command_is_executable(command_text, project_dir):
            continue
        findings.append(command_text)
    return findings


def plan_steps_without_execution_channel(
    plan: List[Dict[str, Any]],
    *,
    project_dir: Optional[Path],
    execution_topology: ExecutionTopology,
) -> Dict[int, List[str]]:
    """Return accepted-step mutations with no capable execution channel.

    Keyed by step number, valued by the offending commands.  ``AGENT_RUNTIME``
    deployments return nothing: the runtime owns native workspace side effects
    (E5), so a shell form the local policy refuses is still executable there.
    """

    if execution_topology is not ExecutionTopology.STRUCTURED_ORCHESTRATOR:
        return {}
    if project_dir is None:
        return {}
    resolved_dir = Path(project_dir)
    findings: Dict[int, List[str]] = {}
    for index, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            continue
        try:
            step_number = int(step.get("step_number", index))
        except (TypeError, ValueError):
            step_number = index
        offending = step_unexecutable_mutating_commands(step, resolved_dir)
        if offending:
            findings[step_number] = offending
    return findings

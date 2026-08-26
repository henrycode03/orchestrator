"""Local shell command policy (Phase 34-A).

The safe-local-shell classifier the execution loop's E2 channel uses to decide
whether the Orchestrator can execute an accepted step's shell command itself.
Extracted verbatim from ``phases/execution_local_steps.py`` so plan admission
can ask the same question the executor will later answer, without importing the
execution phase module (which imports ``ValidatorService``).

Leaf module: it may only import workspace-guard primitives.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from app.services.orchestration.validation.workspace_guard import (
    TaskWorkspaceViolationError,
    normalize_path_reference,
)


def _is_safe_local_shell_command(command: str) -> bool:
    normalized = " ".join(str(command or "").strip().split())
    if not normalized:
        return False
    blocked_tokens = (
        "$(",
        "`",
        "curl",
        "wget",
        "pip",
        "npm",
        "yarn",
        "apt",
        "yum",
        "brew",
        "rm ",
        "rm\t",
        "sudo",
        "; rm",
        "&&rm",
        "||rm",
        "..",
    )
    if any(t in normalized for t in blocked_tokens):
        return False
    safe_prefixes = (
        "echo ",
        "echo\t",
        "printf ",
        "mkdir ",
        "mkdir\t",
        "touch ",
        "cp ",
        "cp\t",
        "mv ",
        "mv\t",
        "chmod ",
        "chmod\t",
    )
    return any(normalized.startswith(p) for p in safe_prefixes)


def _is_workspace_local_path_token(token: str, project_dir: Path) -> bool:
    raw = str(token or "").strip()
    if not raw:
        return False
    if re.match(r"^[A-Za-z]:[\\/]", raw) or raw.startswith(("\\\\", "//")):
        return False
    try:
        normalize_path_reference(raw, project_dir)
    except TaskWorkspaceViolationError:
        return False
    return True


def _local_shell_command_paths_are_safe(command: str, project_dir: Path) -> bool:
    if re.search(r"(^|[\s>])([A-Za-z]:[\\/]|\\\\)", str(command or "")):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False

    executable = tokens[0]
    if executable in {"echo", "printf"}:
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in {">", ">>"}:
                if index + 1 >= len(tokens):
                    return False
                if not _is_workspace_local_path_token(tokens[index + 1], project_dir):
                    return False
                index += 2
                continue
            if token.startswith((">", ">>")):
                target = token[2:] if token.startswith(">>") else token[1:]
                if not _is_workspace_local_path_token(target, project_dir):
                    return False
            index += 1
        return True

    if executable in {"mkdir", "touch"}:
        operands = [token for token in tokens[1:] if not token.startswith("-")]
        return bool(operands) and all(
            _is_workspace_local_path_token(token, project_dir) for token in operands
        )

    if executable in {"cp", "mv"}:
        operands = [token for token in tokens[1:] if not token.startswith("-")]
        return len(operands) >= 2 and all(
            _is_workspace_local_path_token(token, project_dir) for token in operands
        )

    if executable == "chmod":
        operands = [token for token in tokens[1:] if not token.startswith("-")]
        if len(operands) < 2:
            return False
        mode = operands[0]
        if not re.match(r"^(?:[0-7]{3,4}|[ugoa]*[+-=][rwxXstugo,+-=]+)$", mode):
            return False
        paths = operands[1:]
        return all(
            _is_workspace_local_path_token(token, project_dir) for token in paths
        )

    return False


def local_shell_command_is_executable(command: str, project_dir: Path) -> bool:
    """True when the Orchestrator's own local command channel (E2) can run this.

    This is the executability question plan admission must ask: a mutating
    command that fails here has no E2 execution channel, and neither structured
    operations nor a residual reasoning turn can carry it.
    """

    if not _is_safe_local_shell_command(command):
        return False
    return _local_shell_command_paths_are_safe(command, project_dir)

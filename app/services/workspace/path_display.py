"""Helpers for presenting workspace paths consistently in prompts and UI text."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.services.workspace.system_settings import get_effective_workspace_root


def render_workspace_path_for_prompt(
    path: Optional[str | Path],
    *,
    db: Optional[Session] = None,
) -> str:
    """Render a path using the configured workspace root when possible.

    This keeps model-facing prompts aligned with the user-configured runtime
    root (for example `/root/.openclaw/workspace/...`) instead of leaking host
    paths like `/home/...`.
    """

    if not path:
        return "Current task workspace"

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return str(candidate).replace("\\", "/")

    resolved = candidate.resolve()
    workspace_root = get_effective_workspace_root(db=db).resolve()

    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError:
        return resolved.name or str(resolved)

    rendered = workspace_root / relative
    return str(rendered).replace("\\", "/")

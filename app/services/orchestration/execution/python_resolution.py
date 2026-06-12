from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def resolve_project_python(project_dir: Path) -> str:
    """Resolve the interpreter verification should use for a project.

    Preference order:
    1. Project-local `.venv/bin/python`
    2. Project-local `venv/bin/python`
    3. System `python3`
    4. System `python`
    5. Current interpreter as a final fallback
    """

    for candidate in (
        project_dir / ".venv" / "bin" / "python",
        project_dir / "venv" / "bin" / "python",
    ):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    for command in ("python3", "python"):
        resolved = shutil.which(command)
        if resolved:
            return resolved

    return sys.executable or "python3"

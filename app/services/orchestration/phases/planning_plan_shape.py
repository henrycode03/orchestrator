"""Planning plan-shape helpers."""

from __future__ import annotations

from typing import Any

from app.services.orchestration.phases.planning_verification import (
    _python_exists_verification_command,
)


def split_repaired_single_step_full_lifecycle_plan(
    extracted_plan: Any,
) -> list[dict[str, Any]] | None:
    if not isinstance(extracted_plan, list) or len(extracted_plan) != 1:
        return None
    original = extracted_plan[0]
    if not isinstance(original, dict):
        return None

    ops = original.get("ops") if isinstance(original.get("ops"), list) else []
    commands = (
        original.get("commands") if isinstance(original.get("commands"), list) else []
    )
    commands = [
        str(command or "").strip() for command in commands if str(command or "").strip()
    ]
    expected_files = (
        original.get("expected_files")
        if isinstance(original.get("expected_files"), list)
        else []
    )
    expected_files = [
        str(path or "").strip().lstrip("./")
        for path in expected_files
        if str(path or "").strip()
    ]
    op_paths = [
        str(operation.get("path") or "").strip().lstrip("./")
        for operation in ops
        if isinstance(operation, dict)
        and str(operation.get("op") or "")
        in {"write_file", "append_file", "replace_in_file"}
        and str(operation.get("path") or "").strip()
    ]
    material_paths = list(dict.fromkeys(op_paths or expected_files))
    original_verification = str(original.get("verification") or "").strip()
    verifier = (
        _python_exists_verification_command(material_paths)
        if material_paths
        else original_verification
    )
    if not (ops or commands) or not verifier:
        return None

    implementation_step: dict[str, Any] = {
        "step_number": 2,
        "description": str(original.get("description") or "Apply requested change"),
        "commands": commands,
        "verification": verifier,
        "rollback": original.get("rollback"),
        "expected_files": material_paths,
    }
    if ops:
        implementation_step["ops"] = ops

    return [
        {
            "step_number": 1,
            "description": "Inspect the current workspace",
            "commands": ["rg --files . | sort"],
            "verification": 'python -c "import sys; sys.exit(0)"',
            "rollback": None,
            "expected_files": [],
        },
        implementation_step,
        {
            "step_number": 3,
            "description": "Verify the requested change",
            "commands": [verifier],
            "verification": verifier,
            "rollback": None,
            "expected_files": [],
        },
    ]

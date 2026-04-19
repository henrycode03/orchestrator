"""Step execution and repair support helpers for orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from app.services.error_handler import error_handler


def step_needs_command_repair(step: Dict[str, Any]) -> bool:
    commands = step.get("commands", [])
    if not isinstance(commands, list):
        return True
    return not any(str(command or "").strip() for command in commands)


def build_step_repair_prompt(
    task_prompt: str,
    step: Dict[str, Any],
    step_index: int,
    project_dir: Path,
    prior_results_summary: str,
    project_context: str,
) -> str:
    return f"""Repair this execution step so it becomes machine-runnable JSON. Return JSON object only.

Task:
{task_prompt[:2000]}

Current step index:
{step_index + 1}

Current step JSON:
{json.dumps(step, indent=2)[:4000]}

Project context:
{project_context[:3000]}

Prior completed results:
{prior_results_summary[:2000]}

Rules:
1. Working directory is {project_dir}
2. Use relative paths only
3. Do not use .., ~, or absolute paths
4. commands must be a non-empty JSON array of shell commands
5. verification and rollback may be null
6. expected_files must be a JSON array
7. Keep the step intent the same
8. Output JSON object only, no prose

Example:
{{
  "step_number": 1,
  "description": "Inspect project structure and locate implementation entry points",
  "commands": ["rg --files . | head -100"],
  "verification": "test -d . && echo ok",
  "rollback": null,
  "expected_files": []
}}
"""


def repair_step_commands_with_self_correction(
    *,
    openclaw_service: Any,
    db: Session,
    session_id: int,
    task_id: int,
    session_instance_id: Optional[str],
    task_prompt: str,
    step: Dict[str, Any],
    step_index: int,
    project_dir: Path,
    prior_results_summary: str,
    project_context: str,
    logger_obj: logging.Logger,
    extract_structured_text: Callable[[Any], str],
    normalize_step: Callable[
        [Dict[str, Any], Path, logging.Logger, int], Dict[str, Any]
    ],
    record_live_log: Callable[..., None],
) -> Optional[Dict[str, Any]]:
    repair_prompt = build_step_repair_prompt(
        task_prompt=task_prompt,
        step=step,
        step_index=step_index,
        project_dir=project_dir,
        prior_results_summary=prior_results_summary,
        project_context=project_context,
    )
    repair_result = asyncio.run(
        openclaw_service.execute_task(repair_prompt, timeout_seconds=120)
    )
    repair_output = extract_structured_text(repair_result.get("output", "{}"))
    success, repair_data, strategy_info = error_handler.attempt_json_parsing(
        repair_output, context="step_repair"
    )
    if not success or not isinstance(repair_data, dict):
        logger_obj.warning(
            "[ORCHESTRATION] Step %s self-correction failed to parse: %s",
            step_index + 1,
            strategy_info,
        )
        record_live_log(
            db,
            session_id,
            task_id,
            "WARN",
            f"[ORCHESTRATION] Step {step_index + 1} self-correction failed: {strategy_info}",
            session_instance_id=session_instance_id,
            metadata={"phase": "step_validation", "strategy": strategy_info},
        )
        return None

    repaired_step = normalize_step(repair_data, project_dir, logger_obj, step_index + 1)
    if step_needs_command_repair(repaired_step):
        record_live_log(
            db,
            session_id,
            task_id,
            "WARN",
            f"[ORCHESTRATION] Step {step_index + 1} self-correction returned no runnable commands",
            session_instance_id=session_instance_id,
            metadata={"phase": "step_validation"},
        )
        return None

    record_live_log(
        db,
        session_id,
        task_id,
        "INFO",
        f"[ORCHESTRATION] Step {step_index + 1} repaired by self-correction",
        session_instance_id=session_instance_id,
        metadata={"phase": "step_validation", "strategy": strategy_info},
    )
    return repaired_step


def coerce_execution_step_result(
    raw_result: Dict[str, Any],
    *,
    expected_files: Optional[list[str]] = None,
    extract_structured_text: Callable[[Any], str],
) -> Dict[str, Any]:
    """Recover a structured step result when the model returned prose instead of JSON."""
    result = dict(raw_result or {})
    output_text = extract_structured_text(result.get("output", ""))

    if isinstance(result.get("output"), dict):
        return result

    success, parsed_data, _strategy_info = error_handler.attempt_json_parsing(
        output_text, context="execution"
    )
    if success and isinstance(parsed_data, dict):
        merged = dict(result)
        merged.update(parsed_data)
        return merged

    normalized = (output_text or "").strip()
    lowered = normalized.lower()
    if not normalized:
        return result

    success_markers = (
        "status:** success",
        "status: success",
        "step complete",
        "verification results:",
        "files changed:",
        "dependencies installed:",
    )
    failure_markers = (
        "status:** failed",
        "status: failed",
        "error:",
        "failed:",
    )

    coerced = dict(result)
    if any(marker in lowered for marker in success_markers):
        coerced["status"] = "success"
        coerced["output"] = normalized
        coerced.setdefault("verification_output", normalized[:1000])
        coerced.setdefault("files_changed", list(expected_files or []))
        coerced.setdefault("error", "")
        return coerced

    if any(marker in lowered for marker in failure_markers):
        coerced["status"] = "failed"
        coerced["output"] = normalized
        coerced.setdefault("verification_output", normalized[:1000])
        coerced.setdefault("error", normalized[:1000])
        return coerced

    return result

"""Planner-stage helpers for orchestration."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


class PlannerService:
    """Planning-stage fallback and repair helpers."""

    @staticmethod
    def should_retry_with_minimal_prompt(
        planning_result: Dict[str, Any], output_text: str = ""
    ) -> bool:
        error_text = (planning_result.get("error") or "").lower()
        combined_text = f"{error_text}\n{(output_text or '').lower()}"
        retry_markers = (
            "context window exceeded",
            "request timed out before a response was generated",
            "timed out",
            "timeout",
        )
        return any(marker in combined_text for marker in retry_markers)

    @staticmethod
    def should_start_with_minimal_prompt(
        task_prompt: str,
        project_context: str,
    ) -> bool:
        combined = f"{task_prompt or ''}\n{project_context or ''}"
        lowered_context = (project_context or "").lower()
        dense_context_markers = (
            "hydrated baseline sources available directly in this workspace",
            "canonical baseline available",
            "earlier ordered tasks already completed and can be reused",
            "promoted workspaces already accepted into the project baseline",
        )
        return (
            len(combined) > 12000
            or len(project_context or "") > 6000
            or any(marker in lowered_context for marker in dense_context_markers)
        )

    @staticmethod
    def build_minimal_planning_prompt(task_description: str, project_dir: Path) -> str:
        concise_task = " ".join((task_description or "").split())[:2000]
        return f"""Produce a JSON-only execution plan for this software task. Do not implement anything.

Task:
{concise_task}

Rules:
1. Assume working directory is {project_dir}
2. Use relative paths only
3. Do not use absolute paths, .., or ~
4. Return 3 to 6 small sequential steps
5. Each step must include: step_number, description, commands, verification, rollback, expected_files
6. expected_files must be relative paths or []
7. Do not use `cat > file <<EOF`, heredocs, or multi-line inline file creation in planning output
8. Do not join separate shell commands with commas
9. Prefer mkdir/touch/package-manager/editor-friendly commands and one-file-at-a-time edits
10. Output JSON array only
"""

    @staticmethod
    def build_planning_repair_prompt(
        task_description: str,
        malformed_output: str,
        project_dir: Path,
        rejection_reasons: Optional[List[str]] = None,
    ) -> str:
        concise_task = " ".join((task_description or "").split())[:2000]
        broken_output = (malformed_output or "")[:8000]
        structured_feedback = ""
        if rejection_reasons:
            reason_lines = "\n".join(
                f"- {reason[:300]}" for reason in rejection_reasons[:8]
            )
            structured_feedback = (
                "\nPrevious validator rejection reasons:\n"
                f"{reason_lines}\n"
                "You must address every rejection reason in the repaired plan.\n"
            )
        return f"""Repair this malformed planning output into valid machine-runnable JSON. Return JSON array only.

Task:
{concise_task}

Working directory:
{project_dir}

Malformed planning output:
{broken_output}
{structured_feedback}

Rules:
1. Return a JSON array only
2. Keep 3 to 8 sequential steps
3. Each step must include: step_number, description, commands, verification, rollback, expected_files
4. Use relative paths only in shell commands and expected_files
5. Do not use absolute paths, .., or ~
6. Do not use heredocs, `cat > file <<EOF`, or multi-line inline file dumps in the repaired plan
7. Do not join separate shell commands with commas
8. Prefer short setup/edit commands over dumping full source files in planning output
9. If the malformed output contains oversized inline file content, replace it with smaller setup/edit commands that preserve the same step intent
10. expected_files must be a JSON array
"""

    @classmethod
    def retry_with_minimal_prompt(
        cls,
        openclaw_service: Any,
        task_description: str,
        project_dir: Path,
        timeout_seconds: int,
        logger: logging.Logger,
        emit_live: Any,
        reason: str,
        rejection_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        logger.warning(
            "[ORCHESTRATION] Planning output was not machine-parseable; "
            f"retrying with minimal prompt ({reason})"
        )
        emit_live(
            "WARN",
            "[ORCHESTRATION] Planning output needed a strict JSON retry",
            metadata={
                "phase": "planning",
                "retry": "minimal_prompt",
                "reason": reason[:240],
            },
        )
        return asyncio.run(
            openclaw_service.execute_task(
                cls.build_minimal_planning_prompt(task_description, project_dir),
                timeout_seconds=min(timeout_seconds, 180),
            )
        )

    @classmethod
    def repair_output(
        cls,
        openclaw_service: Any,
        task_description: str,
        malformed_output: str,
        project_dir: Path,
        timeout_seconds: int,
        logger: logging.Logger,
        emit_live: Any,
        reason: str,
        rejection_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        logger.warning(
            "[ORCHESTRATION] Planning output was malformed but salvageable; "
            f"attempting repair ({reason})"
        )
        emit_live(
            "WARN",
            "[ORCHESTRATION] Planning output was malformed; attempting one repair pass",
            metadata={
                "phase": "planning",
                "retry": "repair_prompt",
                "reason": reason[:240],
            },
        )
        return asyncio.run(
            openclaw_service.execute_task(
                cls.build_planning_repair_prompt(
                    task_description,
                    malformed_output,
                    project_dir,
                    rejection_reasons=rejection_reasons,
                ),
                timeout_seconds=min(timeout_seconds, 120),
            )
        )

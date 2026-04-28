"""Planner-stage helpers for orchestration."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.orchestration.policy import (
    MINIMAL_PLANNING_TIMEOUT_SECONDS,
    PLANNING_REPAIR_TIMEOUT_SECONDS,
    ULTRA_MINIMAL_PLANNING_TIMEOUT_SECONDS,
)
from app.services.workspace.path_display import render_workspace_path_for_prompt


class PlannerService:
    """Planning-stage fallback and repair helpers."""

    @staticmethod
    def select_prompt_profile(
        backend_name: Optional[str],
        model_family: Optional[str],
    ) -> str:
        backend = (backend_name or "").strip().lower()
        model = (model_family or "").strip().lower()
        if backend == "local_openclaw" and ("qwen" in model or model == "local"):
            return "local_qwen_json_array"
        return "default"

    @staticmethod
    def apply_prompt_profile(prompt: str, prompt_profile: str = "default") -> str:
        if prompt_profile != "local_qwen_json_array":
            return prompt

        return (
            f"{prompt.rstrip()}\n\n"
            "Output discipline for this model:\n"
            "11. Return only a JSON array of steps. Do not wrap it in an object.\n"
            "12. Do not include `payloads`, `text`, `finalAssistantVisibleText`, markdown prose, or commentary.\n"
            "13. The first non-whitespace character must be `[` and the last must be `]`.\n"
            "14. Do not describe the file contents outside the JSON fields for each step.\n"
        )

    @staticmethod
    def looks_salvageable_planning_output(output_text: str) -> bool:
        """Heuristic for whether a failed planning response still contains useful plan content."""

        text = (output_text or "").strip()
        if not text:
            return False
        lowered = text.lower()
        planning_markers = (
            '"step_number"',
            '"commands"',
            '"expected_files"',
            '"description"',
            "finalassistantvisibletext",
            "```json",
            "[",
            "{",
        )
        return any(marker in lowered for marker in planning_markers)

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
        lowered_task = (task_prompt or "").lower()
        implementation_markers = (
            "set up",
            "setup",
            "build",
            "create",
            "implement",
            "frontend",
            "backend",
            "fastapi",
            "node.js",
            "react",
            "vite",
            "clean architecture",
        )
        dense_context_markers = (
            "hydrated baseline sources available directly in this workspace",
            "canonical baseline available",
            "earlier ordered tasks already completed and can be reused",
            "promoted workspaces already accepted into the project baseline",
        )
        compact_task_markers = (
            "regression test",
            "test suite",
            "integration test",
            "spec file",
            "unit test",
            "inspection",
            "analyze",
            "review",
        )
        task_looks_implementation_heavy = any(
            marker in lowered_task for marker in implementation_markers
        )
        return (
            len(combined) > 8000
            or len(project_context or "") > 3500
            or any(marker in lowered_context for marker in dense_context_markers)
            or (
                any(marker in lowered_task for marker in compact_task_markers)
                and not task_looks_implementation_heavy
            )
        )

    @staticmethod
    def build_minimal_planning_prompt(
        task_description: str,
        project_dir: Path,
        prompt_profile: str = "default",
    ) -> str:
        concise_task = " ".join((task_description or "").split())[:1200]
        display_project_dir = render_workspace_path_for_prompt(project_dir)
        prompt = f"""Produce a JSON-only execution plan for this software task. Do not implement anything.

Task:
{concise_task}

Rules:
1. Assume working directory is {display_project_dir}
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
        return PlannerService.apply_prompt_profile(prompt, prompt_profile)

    @staticmethod
    def build_ultra_minimal_planning_prompt(
        task_description: str,
        project_dir: Path,
        prompt_profile: str = "default",
    ) -> str:
        concise_task = " ".join((task_description or "").split())[:700]
        display_project_dir = render_workspace_path_for_prompt(project_dir)
        prompt = f"""Return JSON array only. No prose.

Task:
{concise_task}

Working directory: {display_project_dir}

Requirements:
1. 2 to 5 steps only
2. Use short relative shell commands only
3. No heredocs, no long inline source dumps, no absolute paths, no .., no ~
4. Each step must contain exactly these keys:
   step_number, description, commands, verification, rollback, expected_files
5. Keep each command short and machine-runnable
"""
        return PlannerService.apply_prompt_profile(prompt, prompt_profile)

    @staticmethod
    def _looks_like_timeout_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "timed out" in message or "timeout" in message

    @staticmethod
    def build_planning_repair_prompt(
        task_description: str,
        malformed_output: str,
        project_dir: Path,
        rejection_reasons: Optional[List[str]] = None,
        prompt_profile: str = "default",
    ) -> str:
        concise_task = " ".join((task_description or "").split())[:2000]
        broken_output = (malformed_output or "")[:8000]
        display_project_dir = render_workspace_path_for_prompt(project_dir)
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
        prompt = f"""Repair this malformed planning output into valid machine-runnable JSON. Return JSON array only.

Task:
{concise_task}

Working directory:
{display_project_dir}

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
11. Never repeat workspace root segments inside a path, such as `frontend/src/frontend/src` or `backend/src/backend/src`
12. Paths must be rooted exactly once from the canonical project workspace
"""
        return PlannerService.apply_prompt_profile(prompt, prompt_profile)

    @classmethod
    def retry_with_minimal_prompt(
        cls,
        runtime_service: Any,
        task_description: str,
        project_dir: Path,
        timeout_seconds: int,
        logger: logging.Logger,
        emit_live: Any,
        reason: str,
        rejection_reasons: Optional[List[str]] = None,
        prompt_profile: str = "default",
    ) -> Dict[str, Any]:
        logger.warning(
            "[ORCHESTRATION] Planning output was not machine-parseable; "
            f"retrying with minimal prompt ({reason})"
        )
        minimal_timeout = min(timeout_seconds, MINIMAL_PLANNING_TIMEOUT_SECONDS)
        emit_live(
            "WARN",
            (
                "[ORCHESTRATION] Planning output needed a strict JSON retry; "
                f"starting minimal prompt attempt (timeout: {minimal_timeout}s)"
            ),
            metadata={
                "phase": "planning",
                "retry": "minimal_prompt",
                "reason": reason[:240],
                "timeout_seconds": minimal_timeout,
            },
        )
        emit_live(
            "INFO",
            (
                "[ORCHESTRATION] Planning attempt 2 is now running with the minimal "
                f"prompt (timeout: {minimal_timeout}s)"
            ),
            metadata={
                "phase": "planning",
                "attempt": 2,
                "strategy": "minimal_prompt",
                "timeout_seconds": minimal_timeout,
            },
        )
        try:
            return asyncio.run(
                runtime_service.execute_task(
                    cls.build_minimal_planning_prompt(
                        task_description,
                        project_dir,
                        prompt_profile=prompt_profile,
                    ),
                    timeout_seconds=minimal_timeout,
                )
            )
        except Exception as exc:
            if not cls._looks_like_timeout_error(exc):
                raise
            ultra_minimal_timeout = min(
                timeout_seconds, ULTRA_MINIMAL_PLANNING_TIMEOUT_SECONDS
            )
            logger.warning(
                "[ORCHESTRATION] Minimal planning prompt timed out; retrying with ultra-minimal prompt"
            )
            emit_live(
                "WARN",
                (
                    "[ORCHESTRATION] Minimal planning timed out; retrying with "
                    f"ultra-minimal prompt (timeout: {ultra_minimal_timeout}s)"
                ),
                metadata={
                    "phase": "planning",
                    "retry": "ultra_minimal_prompt",
                    "reason": str(exc)[:240],
                    "timeout_seconds": ultra_minimal_timeout,
                },
            )
            emit_live(
                "INFO",
                (
                    "[ORCHESTRATION] Planning attempt 3 is now running with the "
                    f"ultra-minimal prompt (timeout: {ultra_minimal_timeout}s)"
                ),
                metadata={
                    "phase": "planning",
                    "attempt": 3,
                    "strategy": "ultra_minimal_prompt",
                    "timeout_seconds": ultra_minimal_timeout,
                },
            )
            return asyncio.run(
                runtime_service.execute_task(
                    cls.build_ultra_minimal_planning_prompt(
                        task_description,
                        project_dir,
                        prompt_profile=prompt_profile,
                    ),
                    timeout_seconds=ultra_minimal_timeout,
                )
            )

    @classmethod
    def repair_output(
        cls,
        runtime_service: Any,
        task_description: str,
        malformed_output: str,
        project_dir: Path,
        timeout_seconds: int,
        logger: logging.Logger,
        emit_live: Any,
        reason: str,
        rejection_reasons: Optional[List[str]] = None,
        prompt_profile: str = "default",
    ) -> Dict[str, Any]:
        logger.warning(
            "[ORCHESTRATION] Planning output was malformed but salvageable; "
            f"attempting repair ({reason})"
        )
        repair_timeout = min(timeout_seconds, PLANNING_REPAIR_TIMEOUT_SECONDS)
        emit_live(
            "WARN",
            (
                "[ORCHESTRATION] Planning output was malformed; attempting one "
                f"repair pass (timeout: {repair_timeout}s)"
            ),
            metadata={
                "phase": "planning",
                "retry": "repair_prompt",
                "reason": reason[:240],
                "timeout_seconds": repair_timeout,
            },
        )
        emit_live(
            "INFO",
            (
                "[ORCHESTRATION] Planning repair attempt is now running "
                f"(timeout: {repair_timeout}s)"
            ),
            metadata={
                "phase": "planning",
                "attempt": "repair",
                "strategy": "repair_prompt",
                "timeout_seconds": repair_timeout,
            },
        )
        try:
            return asyncio.run(
                runtime_service.execute_task(
                    cls.build_planning_repair_prompt(
                        task_description,
                        malformed_output,
                        project_dir,
                        rejection_reasons=rejection_reasons,
                        prompt_profile=prompt_profile,
                    ),
                    timeout_seconds=repair_timeout,
                )
            )
        except Exception as exc:
            if not cls._looks_like_timeout_error(exc):
                raise
            ultra_minimal_timeout = min(
                timeout_seconds, ULTRA_MINIMAL_PLANNING_TIMEOUT_SECONDS
            )
            logger.warning(
                "[ORCHESTRATION] Planning repair prompt timed out; retrying with ultra-minimal prompt"
            )
            emit_live(
                "WARN",
                (
                    "[ORCHESTRATION] Planning repair timed out; retrying with "
                    f"ultra-minimal prompt (timeout: {ultra_minimal_timeout}s)"
                ),
                metadata={
                    "phase": "planning",
                    "retry": "ultra_minimal_prompt",
                    "reason": str(exc)[:240],
                    "timeout_seconds": ultra_minimal_timeout,
                },
            )
            emit_live(
                "INFO",
                (
                    "[ORCHESTRATION] Planning attempt after repair is now running "
                    f"with the ultra-minimal prompt (timeout: {ultra_minimal_timeout}s)"
                ),
                metadata={
                    "phase": "planning",
                    "attempt": "repair_fallback",
                    "strategy": "ultra_minimal_prompt",
                    "timeout_seconds": ultra_minimal_timeout,
                },
            )
            return asyncio.run(
                runtime_service.execute_task(
                    cls.build_ultra_minimal_planning_prompt(
                        task_description,
                        project_dir,
                        prompt_profile=prompt_profile,
                    ),
                    timeout_seconds=ultra_minimal_timeout,
                )
            )

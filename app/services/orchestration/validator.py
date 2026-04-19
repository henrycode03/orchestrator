"""Rule-first orchestration validation helpers."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from app.models import TaskCheckpoint
from .types import ValidationVerdict


SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx"}
DOC_NAMES = {"readme.md", "notes.md", "summary.md"}


class ValidatorService:
    """Deterministic plan and completion validation."""

    @staticmethod
    def infer_validation_profile(
        task_prompt: str,
        execution_profile: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        combined = " ".join(
            [task_prompt or "", title or "", description or "", execution_profile or ""]
        ).lower()

        if execution_profile in {"review_only", "test_only"} or any(
            marker in combined
            for marker in ("verify", "verification", "review", "audit", "refine", "qa")
        ):
            return "verification"
        if any(
            marker in combined
            for marker in (
                "inspect",
                "analysis",
                "analyze",
                "architecture",
                "inventory",
                "current project structure",
                "current project architecture",
            )
        ):
            return "verification"
        if any(marker in combined for marker in ("integration", "end-to-end", "e2e")):
            return "integration"
        if any(
            marker in combined
            for marker in ("scaffold", "skeleton", "boilerplate", "initialize only")
        ):
            return "scaffold"
        return "implementation"

    @staticmethod
    def _verification_is_weak(command: Optional[str]) -> bool:
        text = str(command or "").strip().lower()
        if not text:
            return True
        meaningful_markers = (
            "pytest",
            "python -m",
            "node -e",
            "node ",
            "npm test",
            "pnpm test",
            "cargo test",
            "go test",
            "python ",
            "uv run",
        )
        if any(marker in text for marker in meaningful_markers):
            return False
        weak_markers = ("test -f", "test -d", "grep -q", "ls ", "echo ", "cat ")
        return any(marker in text for marker in weak_markers)

    @staticmethod
    def _plan_contains_stack_conflict(
        plan: List[Dict[str, Any]], task_prompt: str
    ) -> bool:
        lowered_task = (task_prompt or "").lower()
        if any(
            marker in lowered_task
            for marker in ("python", "node", "javascript", "typescript")
        ):
            return False

        seen_python = False
        seen_node = False
        for step in plan:
            text = " ".join(
                [step.get("description", "")]
                + [str(command or "") for command in step.get("commands", []) or []]
            ).lower()
            if any(
                token in text
                for token in ("requirements.txt", "python ", ".py", "pip ", "pytest")
            ):
                seen_python = True
            if any(
                token in text
                for token in ("package.json", "npm ", "pnpm ", "node ", ".js", ".ts")
            ):
                seen_node = True
        return seen_python and seen_node

    @staticmethod
    def _plan_contains_placeholder_intent(plan: List[Dict[str, Any]]) -> bool:
        placeholder_markers = (
            "pass",
            "todo",
            "notimplemented",
            "placeholder",
            "stub",
        )
        for step in plan:
            for command in step.get("commands", []) or []:
                lowered = str(command or "").lower()
                if any(marker in lowered for marker in placeholder_markers):
                    return True
        return False

    @staticmethod
    def _task_allows_multiple_stacks(
        task_prompt: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        combined = " ".join([task_prompt or "", title or "", description or ""]).lower()
        explicit_pairs = (
            ("python", "javascript"),
            ("python", "node"),
            ("python", "typescript"),
            ("django", "react"),
            ("flask", "react"),
            ("fastapi", "react"),
            ("backend", "frontend"),
        )
        if any(
            left in combined and right in combined for left, right in explicit_pairs
        ):
            return True
        return any(
            marker in combined
            for marker in ("polyglot", "multi-language", "full stack", "full-stack")
        )

    @staticmethod
    def _infer_stack_from_plan(plan: List[Dict[str, Any]]) -> Optional[str]:
        seen_python = False
        seen_node = False
        for step in plan:
            text = " ".join(
                [
                    str(step.get("description") or ""),
                    str(step.get("verification") or ""),
                ]
                + [str(command or "") for command in step.get("commands", []) or []]
                + [str(path or "") for path in step.get("expected_files", []) or []]
            ).lower()
            if any(
                token in text
                for token in (
                    "requirements.txt",
                    "python ",
                    ".py",
                    "pip ",
                    "pytest",
                    "pyproject.toml",
                )
            ):
                seen_python = True
            if any(
                token in text
                for token in (
                    "package.json",
                    "npm ",
                    "pnpm ",
                    "node ",
                    ".js",
                    ".ts",
                    "tsconfig.json",
                )
            ):
                seen_node = True
        if seen_python and seen_node:
            return "mixed"
        if seen_node:
            return "node"
        if seen_python:
            return "python"
        return None

    @staticmethod
    def _plan_contains_brittle_commands(
        extracted_plan: Optional[List[Dict[str, Any]]], output_text: str = ""
    ) -> bool:
        if not extracted_plan:
            return False

        heredoc_count = 0
        for step in extracted_plan:
            commands = step.get("commands", [])
            if not isinstance(commands, list):
                return True
            for command in commands:
                raw_command = str(command or "")
                lowered = raw_command.lower()
                if "cat >" in lowered and "<< 'eof'" in lowered:
                    heredoc_count += 1
                if "cat >" in lowered and "<< eof" in lowered:
                    heredoc_count += 1
                if re.search(r"mkdir\s+-p\s+[^|;&\n]+,cat\s+>", lowered):
                    return True
                if raw_command.count("\n") > 25:
                    return True
                if len(raw_command) > 1200:
                    return True

        if heredoc_count >= 2:
            return True

        lowered_output = (output_text or "").lower()
        if lowered_output.count("cat >") >= 2 and "```json" in lowered_output:
            return True

        return False

    @staticmethod
    def _is_non_runnable_command(command: str) -> bool:
        text = str(command or "").strip()
        lowered = text.lower()
        if not text:
            return True
        if lowered.startswith("edit "):
            return True
        if lowered.startswith("verify "):
            return True
        if lowered.startswith("check ") and "test " not in lowered:
            return True
        if lowered.startswith("ensure "):
            return True
        if lowered.startswith("confirm "):
            return True
        return False

    @classmethod
    def _plan_contains_non_runnable_commands(
        cls, plan: List[Dict[str, Any]]
    ) -> List[int]:
        bad_steps: List[int] = []
        for step in plan:
            for command in step.get("commands", []) or []:
                if cls._is_non_runnable_command(str(command or "")):
                    bad_steps.append(step.get("step_number"))
                    break
        return [step for step in bad_steps if step is not None]

    @staticmethod
    def _plan_nests_task_workspace(
        plan: List[Dict[str, Any]], project_dir: Optional[Path]
    ) -> List[int]:
        if not project_dir:
            return []
        nested_prefix = f"{project_dir.name}/"
        bad_steps: List[int] = []
        for step in plan:
            step_text_parts = [
                str(step.get("verification") or ""),
                str(step.get("rollback") or ""),
            ]
            step_text_parts.extend(
                str(command or "") for command in step.get("commands", []) or []
            )
            step_text_parts.extend(
                str(path or "") for path in step.get("expected_files", []) or []
            )
            combined = "\n".join(step_text_parts)
            if nested_prefix in combined:
                bad_steps.append(step.get("step_number"))
        return [step for step in bad_steps if step is not None]

    @classmethod
    def validate_plan(
        cls,
        plan: List[Dict[str, Any]],
        *,
        output_text: str,
        task_prompt: str,
        execution_profile: str,
        project_dir: Optional[Path] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ValidationVerdict:
        profile = cls.infer_validation_profile(
            task_prompt, execution_profile, title=title, description=description
        )
        reasons: List[str] = []
        details: Dict[str, Any] = {"plan_length": len(plan)}

        if cls._plan_contains_brittle_commands(plan, output_text):
            reasons.append("Plan contains brittle heredoc-heavy or malformed commands")

        non_runnable_steps = cls._plan_contains_non_runnable_commands(plan)
        if non_runnable_steps:
            reasons.append(
                "Plan contains non-runnable pseudo-commands such as `edit` or prose instructions "
                f"(steps: {non_runnable_steps[:5]})"
            )
            details["non_runnable_steps"] = non_runnable_steps

        nested_workspace_steps = cls._plan_nests_task_workspace(plan, project_dir)
        if nested_workspace_steps:
            reasons.append(
                "Plan incorrectly recreates the current task workspace as a nested folder "
                f"(steps: {nested_workspace_steps[:5]})"
            )
            details["nested_workspace_steps"] = nested_workspace_steps

        if profile == "implementation":
            weak_verification_steps = [
                step.get("step_number")
                for step in plan
                if cls._verification_is_weak(step.get("verification"))
            ]
            if weak_verification_steps:
                reasons.append(
                    "Plan uses weak verification for implementation-heavy work "
                    f"(steps: {weak_verification_steps[:5]})"
                )
                details["weak_verification_steps"] = weak_verification_steps

            if cls._plan_contains_placeholder_intent(plan):
                reasons.append(
                    "Plan appears to generate placeholder or stub implementations"
                )

        if cls._plan_contains_stack_conflict(plan, task_prompt):
            reasons.append("Plan mixes inconsistent implementation stacks for one task")
            details["stack_conflict"] = True

        if not reasons:
            return ValidationVerdict(
                stage="plan",
                status="accepted",
                profile=profile,
                reasons=[],
                details=details,
            )

        status = "repair_required"
        if any("placeholder" in reason.lower() for reason in reasons):
            status = "rejected"
        return ValidationVerdict(
            stage="plan",
            status=status,
            profile=profile,
            reasons=reasons,
            details=details,
        )

    @staticmethod
    def _iter_candidate_files(
        project_dir: Path, file_paths: Iterable[str]
    ) -> List[Path]:
        candidates: List[Path] = []
        for raw_path in file_paths:
            relative = str(raw_path or "").strip().rstrip("/")
            if not relative:
                continue
            candidate = (project_dir / relative).resolve()
            if candidate.exists() and candidate.is_file():
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _detect_placeholder_content(path: Path) -> List[str]:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return []

        reasons: List[str] = []
        lowered = content.lower()
        if re.search(r"^\s*pass\s*$", content, flags=re.MULTILINE):
            reasons.append(f"{path.name} still contains `pass` placeholders")
        if "todo" in lowered or "placeholder" in lowered:
            reasons.append(f"{path.name} still contains TODO or placeholder markers")
        if "notimplemented" in lowered or "raise notimplementederror" in lowered:
            reasons.append(f"{path.name} still contains not-implemented markers")
        if "__main__" in content and "if __name__ == __main__" in content:
            reasons.append(f"{path.name} has a broken Python __main__ entrypoint check")
        if path.suffix == ".py":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                reasons.append(f"{path.name} has Python syntax errors: {exc.msg}")
        return reasons

    @staticmethod
    def _core_expected_files(plan: List[Dict[str, Any]]) -> List[str]:
        files: List[str] = []
        seen = set()
        for step in plan:
            for raw_path in step.get("expected_files", []) or []:
                path_text = str(raw_path or "").strip()
                if (
                    not path_text
                    or path_text.endswith("/")
                    or path_text.lower() in DOC_NAMES
                ):
                    continue
                if Path(path_text).suffix.lower() not in SOURCE_EXTENSIONS:
                    continue
                if path_text not in seen:
                    seen.add(path_text)
                    files.append(path_text)
        return files

    @classmethod
    def validate_step_success(
        cls,
        *,
        project_dir: Path,
        step: Dict[str, Any],
        step_output: str,
        missing_expected_files: List[str],
        tool_failures: List[str],
        validation_profile: str,
    ) -> ValidationVerdict:
        reasons: List[str] = []
        details: Dict[str, Any] = {}

        if missing_expected_files:
            reasons.append(
                f"Expected files are missing: {', '.join(missing_expected_files[:6])}"
            )
            details["missing_expected_files"] = missing_expected_files[:20]

        if tool_failures:
            reasons.append(
                "Task logs contain tool failures during the successful step window"
            )
            details["tool_failures"] = tool_failures[:10]

        if validation_profile == "implementation" and cls._verification_is_weak(
            step.get("verification")
        ):
            reasons.append(
                "Step verification is too weak for implementation-heavy work"
            )

        candidate_files = cls._iter_candidate_files(
            project_dir,
            step.get("expected_files", []) or [],
        )
        placeholder_reasons: List[str] = []
        for candidate in candidate_files:
            placeholder_reasons.extend(cls._detect_placeholder_content(candidate))
        if placeholder_reasons and validation_profile == "implementation":
            reasons.extend(placeholder_reasons[:6])
            details["placeholder_reasons"] = placeholder_reasons[:20]

        status = "accepted" if not reasons else "rejected"
        return ValidationVerdict(
            stage="step_completion",
            status=status,
            profile=validation_profile,
            reasons=reasons,
            details=details | {"step_output_preview": step_output[:240]},
        )

    @classmethod
    def validate_task_completion(
        cls,
        *,
        project_dir: Path,
        plan: List[Dict[str, Any]],
        task_prompt: str,
        execution_profile: str,
        workspace_consistency: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ValidationVerdict:
        profile = cls.infer_validation_profile(
            task_prompt, execution_profile, title=title, description=description
        )
        expected_core_files = cls._core_expected_files(plan)
        candidate_files = cls._iter_candidate_files(project_dir, expected_core_files)

        missing_core = [
            path_text
            for path_text in expected_core_files
            if not (project_dir / path_text).resolve().exists()
        ]
        reasons: List[str] = []
        details: Dict[str, Any] = {
            "expected_core_files": expected_core_files[:20],
            "validated_files": [
                str(path.relative_to(project_dir)) for path in candidate_files[:20]
            ],
        }

        if missing_core:
            reasons.append(
                f"Core implementation files are missing: {', '.join(missing_core[:6])}"
            )
            details["missing_core_files"] = missing_core[:20]

        placeholder_reasons: List[str] = []
        for candidate in candidate_files:
            placeholder_reasons.extend(cls._detect_placeholder_content(candidate))
        if placeholder_reasons and profile == "implementation":
            reasons.extend(placeholder_reasons[:10])
            details["placeholder_reasons"] = placeholder_reasons[:20]

        if profile == "implementation" and not candidate_files:
            reasons.append("No core implementation source files were produced")

        workspace_consistency = workspace_consistency or {}
        plan_stack = cls._infer_stack_from_plan(plan)
        allows_multiple_stacks = cls._task_allows_multiple_stacks(
            task_prompt, title=title, description=description
        )
        details["workspace_consistency"] = workspace_consistency

        if profile == "implementation":
            if workspace_consistency.get("nested_duplicate_dirs"):
                reasons.append(
                    "Workspace contains nested duplicate implementation directories: "
                    + ", ".join(
                        workspace_consistency.get("nested_duplicate_dirs", [])[:4]
                    )
                )
            if workspace_consistency.get("mixed_stack") and not allows_multiple_stacks:
                if plan_stack in {"node", "python"}:
                    reasons.append(
                        "Workspace mixes Python and Node/JS artifacts even though the accepted plan targets a single "
                        f"{plan_stack} stack"
                    )
                else:
                    reasons.append(
                        "Workspace contains mixed Python and Node/JS implementation artifacts for one task"
                    )

        status = "accepted" if not reasons else "rejected"
        return ValidationVerdict(
            stage="task_completion",
            status=status,
            profile=profile,
            reasons=reasons,
            details=details,
        )

    @staticmethod
    def validate_baseline_publish(
        *,
        validation_profile: str,
        baseline_path: str,
        baseline_file_count: int,
        missing_task_expected_files: List[str],
        missing_prior_expected_files: List[Dict[str, Any]],
        consistency_issues: Optional[List[str]] = None,
        consistency_details: Optional[Dict[str, Any]] = None,
    ) -> ValidationVerdict:
        reasons: List[str] = []
        details: Dict[str, Any] = {
            "baseline_path": baseline_path,
            "baseline_file_count": baseline_file_count,
        }

        if baseline_file_count <= 0:
            reasons.append("Canonical baseline is empty after publish")

        if missing_task_expected_files:
            reasons.append(
                "Published baseline is missing current task files: "
                + ", ".join(missing_task_expected_files[:6])
            )
            details["missing_task_expected_files"] = missing_task_expected_files[:20]

        if missing_prior_expected_files:
            reasons.append(
                "Canonical baseline is missing previously completed task files"
            )
            details["missing_prior_expected_files"] = missing_prior_expected_files[:20]
        if consistency_issues:
            reasons.extend(consistency_issues[:4])
            details["consistency_issues"] = consistency_issues[:10]
        if consistency_details:
            details["consistency"] = consistency_details

        return ValidationVerdict(
            stage="baseline_publish",
            status="accepted" if not reasons else "rejected",
            profile=validation_profile,
            reasons=reasons,
            details=details,
        )

    @staticmethod
    def persist_validation_result(
        db: Session,
        *,
        task_id: int,
        session_id: Optional[int],
        stage: str,
        verdict: ValidationVerdict,
        step_number: Optional[int] = None,
    ) -> None:
        db.add(
            TaskCheckpoint(
                task_id=task_id,
                session_id=session_id,
                checkpoint_type=f"validation_{stage}",
                step_number=step_number,
                description=f"{stage}:{verdict.status}",
                state_snapshot=json.dumps(verdict.to_dict()),
            )
        )

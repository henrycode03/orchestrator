"""Declarative Task-1 bootstrap planning contract."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.orchestration.validation.workspace_checks import SOURCE_EXTENSIONS


TEST_ROOTS = {"test", "tests", "spec", "specs"}
PLACEHOLDER_RE = re.compile(
    r"\b(?:pass|todo|fixme|stub|placeholder|notimplemented|notimplementederror)\b|"
    r"\bnot[-_\s]*implemented\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskBootstrapContract:
    expected_source_files: list[str] = field(default_factory=list)
    expected_test_files: list[str] = field(default_factory=list)
    required_verification: list[str] = field(default_factory=list)
    forbidden_path_drift: list[str] = field(default_factory=list)
    minimum_implementation_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_source_files": list(self.expected_source_files),
            "expected_test_files": list(self.expected_test_files),
            "required_verification": list(self.required_verification),
            "forbidden_path_drift": list(self.forbidden_path_drift),
            "minimum_implementation_evidence": self.minimum_implementation_evidence,
        }


@dataclass(frozen=True)
class TaskBootstrapContractVerdict:
    contract: TaskBootstrapContract
    passed: bool
    violations: list[str] = field(default_factory=list)
    violation_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": list(self.violations),
            "violation_codes": list(self.violation_codes),
            **self.contract.to_dict(),
        }


def _normalize_path(path_text: Any) -> str:
    return str(path_text or "").strip().rstrip("/").lstrip("./")


def _is_test_path(path_text: str) -> bool:
    parts = Path(path_text).parts
    return bool(parts and parts[0].lower() in TEST_ROOTS)


def _is_source_path(path_text: str) -> bool:
    normalized = _normalize_path(path_text)
    if not normalized or _is_test_path(normalized):
        return False
    return Path(normalized).suffix.lower() in SOURCE_EXTENSIONS


def _materialized_file_targets(plan: list[dict[str, Any]]) -> set[str]:
    targets: set[str] = set()
    for step in plan:
        for operation in step.get("ops") or []:
            if not isinstance(operation, dict):
                continue
            if str(operation.get("op") or "") not in {
                "write_file",
                "append_file",
                "replace_in_file",
            }:
                continue
            path = _normalize_path(operation.get("path"))
            if path:
                targets.add(path)
    return targets


def _declared_expected_files(plan: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for step in plan:
        for path_text in step.get("expected_files") or []:
            path = _normalize_path(path_text)
            if path:
                paths.add(path)
    return paths


def _verification_commands(plan: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    for step in plan:
        verification = str(step.get("verification") or "").strip()
        if verification:
            commands.append(verification)
    return list(dict.fromkeys(commands))


def _minimum_implementation_evidence(plan: list[dict[str, Any]]) -> bool:
    for step in plan:
        for operation in step.get("ops") or []:
            if not isinstance(operation, dict):
                continue
            if str(operation.get("op") or "") not in {"write_file", "append_file"}:
                continue
            path = _normalize_path(operation.get("path"))
            if not _is_source_path(path):
                continue
            content = str(operation.get("content") or "").strip()
            if len(content) < 24:
                continue
            if PLACEHOLDER_RE.search(content):
                continue
            return True
    return False


def build_task1_bootstrap_contract(
    *,
    plan: list[dict[str, Any]],
    forbidden_path_drift: list[str] | None = None,
) -> TaskBootstrapContract:
    materialized = _materialized_file_targets(plan)
    declared = _declared_expected_files(plan)
    source_candidates = sorted(
        path for path in (materialized | declared) if _is_source_path(path)
    )
    test_candidates = sorted(
        path for path in (materialized | declared) if _is_test_path(path)
    )
    return TaskBootstrapContract(
        expected_source_files=source_candidates,
        expected_test_files=test_candidates,
        required_verification=_verification_commands(plan),
        forbidden_path_drift=sorted(set(forbidden_path_drift or [])),
        minimum_implementation_evidence=_minimum_implementation_evidence(plan),
    )


def validate_task1_bootstrap_contract(
    *,
    plan: list[dict[str, Any]],
    task_prompt: str = "",
    forbidden_path_drift: list[str] | None = None,
) -> TaskBootstrapContractVerdict:
    contract = build_task1_bootstrap_contract(
        plan=plan,
        forbidden_path_drift=forbidden_path_drift,
    )
    violations: list[str] = []
    codes: list[str] = []

    if not contract.expected_source_files:
        violations.append("Task 1 bootstrap must declare or materialize source files")
        codes.append("task1_bootstrap_missing_expected_source_files")

    prompt_lower = str(task_prompt or "").lower()
    if "test" in prompt_lower and not contract.expected_test_files:
        violations.append(
            "Task 1 bootstrap prompt asks for tests but no test files are declared or materialized"
        )
        codes.append("task1_bootstrap_missing_expected_test_files")

    if not contract.required_verification:
        violations.append("Task 1 bootstrap must include required verification")
        codes.append("task1_bootstrap_missing_required_verification")

    if contract.forbidden_path_drift:
        violations.append("Task 1 bootstrap contains forbidden path drift")
        codes.append("task1_bootstrap_forbidden_path_drift")

    if not contract.minimum_implementation_evidence:
        violations.append("Task 1 bootstrap lacks minimum implementation evidence")
        codes.append("task1_bootstrap_minimum_implementation_evidence_missing")

    return TaskBootstrapContractVerdict(
        contract=contract,
        passed=not violations,
        violations=violations,
        violation_codes=codes,
    )

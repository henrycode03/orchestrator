"""Phase 34-C8 accepted-plan temporal-authority regressions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
)
from app.services.orchestration.execution.executor import ExecutorService
from app.services.orchestration.validation.accepted_path_authority import (
    accepted_path_authority_from_verdict,
)
from app.services.orchestration.validation.path_authority import declare
from app.services.orchestration.validation.validator import ValidatorService


def _two_write_plan(path: str = "tiny_calc.py") -> list[dict[str, Any]]:
    return [
        {
            "step_number": 1,
            "description": f"Create {path} once",
            "commands": ["python3 -m pytest -q test_tiny_calc.py"],
            "verification": "python3 -m pytest -q test_tiny_calc.py",
            "rollback": None,
            "expected_files": [path],
            "ops": [
                {"op": "write_file", "path": path, "content": "VALUE = 1\n"},
                {"op": "write_file", "path": path, "content": "VALUE = 2\n"},
            ],
        }
    ]


def _plan_for_ops(
    ops: list[dict[str, Any]],
    *,
    expected_files: list[str] | None = None,
    commands: list[str] | None = None,
    step_count: int = 1,
) -> list[dict[str, Any]]:
    expected = expected_files or list(dict.fromkeys(str(op.get("path")) for op in ops))
    command_list = commands or ["python3 -m pytest -q test_tiny_calc.py"]
    if step_count == 1:
        return [
            {
                "step_number": 1,
                "description": "Implement tiny_calc.py",
                "commands": command_list,
                "verification": "python3 -m pytest -q test_tiny_calc.py",
                "rollback": None,
                "expected_files": expected,
                "ops": ops,
            }
        ]
    return [
        {
            "step_number": index,
            "description": "Implement tiny_calc.py",
            "commands": command_list,
            "verification": "python3 -m pytest -q test_tiny_calc.py",
            "rollback": None,
            "expected_files": expected,
            "ops": [ops[index - 1]],
        }
        for index in range(1, step_count + 1)
    ]


def _validate(
    tmp_path: Path,
    plan: list[dict[str, Any]],
    *,
    existing: bool = False,
    commands: list[str] | None = None,
):
    test_path = tmp_path / "test_tiny_calc.py"
    if not test_path.exists():
        test_path.write_text("def test_answer():\n    assert True\n", encoding="utf-8")
    if existing and not (tmp_path / "tiny_calc.py").exists():
        (tmp_path / "tiny_calc.py").write_text("VALUE = 1\n", encoding="utf-8")
    paths = list(
        dict.fromkeys(
            str(operation.get("path"))
            for step in plan
            for operation in step.get("ops") or []
            if operation.get("path")
        )
    )
    task_prompt = (
        "Rewrite tiny_calc.py; preserve and verify test_tiny_calc.py."
        if existing
        else "Implement tiny_calc.py; preserve and verify test_tiny_calc.py."
    )
    source_materialization = materialize_planner_source_context(
        tmp_path,
        task_description=task_prompt,
        expected_paths=paths,
        supporting_paths=["test_tiny_calc.py"],
        creation_authorized_paths=[] if existing else paths,
    )
    return ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=task_prompt,
        execution_profile="implementation",
        project_dir=tmp_path,
        title="Phase 34-C8",
        source_materialization=source_materialization,
    )


def test_absent_path_repeated_write_is_rejected_before_apa_creation(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_tiny_calc.py").write_text(
        "def test_answer():\n    assert True\n",
        encoding="utf-8",
    )
    plan = _two_write_plan()
    task_prompt = "Implement tiny_calc.py; preserve and verify test_tiny_calc.py."
    source_materialization = materialize_planner_source_context(
        tmp_path,
        task_description=task_prompt,
        expected_paths=["tiny_calc.py"],
        supporting_paths=["test_tiny_calc.py"],
        creation_authorized_paths=["tiny_calc.py"],
    )

    outcome = ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=task_prompt,
        execution_profile="implementation",
        project_dir=tmp_path,
        title="Phase 34-C8",
        source_materialization=source_materialization,
    )

    assert outcome.status == "repair_required"
    assert "incompatible_same_path_mutation_sequence" in outcome.reasons


def test_absent_path_create_then_replace_is_rejected_before_apa_creation(
    tmp_path: Path,
) -> None:
    plan = _plan_for_ops(
        [
            {"op": "write_file", "path": "tiny_calc.py", "content": "VALUE = 1\n"},
            {
                "op": "replace_in_file",
                "path": "tiny_calc.py",
                "old": "VALUE = 1\n",
                "new": "VALUE = 2\n",
            },
        ]
    )

    outcome = _validate(tmp_path, plan)

    assert outcome.status == "repair_required"
    assert outcome.details["incompatible_same_path_mutation_sequence"][0]["path"] == (
        "tiny_calc.py"
    )
    assert accepted_path_authority_from_verdict(outcome.verdict) is None


def test_existing_path_repeated_mutations_keep_existing_mutable_authority(
    tmp_path: Path,
) -> None:
    plan = _plan_for_ops(
        [
            {"op": "write_file", "path": "tiny_calc.py", "content": "VALUE = 2\n"},
            {"op": "write_file", "path": "tiny_calc.py", "content": "VALUE = 3\n"},
        ]
    )

    outcome = _validate(tmp_path, plan, existing=True)
    authority = accepted_path_authority_from_verdict(outcome.verdict)

    assert outcome.status == "accepted"
    assert authority is not None
    assert authority.grant_for(declare("tiny_calc.py")).grant_class.value == (
        "existing_mutable"
    )
    first = ExecutorService.execute_file_ops(
        tmp_path, [plan[0]["ops"][0]], accepted_path_authority=authority
    )
    second = ExecutorService.execute_file_ops(
        tmp_path, [plan[0]["ops"][1]], accepted_path_authority=authority
    )
    assert first["success"] is True
    assert second["success"] is True
    assert (tmp_path / "tiny_calc.py").read_text(encoding="utf-8") == "VALUE = 3\n"


def test_mixed_shell_write_same_absent_path_is_rejected(tmp_path: Path) -> None:
    plan = _plan_for_ops(
        [{"op": "write_file", "path": "tiny_calc.py", "content": "VALUE = 1\n"}],
        commands=["printf 'VALUE = 2\\n' > tiny_calc.py"],
    )

    outcome = _validate(tmp_path, plan)

    assert outcome.status == "repair_required"
    assert "incompatible_same_path_mutation_sequence" in outcome.reasons


def test_shell_write_then_structured_write_same_absent_path_is_rejected(
    tmp_path: Path,
) -> None:
    plan = [
        {
            "step_number": 1,
            "description": "Create tiny_calc.py by safe shell write",
            "commands": ["printf 'VALUE = 1\\n' > tiny_calc.py"],
            "verification": "python3 -m pytest -q test_tiny_calc.py",
            "rollback": None,
            "expected_files": ["tiny_calc.py"],
            "ops": [],
        },
        {
            "step_number": 2,
            "description": "Rewrite tiny_calc.py",
            "commands": ["python3 -m pytest -q test_tiny_calc.py"],
            "verification": "python3 -m pytest -q test_tiny_calc.py",
            "rollback": None,
            "expected_files": ["tiny_calc.py"],
            "ops": [
                {"op": "write_file", "path": "tiny_calc.py", "content": "VALUE = 2\n"}
            ],
        },
    ]

    outcome = _validate(tmp_path, plan)

    assert outcome.status == "repair_required"
    assert "incompatible_same_path_mutation_sequence" in outcome.reasons


def test_structured_write_and_verification_read_remain_accepted(tmp_path: Path) -> None:
    plan = _plan_for_ops(
        [{"op": "write_file", "path": "tiny_calc.py", "content": "VALUE = 1\n"}],
        commands=["cat tiny_calc.py", "python3 -m pytest -q test_tiny_calc.py"],
    )

    outcome = _validate(tmp_path, plan)

    assert outcome.status == "accepted"
    assert accepted_path_authority_from_verdict(outcome.verdict) is not None


def test_original_order_is_checked_for_reordered_same_path_steps(
    tmp_path: Path,
) -> None:
    operations = [
        {"op": "write_file", "path": "tiny_calc.py", "content": "VALUE = 1\n"},
        {"op": "write_file", "path": "tiny_calc.py", "content": "VALUE = 2\n"},
    ]
    plan = _plan_for_ops(operations, step_count=2)

    outcome = _validate(tmp_path, plan)

    assert outcome.status == "repair_required"
    assert "incompatible_same_path_mutation_sequence" in outcome.reasons

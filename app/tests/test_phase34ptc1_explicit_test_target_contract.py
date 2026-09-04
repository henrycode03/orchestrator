"""Provider-free PTC1 tests for explicit test-target planning guidance."""

import hashlib
import json
from pathlib import Path

from app.services.orchestration.operations.source_region_identity import (
    SourceRegionIdentity,
)
from app.services.orchestration.planning.planner import PlannerService
from app.services.orchestration.planning.prompt_contracts import (
    render_test_scaffold_contract,
)
from app.services.orchestration.planning.source_materialization import (
    current_source_version_identity,
)
from app.services.orchestration.validation.validator import ValidatorService


TASK = (
    "Update the formatter and add regression coverage for repeated spaces and "
    "blank input."
)


def _source_replace(root: Path) -> dict:
    path = root / "src" / "greeting" / "formatting.py"
    source = path.read_bytes()
    selector = SourceRegionIdentity.from_region(
        canonical_path="src/greeting/formatting.py",
        expected_source_version=current_source_version_identity(path),
        start_byte=0,
        end_byte=len(source),
        selected_region_sha256=hashlib.sha256(source).hexdigest(),
    )
    return {
        "op": "replace_in_file",
        "path": "src/greeting/formatting.py",
        "selector": selector.to_dict(),
        "new": source.decode("utf-8").replace(
            'return " ".join(str(value).strip().split())',
            'return " ".join(str(value).strip().split()).title()',
        ),
    }


def _test_target_plan(
    root: Path,
    *,
    inspection: bool = False,
    expected_files: bool = False,
    verification: bool = False,
    mutation: bool = False,
) -> list[dict]:
    test_step = {
        "step_number": 2,
        "description": "Handle regression coverage",
        "commands": [
            "cat tests/test_formatting.py" if inspection else 'python -c "pass"'
        ],
        "verification": (
            "python -m pytest tests/test_formatting.py -q"
            if verification
            else 'python -c "pass"'
        ),
        "rollback": None,
        "expected_files": ["tests/test_formatting.py"] if expected_files else [],
        "ops": (
            [
                {
                    "op": "append_file",
                    "path": "tests/test_formatting.py",
                    "content": "\n\ndef test_regression():\n    assert True\n",
                }
            ]
            if mutation
            else []
        ),
    }
    return [
        {
            "step_number": 1,
            "description": "Update the formatter implementation",
            "commands": [],
            "verification": "python -m pytest tests/test_formatting.py -q",
            "rollback": None,
            "expected_files": ["src/greeting/formatting.py"],
            "ops": [_source_replace(root)],
        },
        test_step,
    ]


def _validate(root: Path, plan: list[dict], task: str = TASK):
    return ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=task,
        execution_profile="implementation",
        project_dir=root,
        is_first_ordered_task=True,
    )


def test_shared_test_contract_defines_executable_target_and_non_authority_sources():
    contract = render_test_scaffold_contract().lower()

    assert "explicit new/regression intent" in contract
    assert "supported test mutation" in contract
    assert "inspection" in contract
    assert "verification" in contract
    assert "expected_files" in contract
    assert "never invent paths" in contract


def test_initial_planning_prompt_carries_explicit_test_target_obligation(tmp_path):
    prompt = PlannerService.build_minimal_planning_prompt(
        TASK,
        Path(tmp_path),
        workspace_has_existing_files=True,
    ).lower()

    assert "supported test mutation" in prompt
    assert "inspection" in prompt
    assert "expected_files" in prompt


def test_repair_prompt_carries_explicit_test_target_obligation(tmp_path):
    prompt = PlannerService.build_planning_repair_prompt(
        TASK,
        '[{"description":"inspect tests","commands":["cat tests/test_formatting.py"],'
        '"verification":"python -m pytest tests/test_formatting.py -v",'
        '"rollback":null,"expected_files":["tests/test_formatting.py"],"ops":[]}]',
        Path(tmp_path),
        rejection_reasons=[
            "Task 1 bootstrap prompt asks for tests but no test files are materialized"
        ],
    ).lower()

    assert "supported test mutation" in prompt
    assert "inspection" in prompt
    assert "expected_files" in prompt


def test_non_mutating_test_representations_remain_invalid(tmp_path):
    from app.tests.test_phase34pca1_plan_contract_admission import _seed_b_workspace

    _seed_b_workspace(tmp_path)
    for kwargs in (
        {"inspection": True},
        {"expected_files": True},
        {"verification": True},
    ):
        outcome = _validate(tmp_path, _test_target_plan(tmp_path, **kwargs))
        assert not outcome.accepted
        assert any(
            "test" in reason.lower() or "bootstrap" in reason.lower()
            for reason in outcome.reasons
        )


def test_explicit_existing_test_mutation_satisfies_target_contract(tmp_path):
    from app.tests.test_phase34pca1_plan_contract_admission import _seed_b_workspace

    _seed_b_workspace(tmp_path)
    outcome = _validate(
        tmp_path,
        _test_target_plan(
            tmp_path,
            inspection=True,
            expected_files=True,
            verification=True,
            mutation=True,
        ),
    )
    assert outcome.accepted, outcome.reasons


def test_tasks_without_explicit_test_intent_do_not_require_test_mutation(tmp_path):
    from app.tests.test_phase34pca1_plan_contract_admission import _seed_b_workspace

    _seed_b_workspace(tmp_path)
    outcome = _validate(
        tmp_path,
        _test_target_plan(tmp_path),
        task="Update the formatter implementation to normalize whitespace.",
    )
    assert outcome.accepted, outcome.reasons


def test_unknown_test_path_is_not_invented_by_prompt_projection(tmp_path):
    prompt = PlannerService.build_minimal_planning_prompt(
        "Add regression coverage for the formatter.",
        Path(tmp_path),
        workspace_has_existing_files=True,
    )

    assert "tests/test_formatting.py" not in prompt
    assert "Never invent paths" in prompt

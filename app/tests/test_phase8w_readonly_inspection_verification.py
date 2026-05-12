from __future__ import annotations

import json

from app.services.orchestration.planning.planner import PlannerService
from app.services.orchestration.validation.validator import ValidatorService


def _inspection_step():
    return {
        "step_number": 1,
        "description": "Inspect current workspace structure",
        "commands": ["ls -la", "cat app_config.py", "cat README.md"],
        "verification": None,
        "rollback": None,
        "expected_files": ["app_config.py", "README.md"],
    }


def test_readonly_inspection_step_does_not_trigger_immediate_weak_repair():
    issues = PlannerService.find_immediate_repair_step_issues([_inspection_step()])

    assert "weak_verification_steps" not in issues


def test_validator_does_not_require_verification_for_readonly_inspection_step(tmp_path):
    plan = [_inspection_step()]

    verdict = ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt="Update the Python config workspace",
        execution_profile="full_lifecycle",
        project_dir=tmp_path,
    )

    assert "missing_verification_steps" not in verdict.details
    assert "weak_verification_steps" not in verdict.details
    assert "missing_verification" not in verdict.details.get(
        "semantic_violation_codes", []
    )
    assert "weak_verification" not in verdict.details.get(
        "semantic_violation_codes", []
    )


def test_mutating_step_still_triggers_weak_verification_repair():
    issues = PlannerService.find_immediate_repair_step_issues(
        [
            {
                "step_number": 1,
                "description": "Update app_config.py",
                "commands": [
                    "python - <<'PY'\nPath('app_config.py').write_text('x')\nPY"
                ],
                "verification": "ls app_config.py",
                "rollback": None,
                "expected_files": ["app_config.py"],
            }
        ]
    )

    assert issues["weak_verification_steps"] == [1]

"""Characterization tests for the verification_mutates_source_assets repair
deadlock fix and the verification-profile planning contract.

Plans and rejection reasons are minimized reproductions of the T6 "Final
verification" failures from tasks 748, 802, and 814 (see
docs/roadmap/reports/maintenance/verification-mutates-source-assets-analysis-20260611.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.orchestration.planning.planner import (
    PlannerService,
    VERIFICATION_PROFILE_PLANNING_CONTRACT_LINE,
)
from app.services.orchestration.planning.repair_prompts import (
    build_compact_planning_repair_prompt,
    build_planning_repair_prompt,
)
from app.services.orchestration.validation.validator import ValidatorService

PRESERVATION_CONTRACT_HEADER = "Source materialization preservation contract"
PRESERVATION_DO_NOT_REMOVE = "Do not remove write_file"
COMPACT_PRESERVATION_DO_NOT_REMOVE = "Do not remove write_file/append_file"
VMA_REMOVE_WRITES_GUIDANCE = (
    "This is a verification-profile task. Remove write_file, append_file, "
    "and replace_in_file operations that target source files. Replace them "
    "with read-only inspection commands and test/verification commands."
)

VMA_MESSAGE_REASON_802 = (
    "Verification/review plan mutates app source assets instead of only "
    "verifying the current workspace "
    "(files: ['calclib/arithmetic.py', 'calclib/stats.py'])"
)
VMA_CODE_REASON_802 = (
    "verification_mutates_source_assets "
    "(files: ['calclib/arithmetic.py', 'calclib/stats.py'])"
)
VMA_MESSAGE_REASON_814 = (
    "Verification/review plan mutates app source assets instead of only "
    "verifying the current workspace "
    "(files: ['strtools/format.py', 'strtools/transform.py', "
    "'strtools/validate.py'])"
)
NESTED_FOLDER_REASON_748 = (
    "Plan appears to generate the deliverable inside a new nested project "
    "folder instead of the task workspace root (steps: [2])"
)
IMPLEMENTATION_WEAK_VERIFICATION_REASON = "weak verification commands in steps [2, 3]"

T6_TASK_DESCRIPTION = (
    "Run .venv/bin/python3 -m pytest --tb=short. All tests must pass. "
    "Report pass count and any failures."
)
IMPLEMENTATION_TASK_DESCRIPTION = (
    "Implement calclib arithmetic and stats modules so all tests pass."
)


def _calclib_workspace(tmp_path: Path) -> Path:
    (tmp_path / "calclib").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "calclib" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "calclib" / "arithmetic.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "calclib" / "stats.py").write_text(
        "def mean(values):\n    return sum(values) / len(values)\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_arithmetic.py").write_text(
        "from calclib.arithmetic import add\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return tmp_path


def _strtools_workspace(tmp_path: Path) -> Path:
    (tmp_path / "strtools").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "strtools" / "__init__.py").write_text("", encoding="utf-8")
    for module in ("format", "transform", "validate"):
        (tmp_path / "strtools" / f"{module}.py").write_text(
            "def truncate(text, length):\n    return text[:length]\n",
            encoding="utf-8",
        )
    (tmp_path / "tests" / "test_format.py").write_text(
        "from strtools.format import truncate\n\n"
        "def test_truncate():\n    assert truncate('abcdef', 3) == 'abc'\n",
        encoding="utf-8",
    )
    return tmp_path


def _calclib_t6_plan_802() -> list[dict]:
    return [
        {
            "step_number": 1,
            "description": "Inspect current workspace sources and tests",
            "commands": [
                "cat calclib/arithmetic.py calclib/stats.py " "tests/test_arithmetic.py"
            ],
            "verification": None,
            "rollback": None,
            "expected_files": [],
        },
        {
            "step_number": 2,
            "description": (
                "Fix calclib/arithmetic.py to ensure correct implementations "
                "of add, subtract, multiply, divide"
            ),
            "commands": [],
            "verification": "python3 -m py_compile calclib/arithmetic.py",
            "rollback": None,
            "expected_files": ["calclib/arithmetic.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "calclib/arithmetic.py",
                    "content": (
                        "def add(a, b):\n    return a + b\n\n"
                        "def subtract(a, b):\n    return a - b\n\n"
                        "def multiply(a, b):\n    return a * b\n\n"
                        "def divide(a, b):\n    if b == 0:\n"
                        "        raise ValueError('division by zero')\n"
                        "    return a / b\n"
                    ),
                }
            ],
        },
        {
            "step_number": 3,
            "description": "Fix calclib/stats.py mean and median implementations",
            "commands": [],
            "verification": "python3 -m py_compile calclib/stats.py",
            "rollback": None,
            "expected_files": ["calclib/stats.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "calclib/stats.py",
                    "content": (
                        "def mean(values):\n" "    return sum(values) / len(values)\n"
                    ),
                }
            ],
        },
        {
            "step_number": 4,
            "description": "Run the full test suite",
            "commands": [".venv/bin/python3 -m pytest --tb=short"],
            "verification": ".venv/bin/python3 -m pytest --tb=short",
            "rollback": None,
            "expected_files": [],
        },
    ]


def _strtools_t6_plan_814() -> list[dict]:
    return [
        {
            "step_number": 1,
            "description": "Inspect strtools modules and tests",
            "commands": ["cat strtools/format.py tests/test_format.py"],
            "verification": None,
            "rollback": None,
            "expected_files": [],
        },
        {
            "step_number": 2,
            "description": "Implement strtools/format.py based on test expectations",
            "commands": [],
            "verification": "python3 -m py_compile strtools/format.py",
            "rollback": None,
            "expected_files": ["strtools/format.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "strtools/format.py",
                    "content": "def format_string(text):\n    return text\n",
                }
            ],
        },
        {
            "step_number": 3,
            "description": "Implement strtools/transform.py and strtools/validate.py",
            "commands": [],
            "verification": "python3 -m py_compile strtools/transform.py",
            "rollback": None,
            "expected_files": ["strtools/transform.py", "strtools/validate.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "strtools/transform.py",
                    "content": "def transform_string(text):\n    return text\n",
                },
                {
                    "op": "write_file",
                    "path": "strtools/validate.py",
                    "content": "def validate_string(text):\n    return True\n",
                },
            ],
        },
        {
            "step_number": 4,
            "description": "Run the full test suite",
            "commands": [".venv/bin/python3 -m pytest --tb=short"],
            "verification": ".venv/bin/python3 -m pytest --tb=short",
            "rollback": None,
            "expected_files": [],
        },
    ]


def _build_vma_repair_prompt(
    tmp_path: Path,
    rejection_reasons: list[str],
    plan: list[dict] | None = None,
    workspace_builder=_calclib_workspace,
    task_description: str = T6_TASK_DESCRIPTION,
) -> str:
    project_dir = workspace_builder(tmp_path)
    return build_planning_repair_prompt(
        task_description=task_description,
        malformed_output=json.dumps(plan or _calclib_t6_plan_802()),
        project_dir=project_dir,
        rejection_reasons=rejection_reasons,
    )


def test_vma_message_rejection_omits_preservation_contract(tmp_path):
    prompt = _build_vma_repair_prompt(tmp_path, [VMA_MESSAGE_REASON_802])

    assert PRESERVATION_CONTRACT_HEADER not in prompt
    assert PRESERVATION_DO_NOT_REMOVE not in prompt
    assert "Required materialization paths" not in prompt


def test_vma_code_rejection_omits_preservation_contract(tmp_path):
    prompt = _build_vma_repair_prompt(tmp_path, [VMA_CODE_REASON_802])

    assert PRESERVATION_CONTRACT_HEADER not in prompt
    assert PRESERVATION_DO_NOT_REMOVE not in prompt
    assert "Required materialization paths" not in prompt


def test_vma_message_rejection_includes_remove_writes_guidance(tmp_path):
    prompt = _build_vma_repair_prompt(tmp_path, [VMA_MESSAGE_REASON_802])

    assert VMA_REMOVE_WRITES_GUIDANCE in prompt


def test_vma_code_rejection_includes_remove_writes_guidance(tmp_path):
    prompt = _build_vma_repair_prompt(tmp_path, [VMA_CODE_REASON_802])

    assert VMA_REMOVE_WRITES_GUIDANCE in prompt


def test_vma_rejection_shows_violating_paths_802(tmp_path):
    prompt = _build_vma_repair_prompt(tmp_path, [VMA_CODE_REASON_802])

    assert "calclib/arithmetic.py" in prompt
    assert "calclib/stats.py" in prompt
    assert "write_file" in prompt


def test_vma_rejection_shows_violating_paths_814(tmp_path):
    prompt = _build_vma_repair_prompt(
        tmp_path,
        [VMA_MESSAGE_REASON_814],
        plan=_strtools_t6_plan_814(),
        workspace_builder=_strtools_workspace,
    )

    assert PRESERVATION_CONTRACT_HEADER not in prompt
    assert VMA_REMOVE_WRITES_GUIDANCE in prompt
    assert "strtools/format.py" in prompt
    assert "strtools/transform.py" in prompt
    assert "strtools/validate.py" in prompt


def test_vma_with_secondary_rejection_omits_preservation_contract_748(tmp_path):
    # Task 748 first pass failed with vma + a nested-folder reason together.
    prompt = _build_vma_repair_prompt(
        tmp_path,
        [VMA_MESSAGE_REASON_802, NESTED_FOLDER_REASON_748],
    )

    assert PRESERVATION_CONTRACT_HEADER not in prompt
    assert PRESERVATION_DO_NOT_REMOVE not in prompt
    assert VMA_REMOVE_WRITES_GUIDANCE in prompt


def test_implementation_rejection_keeps_preservation_contract(tmp_path):
    prompt = _build_vma_repair_prompt(
        tmp_path,
        [IMPLEMENTATION_WEAK_VERIFICATION_REASON],
        task_description=IMPLEMENTATION_TASK_DESCRIPTION,
    )

    assert PRESERVATION_CONTRACT_HEADER in prompt
    assert PRESERVATION_DO_NOT_REMOVE in prompt
    assert (
        "Required materialization paths from the rejected plan: "
        "calclib/arithmetic.py, calclib/stats.py"
    ) in prompt
    assert "Verification-profile repair required" not in prompt
    assert VMA_REMOVE_WRITES_GUIDANCE not in prompt


def test_compact_vma_rejection_omits_preservation_contract():
    prompt = build_compact_planning_repair_prompt(
        malformed_output=json.dumps(_calclib_t6_plan_802()),
        rejection_reasons=[VMA_MESSAGE_REASON_802],
    )

    assert PRESERVATION_CONTRACT_HEADER not in prompt
    assert COMPACT_PRESERVATION_DO_NOT_REMOVE not in prompt
    assert VMA_REMOVE_WRITES_GUIDANCE in prompt


def test_compact_implementation_rejection_keeps_preservation_contract():
    prompt = build_compact_planning_repair_prompt(
        malformed_output=json.dumps(_calclib_t6_plan_802()),
        rejection_reasons=[IMPLEMENTATION_WEAK_VERIFICATION_REASON],
    )

    assert PRESERVATION_CONTRACT_HEADER in prompt
    assert "Verification-profile repair required" not in prompt


def test_verification_profile_minimal_prompt_includes_read_only_contract(tmp_path):
    project_dir = _calclib_workspace(tmp_path)
    prompt = PlannerService.build_minimal_planning_prompt(
        T6_TASK_DESCRIPTION,
        project_dir,
        validation_profile="verification",
    )

    assert VERIFICATION_PROFILE_PLANNING_CONTRACT_LINE in prompt


def test_verification_profile_ultra_minimal_prompt_includes_read_only_contract(
    tmp_path,
):
    project_dir = _calclib_workspace(tmp_path)
    prompt = PlannerService.build_ultra_minimal_planning_prompt(
        T6_TASK_DESCRIPTION,
        project_dir,
        validation_profile="verification",
    )

    assert VERIFICATION_PROFILE_PLANNING_CONTRACT_LINE in prompt


def test_implementation_profile_minimal_prompt_excludes_read_only_contract(
    tmp_path,
):
    project_dir = _calclib_workspace(tmp_path)
    for profile in (None, "implementation"):
        prompt = PlannerService.build_minimal_planning_prompt(
            IMPLEMENTATION_TASK_DESCRIPTION,
            project_dir,
            validation_profile=profile,
        )
        assert VERIFICATION_PROFILE_PLANNING_CONTRACT_LINE not in prompt


def test_implementation_profile_minimal_prompt_is_byte_identical(tmp_path):
    project_dir = _calclib_workspace(tmp_path)
    default_prompt = PlannerService.build_minimal_planning_prompt(
        IMPLEMENTATION_TASK_DESCRIPTION,
        project_dir,
    )
    implementation_prompt = PlannerService.build_minimal_planning_prompt(
        IMPLEMENTATION_TASK_DESCRIPTION,
        project_dir,
        validation_profile="implementation",
    )

    assert default_prompt == implementation_prompt


def test_implementation_profile_ultra_minimal_prompt_is_byte_identical(tmp_path):
    project_dir = _calclib_workspace(tmp_path)
    default_prompt = PlannerService.build_ultra_minimal_planning_prompt(
        IMPLEMENTATION_TASK_DESCRIPTION,
        project_dir,
    )
    implementation_prompt = PlannerService.build_ultra_minimal_planning_prompt(
        IMPLEMENTATION_TASK_DESCRIPTION,
        project_dir,
        validation_profile="implementation",
    )

    assert default_prompt == implementation_prompt


def test_read_only_contract_line_avoids_profile_marker_words():
    lowered = VERIFICATION_PROFILE_PLANNING_CONTRACT_LINE.lower()
    for marker in ("fix", "repair", "update", "modify", "write", "change", "preserve"):
        assert marker not in lowered


def test_read_only_contract_line_does_not_flip_profile_inference():
    # Task 706's "Do not modify any files" flipped the profile to
    # implementation; the contract line must not repeat that trap.
    profile = ValidatorService.infer_validation_profile(
        f"{T6_TASK_DESCRIPTION}\n{VERIFICATION_PROFILE_PLANNING_CONTRACT_LINE}",
        "full_lifecycle",
        title="Final verification",
        description=T6_TASK_DESCRIPTION,
    )

    assert profile == "verification"

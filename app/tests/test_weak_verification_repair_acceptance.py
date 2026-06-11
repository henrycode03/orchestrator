from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services.orchestration.phases.planning_repair_arbitration_control import (
    arbitrate_planning_repair_candidate,
)
from app.services.orchestration.phases.planning_support import _PlanningRetryState


def _bootstrap_plan(package: str) -> list[dict]:
    return [
        {
            "step_number": 1,
            "description": "Create package skeleton",
            "commands": [],
            "verification": (
                f"test -f {package}/__init__.py && "
                "test -f setup.py && test -f requirements.txt"
            ),
            "rollback": None,
            "expected_files": [
                f"{package}/__init__.py",
                "tests/__init__.py",
                "setup.py",
                "requirements.txt",
            ],
            "ops": [
                {"op": "mkdir", "path": package},
                {
                    "op": "write_file",
                    "path": f"{package}/__init__.py",
                    "content": "__version__ = '0.1.0'\n",
                },
                {"op": "mkdir", "path": "tests"},
                {
                    "op": "write_file",
                    "path": "tests/__init__.py",
                    "content": "",
                },
                {
                    "op": "write_file",
                    "path": "setup.py",
                    "content": (
                        "from setuptools import setup\n"
                        f"setup(name='{package}', version='0.1.0', "
                        f"packages=['{package}'])\n"
                    ),
                },
                {
                    "op": "write_file",
                    "path": "requirements.txt",
                    "content": "pytest\n",
                },
            ],
        },
        {
            "step_number": 2,
            "description": "Create virtual environment and install dependencies",
            "commands": [
                "python3 -m venv .venv",
                ".venv/bin/pip install -e .",
                ".venv/bin/pip install -r requirements.txt",
            ],
            "verification": f".venv/bin/pip show {package}",
            "rollback": "rm -rf .venv",
            "expected_files": [],
            "ops": [],
        },
        {
            "step_number": 3,
            "description": "Verify test discovery",
            "commands": [".venv/bin/python3 -m pytest --collect-only"],
            "verification": ".venv/bin/python3 -m pytest --collect-only",
            "rollback": None,
            "expected_files": [],
            "ops": [],
        },
    ]


def _collapsed_pathtools_repair() -> list[dict]:
    files = [
        "pathtools/__init__.py",
        "tests/__init__.py",
        "setup.py",
    ]
    verify = (
        'python -c "import pathlib,sys; files='
        + repr(files)
        + '; sys.exit(0 if all(pathlib.Path(p).exists() for p in files) else 1)"'
    )
    return [
        {
            "step_number": 1,
            "description": "Inspect workspace",
            "commands": ["rg --files . | sort"],
            "verification": 'python -c "import sys; sys.exit(0)"',
            "rollback": None,
            "expected_files": [],
            "ops": [],
        },
        {
            "step_number": 2,
            "description": "Create package skeleton",
            "commands": [verify],
            "verification": verify,
            "rollback": None,
            "expected_files": files,
            "ops": [
                {"op": "mkdir", "path": "pathtools"},
                {
                    "op": "write_file",
                    "path": "pathtools/__init__.py",
                    "content": "__version__ = '0.1.0'\n",
                },
                {"op": "mkdir", "path": "tests"},
                {
                    "op": "write_file",
                    "path": "tests/__init__.py",
                    "content": "",
                },
                {
                    "op": "write_file",
                    "path": "setup.py",
                    "content": (
                        "from setuptools import setup\n"
                        "setup(name='pathtools', version='0.1.0', "
                        "packages=['pathtools'])\n"
                    ),
                },
            ],
        },
        {
            "step_number": 3,
            "description": "Verify requested change",
            "commands": [verify],
            "verification": verify,
            "rollback": None,
            "expected_files": [],
            "ops": [],
        },
    ]


def _collapsed_strtools_repair() -> list[dict]:
    return [
        {
            "step_number": 1,
            "description": "Create package structure",
            "commands": [],
            "verification": (
                "python -c \"import sys; sys.path.insert(0, '.'); "
                "import strtools; assert strtools.__version__ == '0.1.0'\""
            ),
            "rollback": None,
            "expected_files": [
                "strtools/__init__.py",
                "tests/__init__.py",
                "setup.py",
            ],
            "ops": [
                {"op": "mkdir", "path": "strtools"},
                {"op": "mkdir", "path": "tests"},
                {
                    "op": "write_file",
                    "path": "strtools/__init__.py",
                    "content": "__version__ = '0.1.0'\n",
                },
                {
                    "op": "write_file",
                    "path": "tests/__init__.py",
                    "content": "",
                },
                {
                    "op": "write_file",
                    "path": "setup.py",
                    "content": (
                        "from setuptools import setup\n"
                        "setup(name='strtools', version='0.1.0', "
                        "packages=['strtools'])\n"
                    ),
                },
            ],
        },
        {
            "step_number": 2,
            "description": "Install package",
            "commands": ["pip install -e ."],
            "verification": (
                'python -c "import strtools; print(strtools.__version__)"'
            ),
            "rollback": "pip uninstall -y strtools",
            "expected_files": [],
            "ops": [],
        },
        {
            "step_number": 3,
            "description": "Verify package",
            "commands": ["python -m pytest --collect-only"],
            "verification": (
                'python -c "import strtools; '
                "assert strtools.__version__ == '0.1.0'\""
            ),
            "rollback": None,
            "expected_files": [],
            "ops": [],
        },
    ]


def _ctx(*, plan: list[dict], project_dir: Path, package: str) -> SimpleNamespace:
    prompt = f"Bootstrap the {package} package with a venv and pytest."
    task = SimpleNamespace(
        title=f"Bootstrap {package} package",
        description=prompt,
        plan_position=1,
        status=None,
        error_message=None,
    )
    return SimpleNamespace(
        task=task,
        orchestration_state=SimpleNamespace(
            plan=plan,
            project_dir=project_dir,
            project_context="",
            status=None,
            abort_reason=None,
            reasoning_artifact=None,
        ),
        prompt=prompt,
        execution_profile="full_lifecycle",
        validation_severity="standard",
        workflow_profile=None,
        workflow_stage=None,
        session_id=1,
        task_id=1,
        task_execution_id=1,
        session_instance_id=None,
        logger=logging.getLogger("test.weak_verification_repair_acceptance"),
        emit_live=MagicMock(),
        db=MagicMock(),
        restore_workspace_snapshot_if_needed=None,
    )


def test_regressed_weak_verification_repair_preserves_complete_bootstrap_plan(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "app.services.orchestration.phases.planning_repair_arbitration_control"
        ".append_orchestration_event",
        lambda *args, **kwargs: {},
    )
    original = _bootstrap_plan("pathtools")
    candidate = _collapsed_pathtools_repair()
    ctx = _ctx(plan=candidate, project_dir=tmp_path, package="pathtools")
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.last_repair_reason = "plan_contains_immediate_repair_issues"

    result = arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=retry_state,
        previous_plan=copy.deepcopy(original),
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text="[]",
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=lambda **kwargs: {"output": "[]"},
    )

    assert result["action"] == "replace"
    assert ctx.orchestration_state.plan[1:] == original[1:]
    assert ctx.orchestration_state.plan[0]["ops"] == original[0]["ops"]
    assert (
        ctx.orchestration_state.plan[0]["verification"] == candidate[1]["verification"]
    )


def test_strtools_regression_preserves_venv_install_and_test_obligations(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "app.services.orchestration.phases.planning_repair_arbitration_control"
        ".append_orchestration_event",
        lambda *args, **kwargs: {},
    )
    original = _bootstrap_plan("strtools")
    candidate = _collapsed_strtools_repair()
    ctx = _ctx(plan=candidate, project_dir=tmp_path, package="strtools")
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.last_repair_reason = "plan_contains_immediate_repair_issues"

    result = arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=retry_state,
        previous_plan=copy.deepcopy(original),
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text="[]",
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=lambda **kwargs: {"output": "[]"},
    )

    assert result["action"] == "replace"
    assert ctx.orchestration_state.plan[1:] == original[1:]
    assert (
        ctx.orchestration_state.plan[0]["verification"] == candidate[0]["verification"]
    )


def test_full_shape_weak_verification_improvement_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.orchestration.phases.planning_repair_arbitration_control"
        ".append_orchestration_event",
        lambda *args, **kwargs: {},
    )
    original = _bootstrap_plan("pathtools")
    candidate = copy.deepcopy(original)
    candidate[0]["verification"] = (
        'python -c "import pathlib; '
        "assert pathlib.Path('requirements.txt').read_text().strip() == 'pytest'\""
    )
    ctx = _ctx(plan=candidate, project_dir=tmp_path, package="pathtools")
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.last_repair_reason = "plan_contains_immediate_repair_issues"

    result = arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=retry_state,
        previous_plan=copy.deepcopy(original),
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text="[]",
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=lambda **kwargs: {"output": "[]"},
    )

    assert result["action"] == "none"
    assert ctx.orchestration_state.plan == candidate


def test_non_weak_repair_acceptance_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.orchestration.phases.planning_repair_arbitration_control"
        ".append_orchestration_event",
        lambda *args, **kwargs: {},
    )
    original = _bootstrap_plan("pathtools")
    original[0]["verification"] = (
        'python -c "import pathlib; '
        "assert pathlib.Path('requirements.txt').exists()\""
    )
    candidate = _collapsed_pathtools_repair()
    ctx = _ctx(plan=candidate, project_dir=tmp_path, package="pathtools")
    retry_state = _PlanningRetryState()
    retry_state.repair_prompt_used = True
    retry_state.last_repair_reason = "plan_contains_immediate_repair_issues"

    result = arbitrate_planning_repair_candidate(
        ctx=ctx,
        retry_state=retry_state,
        previous_plan=copy.deepcopy(original),
        immediate_repair_issues={},
        planning_phase_event=None,
        output_text="[]",
        planning_timeout_seconds=60,
        prompt_profile=None,
        repair_planning_output=lambda **kwargs: {"output": "[]"},
    )

    assert result["action"] == "none"
    assert ctx.orchestration_state.plan == candidate

"""PHASE34-P1 — existing-file mutation contract expressibility.

GX1 produced a functionally correct plan that the validator rejected with
``existing_file_write_requires_explicit_replace_authorization``.  The
authorization is lexical: ``_existing_write_authorized`` looks for
replace/rewrite/overwrite/rebuild/preserve near the path.  Planning and repair
were never told that, so a correct edit could fail purely on the verb chosen
for the step description.  These tests pin the guidance to the prompts that can
act on it, and pin the validator semantics that are deliberately unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.orchestration.planning.planner import PlannerService
from app.services.orchestration.planning.prompt_contracts import (
    render_existing_file_mutation_contract,
)
from app.services.orchestration.planning.repair_prompts import (
    build_planning_repair_prompt,
)
from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
)
from app.services.orchestration.validation.validator import ValidatorService

AUTHORIZATION_CODE = "existing_file_write_requires_explicit_replace_authorization"
TASK = "Change greet() in greeter.py so it strips whitespace from name."
ORIGINAL = 'def greet(name):\n    return "Hello, " + name\n'
UPDATED = 'def greet(name):\n    return "Hello, " + name.strip()\n'
VERIFY = "python -m pytest -q test_greeter.py"


def _project(tmp_path: Path) -> Path:
    (tmp_path / "greeter.py").write_text(ORIGINAL)
    (tmp_path / "test_greeter.py").write_text(
        "from greeter import greet\n\n\ndef test_greet():\n"
        '    assert greet("world") == "Hello, world"\n'
    )
    return tmp_path


def _materialization(project_dir: Path, *, expected=("greeter.py",), create=()):
    return materialize_planner_source_context(
        project_dir,
        task_description=TASK,
        expected_paths=list(expected),
        creation_authorized_paths=list(create),
    )


def _plan(description: str, ops: list[dict], expected_files: list[str]):
    return [
        {
            "step_number": 1,
            "description": description,
            "commands": [],
            "verification": VERIFY,
            "rollback": None,
            "expected_files": list(expected_files),
            "ops": ops,
        },
        {
            "step_number": 2,
            "description": "Run the focused test",
            "commands": [VERIFY],
            "verification": VERIFY,
            "rollback": None,
            "expected_files": [],
            "ops": [],
        },
    ]


def _validate(project_dir: Path, plan, *, expected=("greeter.py",), create=()):
    return ValidatorService().validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=TASK,
        execution_profile="full_lifecycle",
        project_dir=project_dir,
        source_materialization=_materialization(
            project_dir, expected=expected, create=create
        ),
    )


def test_contract_names_both_legal_existing_file_shapes():
    text = render_existing_file_mutation_contract()
    assert "replace_in_file" in text
    assert "write_file" in text
    for word in ("replace", "rewrite", "overwrite", "rebuild", "preserve"):
        assert word in text
    assert "expected_files" in text


def test_contract_is_omitted_without_grounded_existing_source():
    assert render_existing_file_mutation_contract(legacy_replace_available=False) == ""


def test_planning_prompt_exposes_contract_when_existing_source_is_supplied(tmp_path):
    project_dir = _project(tmp_path)
    prompt = PlannerService.build_minimal_planning_prompt(
        TASK,
        project_dir,
        workspace_has_existing_files=True,
        source_materialization=_materialization(project_dir),
    )
    assert any(line.startswith("13b.") for line in prompt.splitlines())
    assert "no bare `write_file`" in prompt


def test_planning_prompt_omits_contract_without_source_materialization(tmp_path):
    prompt = PlannerService.build_minimal_planning_prompt(
        TASK, _project(tmp_path), workspace_has_existing_files=True
    )
    assert not any(line.startswith("13b.") for line in prompt.splitlines())


def test_repair_prompt_explains_the_authorization_code_it_reports(tmp_path):
    prompt = build_planning_repair_prompt(
        TASK, "[]", _project(tmp_path), rejection_reasons=[AUTHORIZATION_CODE]
    )
    assert any(line.startswith("2y.") for line in prompt.splitlines())
    assert "no bare `write_file`" in prompt


def test_repair_prompt_omits_contract_for_unrelated_rejections(tmp_path):
    prompt = build_planning_repair_prompt(
        TASK, "[]", _project(tmp_path), rejection_reasons=["malformed planning output"]
    )
    assert not any(line.startswith("2y.") for line in prompt.splitlines())


# ---- validator semantics are deliberately unchanged -----------------------


def test_new_file_creation_still_validates(tmp_path):
    project_dir = _project(tmp_path)
    outcome = _validate(
        project_dir,
        _plan(
            "Create notes.md with project notes",
            [{"op": "write_file", "path": "notes.md", "content": "# Notes\n"}],
            ["notes.md"],
        ),
        expected=("notes.md",),
        create=("notes.md",),
    )
    assert outcome.status == "accepted"


def test_bare_write_to_existing_file_is_still_rejected(tmp_path):
    project_dir = _project(tmp_path)
    outcome = _validate(
        project_dir,
        _plan(
            "Update greeter.py to strip whitespace from name",
            [{"op": "write_file", "path": "greeter.py", "content": UPDATED}],
            ["greeter.py"],
        ),
    )
    assert outcome.status == "repair_required"
    assert any(AUTHORIZATION_CODE in reason for reason in outcome.reasons)


def test_explicit_whole_file_replacement_validates(tmp_path):
    project_dir = _project(tmp_path)
    outcome = _validate(
        project_dir,
        _plan(
            "Rewrite greeter.py to strip whitespace from name",
            [{"op": "write_file", "path": "greeter.py", "content": UPDATED}],
            ["greeter.py"],
        ),
    )
    assert outcome.status == "accepted"


def test_grounded_replace_in_file_validates(tmp_path):
    project_dir = _project(tmp_path)
    outcome = _validate(
        project_dir,
        _plan(
            "Update greeter.py to strip whitespace from name",
            [
                {
                    "op": "replace_in_file",
                    "path": "greeter.py",
                    "old": 'return "Hello, " + name',
                    "new": 'return "Hello, " + name.strip()',
                }
            ],
            ["greeter.py"],
        ),
    )
    assert outcome.status == "accepted"


def test_stale_replace_old_text_is_still_rejected(tmp_path):
    project_dir = _project(tmp_path)
    outcome = _validate(
        project_dir,
        _plan(
            "Update greeter.py to strip whitespace from name",
            [
                {
                    "op": "replace_in_file",
                    "path": "greeter.py",
                    "old": 'return "Howdy, " + name',
                    "new": 'return "Hello, " + name.strip()',
                }
            ],
            ["greeter.py"],
        ),
    )
    assert outcome.status == "repair_required"

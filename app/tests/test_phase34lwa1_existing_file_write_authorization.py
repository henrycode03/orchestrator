"""PHASE34-LWA1 — structured whole-file intent replaces prose authorization.

The two historical Fixture-D Plans are loaded verbatim from the durable S2A
and S2A-R1 normalized evidence.  The tests below are intentionally written
against the desired post-fix contract; the neutral-description and historical
anchor tests reproduce the current lexical false positive before the validator
change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
)
from app.services.orchestration.validation.validator import ValidatorService

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION_CODE = "existing_file_write_requires_explicit_replace_authorization"
TASK_D = (
    "Add support for recording a task priority in the task API and data handling. "
    "Update automated tests."
)
TASK_NEUTRAL = "Add priority support to the task data handling file."
TASK_CREATE_ONLY = "Create a new task helper module."
TASK_UNGROUNDED = "Update mystery.py with the requested behavior."
ORIGINAL_TASKS = (
    '"""Task data handling."""\n\n'
    "def make_task(title: str) -> dict:\n"
    '    return {"title": title}\n'
)
UPDATED_TASKS = (
    '"""Task data handling."""\n\n'
    "def make_task(title: str, priority: str | None = None) -> dict:\n"
    '    task = {"title": title}\n'
    "    if priority is not None:\n"
    '        task["priority"] = priority\n'
    "    return task\n"
)
API_SOURCE = (
    "from tasks import make_task\n\n\n"
    "def create_task(payload: dict) -> dict:\n"
    '    return make_task(payload["title"])\n'
)
TEST_SOURCE = (
    "from api import create_task\n\n\n"
    "def test_create_task():\n"
    '    assert create_task({"title": "Write docs"}) == {"title": "Write docs"}\n'
)
VERIFY = "python -m pytest test_tasks.py -v"


def _project(tmp_path: Path, *, include_task_files: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if include_task_files:
        (tmp_path / "tasks.py").write_text(ORIGINAL_TASKS, encoding="utf-8")
        (tmp_path / "api.py").write_text(API_SOURCE, encoding="utf-8")
        (tmp_path / "test_tasks.py").write_text(TEST_SOURCE, encoding="utf-8")
    return tmp_path


def _materialization(
    project_dir: Path,
    *,
    task: str = TASK_D,
    expected: tuple[str, ...] = ("tasks.py",),
    create: tuple[str, ...] = (),
):
    return materialize_planner_source_context(
        project_dir,
        task_description=task,
        expected_paths=list(expected),
        creation_authorized_paths=list(create),
    )


def _validate(
    project_dir: Path,
    plan: list[dict],
    *,
    task: str = TASK_D,
    expected: tuple[str, ...] = ("tasks.py",),
    create: tuple[str, ...] = (),
    intent_mode: str = "default",
):
    return ValidatorService().validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=task,
        execution_profile="full_lifecycle",
        project_dir=project_dir,
        source_materialization=_materialization(
            project_dir, task=task, expected=expected, create=create
        ),
        intent_mode=intent_mode,
    )


def _write_plan(
    description: str,
    *,
    path: str = "tasks.py",
    content: str = UPDATED_TASKS,
    expected_files: list[str] | None = None,
    verification: str = VERIFY,
):
    return [
        {
            "step_number": 1,
            "description": description,
            "commands": [],
            "verification": verification,
            "rollback": None,
            "expected_files": expected_files if expected_files is not None else [path],
            "ops": [{"op": "write_file", "path": path, "content": content}],
        },
        {
            "step_number": 2,
            "description": "Run the focused project test.",
            "commands": [verification],
            "verification": verification,
            "rollback": None,
            "expected_files": [],
            "ops": [],
        },
    ]


def _historical_anchor(evidence_path: str) -> list[dict]:
    evidence = json.loads((ROOT / evidence_path).read_text(encoding="utf-8"))
    row = next(row for row in evidence["results"] if row["fixture_id"] == "D")
    return row["plan"]


def _historical_project(tmp_path: Path) -> Path:
    return _project(tmp_path)


def test_exact_s2a_fixture_d_anchor_reproduces_current_false_positive(tmp_path):
    outcome = _validate(
        _historical_project(tmp_path),
        _historical_anchor(
            "docs/roadmap/reports/evidence/phase34-s2a/current-results.json"
        ),
        expected=("tasks.py", "api.py", "test_tasks.py"),
    )
    assert outcome.status == "accepted"
    assert AUTHORIZATION_CODE not in outcome.reasons


def test_exact_s2a_r1_fixture_d_anchor_reproduces_current_false_positive(tmp_path):
    outcome = _validate(
        _historical_project(tmp_path),
        _historical_anchor(
            "docs/roadmap/reports/evidence/phase34-s2a-r1/current-results.json"
        ),
        expected=("tasks.py", "api.py", "test_tasks.py"),
    )
    assert outcome.status == "accepted"
    assert AUTHORIZATION_CODE not in outcome.reasons


def test_grounded_existing_write_with_neutral_description_is_accepted(tmp_path):
    outcome = _validate(
        _project(tmp_path),
        _write_plan("Adjust tasks.py to support the requested priority behavior."),
        task=TASK_NEUTRAL,
    )
    assert outcome.status == "accepted"


def test_grounded_existing_write_with_lexical_description_is_accepted(tmp_path):
    outcome = _validate(
        _project(tmp_path),
        _write_plan("Rewrite tasks.py to support the requested priority behavior."),
    )
    assert outcome.status == "accepted"


def test_description_vocabulary_alone_does_not_change_outcome(tmp_path):
    lexical = _validate(
        _project(tmp_path / "lexical"),
        _write_plan("Rewrite tasks.py to support priority."),
    )
    neutral = _validate(
        _project(tmp_path / "neutral"),
        _write_plan("Adjust tasks.py to support priority."),
        task=TASK_NEUTRAL,
    )
    assert lexical.status == neutral.status == "accepted"


def test_grounded_replace_in_file_behavior_is_preserved(tmp_path):
    plan = _write_plan("Adjust tasks.py to support priority.")
    plan[0]["ops"] = [
        {
            "op": "replace_in_file",
            "path": "tasks.py",
            "old": '    return {"title": title}',
            "new": '    return {"title": title, "priority": "normal"}',
        }
    ]
    outcome = _validate(_project(tmp_path), plan, task=TASK_NEUTRAL)
    assert outcome.status == "accepted"


def test_new_file_write_under_normal_creation_authority_is_preserved(tmp_path):
    project = _project(tmp_path)
    outcome = _validate(
        project,
        _write_plan(
            "Create helper.py with the task helper.",
            path="helper.py",
            content="def helper():\n    return 1\n",
            expected_files=["helper.py"],
            verification='python -c "import helper; assert helper.helper() == 1"',
        ),
        task="Create helper.py for the task.",
        expected=("helper.py",),
        create=("helper.py",),
    )
    assert outcome.status == "accepted"


def test_create_only_new_file_write_remains_legal(tmp_path):
    outcome = _validate(
        _project(tmp_path, include_task_files=False),
        _write_plan(
            "Create helper.py.",
            path="helper.py",
            content="def helper():\n    return 1\n",
            expected_files=["helper.py"],
            verification='python -c "import helper; assert helper.helper() == 1"',
        ),
        task=TASK_CREATE_ONLY,
        expected=("helper.py",),
        create=("helper.py",),
        intent_mode="create_only",
    )
    assert outcome.status == "accepted"


def test_create_only_existing_write_remains_rejected(tmp_path):
    outcome = _validate(
        _project(tmp_path),
        _write_plan("Adjust tasks.py."),
        task=TASK_CREATE_ONLY,
        intent_mode="create_only",
    )
    assert outcome.status != "accepted"
    assert any(
        "create_only_task_existing_path_mutation" in reason
        for reason in outcome.reasons
    )


def test_ungrounded_existing_write_remains_rejected(tmp_path):
    project = _project(tmp_path, include_task_files=False)
    (project / "mystery.py").write_text("VALUE = 1\n", encoding="utf-8")
    outcome = _validate(
        project,
        _write_plan(
            "Adjust mystery.py.",
            path="mystery.py",
            content="VALUE = 2\n",
            expected_files=["mystery.py"],
        ),
        task=TASK_UNGROUNDED,
        expected=(),
    )
    assert outcome.status != "accepted"


def test_traversal_write_remains_rejected(tmp_path):
    outcome = _validate(
        _project(tmp_path),
        _write_plan("Adjust the task data.", path="../tasks.py"),
        task=TASK_NEUTRAL,
    )
    assert outcome.status != "accepted"


def test_absolute_external_write_remains_rejected(tmp_path):
    outcome = _validate(
        _project(tmp_path),
        _write_plan("Adjust the task data.", path="/tmp/outside.py"),
        task=TASK_NEUTRAL,
    )
    assert outcome.status != "accepted"


@pytest.mark.parametrize(
    "protected_path", [".agent/config.json", ".openclaw/runtime.json"]
)
def test_protected_runtime_write_remains_rejected(tmp_path, protected_path):
    outcome = _validate(
        _project(tmp_path),
        _write_plan("Adjust the runtime data.", path=protected_path),
        task=TASK_NEUTRAL,
        expected=(),
    )
    assert outcome.status != "accepted"


def test_existing_path_multiple_legal_writes_keep_c8_behavior(tmp_path):
    plan = _write_plan("Adjust tasks.py.")
    plan[0]["ops"].append(
        {
            "op": "write_file",
            "path": "tasks.py",
            "content": UPDATED_TASKS + "# bounded\n",
        }
    )
    outcome = _validate(_project(tmp_path), plan, task=TASK_NEUTRAL)
    assert outcome.status == "accepted"


def test_absent_path_duplicate_writes_still_fail_c8(tmp_path):
    plan = _write_plan(
        "Create helper.py.",
        path="helper.py",
        content="VALUE = 1\n",
        expected_files=["helper.py"],
        verification="python -c \"import pathlib; assert pathlib.Path('helper.py').exists()\"",
    )
    plan[0]["ops"].append(
        {"op": "write_file", "path": "helper.py", "content": "VALUE = 2\n"}
    )
    outcome = _validate(
        _project(tmp_path),
        plan,
        task="Create helper.py.",
        expected=("helper.py",),
        create=("helper.py",),
    )
    assert outcome.status != "accepted"
    assert any(
        "incompatible_same_path_mutation_sequence" in reason
        for reason in outcome.reasons
    )


def test_independent_invalid_expected_file_path_still_fails(tmp_path):
    outcome = _validate(
        _project(tmp_path),
        _write_plan("Adjust tasks.py.", expected_files=["../tasks.py"]),
        task=TASK_NEUTRAL,
    )
    assert outcome.status != "accepted"
    assert any("unsafe expected file paths" in reason for reason in outcome.reasons)


def test_shell_overwrite_does_not_infer_structured_whole_file_authority(tmp_path):
    plan = _write_plan("Run the task data update.")
    plan[0]["ops"] = []
    plan[0]["commands"] = [
        "python -c \"from pathlib import Path; Path('tasks.py').write_text('VALUE = 2')\""
    ]
    outcome = _validate(_project(tmp_path), plan, task=TASK_NEUTRAL)
    assert outcome.status != "accepted"


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "delete_file", "path": "tasks.py"},
        {"op": "rename_file", "path": "tasks.py", "new_path": "renamed.py"},
        {"op": "move_file", "path": "tasks.py", "destination": "renamed.py"},
        {"op": "copy_file", "path": "tasks.py", "destination": "copy.py"},
    ],
)
def test_non_write_operation_classes_do_not_gain_whole_file_authority(
    tmp_path, operation
):
    plan = _write_plan("Adjust task data.")
    plan[0]["ops"] = [operation]
    outcome = _validate(_project(tmp_path), plan, task=TASK_NEUTRAL)
    assert outcome.status != "accepted"

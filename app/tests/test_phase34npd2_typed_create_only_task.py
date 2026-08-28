"""Provider-free NPD2 tests for the typed create-only Task contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.models import Project, Task
from app.schemas import TaskCreate, TaskResponse, TaskUpdate
from app.services.orchestration.planning.planning_prompts import (
    build_minimal_planning_prompt,
)
from app.services.orchestration.planning.read_only_discovery import (
    DISCOVERY_ADMISSION_REQUIRED,
    DISCOVERY_ADMISSION_SKIPPED,
    assess_discovery_admission,
)
from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
)
from app.task_intent import TaskIntentMode
from app.services.orchestration.validation.accepted_path_authority import (
    GrantClass,
    build_accepted_path_authority,
)
from app.services.orchestration.validation.validator import ValidatorService


TASK_229_TEXT = (
    "Create a useful local Python capability for converting temperatures between "
    "Celsius and Fahrenheit. Include a clean public conversion API, a simple "
    "command-line interface runnable with Python, automated tests covering normal "
    "values and boundary cases, and a short README with usage. Keep it "
    "deterministic and dependency-light; do not use secrets or external network "
    "services. Run the tests after implementation."
)


def _admission(root: Path, task: str, intent: str = "default"):
    materialization = materialize_planner_source_context(
        root,
        task_description=task,
        supporting_paths=(),
    )
    return assess_discovery_admission(
        prompt=task,
        planner_contract=None,
        materialization=materialization,
        intent_mode=intent,
    )


def _plan(*, path: str, operation: dict, expected_files=None, command=None):
    return [
        {
            "step_number": 1,
            "description": "Implement and verify the requested change.",
            "commands": [command] if command else [],
            "verification": "python -c \"print('verified')\"",
            "rollback": "",
            "expected_files": expected_files if expected_files is not None else [path],
            "ops": [operation],
        }
    ]


def _validate(root: Path, plan, task: str, intent: str = "default"):
    return ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=task,
        execution_profile="implementation",
        project_dir=root,
        intent_mode=intent,
    )


def test_t1_default_empty_project_keeps_conservative_discovery(tmp_path: Path):
    admission = _admission(tmp_path, TASK_229_TEXT)
    assert admission == type(admission)(
        DISCOVERY_ADMISSION_REQUIRED, "no_explicit_source_or_creation_path"
    )


def test_t2_typed_task_229_is_admitted_without_source_discovery(tmp_path: Path):
    admission = _admission(tmp_path, TASK_229_TEXT, TaskIntentMode.CREATE_ONLY.value)
    assert admission == type(admission)(
        DISCOVERY_ADMISSION_SKIPPED, "typed_create_only_intent"
    )


def test_t3_t4_new_file_plans_keep_normal_creation_authority(tmp_path: Path):
    paths = ["src/converter.py", "tests/test_converter.py"]
    task = "Create a deterministic temperature converter with tests."
    plan = _plan(
        path=paths[0],
        expected_files=paths,
        operation={
            "op": "write_file",
            "path": paths[0],
            "content": "def celsius_to_fahrenheit(value):\n    return value * 9 / 5 + 32\n",
        },
    )
    plan[0]["ops"].append(
        {
            "op": "write_file",
            "path": paths[1],
            "content": "def test_conversion():\n    assert True\n",
        }
    )
    verdict = _validate(tmp_path, plan, task, TaskIntentMode.CREATE_ONLY.value)
    assert verdict.accepted, verdict.reasons
    authority, undeclarable = build_accepted_path_authority(
        plan=plan,
        source_materialization=materialize_planner_source_context(
            tmp_path,
            task_description=task,
            expected_paths=paths,
            creation_authorized_paths=paths,
            supporting_paths=(),
        ),
        creation_requested_paths=paths,
    )
    assert undeclarable == ()
    assert {grant.path.value for grant in authority.grants} == set(paths)
    assert all(
        grant.grant_class is GrantClass.CREATION_AUTHORIZED
        and grant.baseline_content_hash is None
        for grant in authority.grants
    )


def test_t5_t6_existing_path_write_and_replace_are_rejected(tmp_path: Path):
    existing = tmp_path / "src" / "api.py"
    existing.parent.mkdir()
    existing.write_text("def answer():\n    return 1\n", encoding="utf-8")
    task = "Create an independent new capability."

    write_verdict = _validate(
        tmp_path,
        _plan(
            path="src/api.py",
            operation={"op": "write_file", "path": "src/api.py", "content": "x\n"},
        ),
        task,
        TaskIntentMode.CREATE_ONLY.value,
    )
    replace_verdict = _validate(
        tmp_path,
        _plan(
            path="src/api.py",
            operation={
                "op": "replace_in_file",
                "path": "src/api.py",
                "old": "return 1",
                "new": "return 2",
            },
        ),
        task,
        TaskIntentMode.CREATE_ONLY.value,
    )
    assert write_verdict.rejected
    assert replace_verdict.rejected
    assert "create_only_task_existing_path_mutation" in write_verdict.reasons
    assert "create_only_task_existing_path_mutation" in replace_verdict.reasons


def test_t7_delete_is_rejected_even_when_path_is_not_a_creation_target(tmp_path: Path):
    verdict = _validate(
        tmp_path,
        _plan(
            path="src/api.py",
            operation={"op": "delete_file", "path": "src/api.py"},
            expected_files=[],
        ),
        "Create an independent capability.",
        TaskIntentMode.CREATE_ONLY.value,
    )
    assert verdict.rejected
    assert "create_only_task_delete" in verdict.reasons


def test_t8_shell_write_to_existing_path_is_rejected(tmp_path: Path):
    existing = tmp_path / "existing.py"
    existing.write_text("value = 1\n", encoding="utf-8")
    verdict = _validate(
        tmp_path,
        _plan(
            path="existing.py",
            operation={"op": "mkdir", "path": "new"},
            command="printf 'value = 2\\n' > existing.py",
        ),
        "Create an independent capability.",
        TaskIntentMode.CREATE_ONLY.value,
    )
    assert verdict.rejected
    assert "create_only_task_existing_path_shell_write" in verdict.reasons


def test_t9_nonempty_project_can_add_an_independent_new_file(tmp_path: Path):
    existing = tmp_path / "src" / "api.py"
    existing.parent.mkdir()
    existing.write_text("value = 1\n", encoding="utf-8")
    task = "Add an independent documentation example."
    verdict = _validate(
        tmp_path,
        _plan(
            path="docs/example.md",
            operation={
                "op": "write_file",
                "path": "docs/example.md",
                "content": "# Example\n",
            },
        ),
        task,
        TaskIntentMode.CREATE_ONLY.value,
    )
    assert verdict.accepted, verdict.reasons


def test_t10_default_absent_alleged_existing_file_remains_fail_closed(tmp_path: Path):
    task = "Update app/config.py with the existing settings behavior."
    admission = _admission(tmp_path, task)
    assert admission.status == DISCOVERY_ADMISSION_REQUIRED
    assert admission.reason == "expected_source_status_not_grounded"
    verdict = _validate(
        tmp_path,
        _plan(
            path="app/config.py",
            operation={
                "op": "write_file",
                "path": "app/config.py",
                "content": "settings = {}\n",
            },
        ),
        task,
    )
    assert not verdict.accepted


def test_t11_create_only_does_not_reinterpret_update_as_create(tmp_path: Path):
    task = "Update app/config.py with the existing settings behavior."
    verdict = _validate(
        tmp_path,
        _plan(
            path="app/config.py",
            operation={
                "op": "replace_in_file",
                "path": "app/config.py",
                "old": "old",
                "new": "new",
            },
        ),
        task,
        TaskIntentMode.CREATE_ONLY.value,
    )
    assert verdict.rejected
    assert "create_only_task_existing_path_mutation" in verdict.reasons


def test_t12_unexpected_new_path_has_no_authority(tmp_path: Path):
    materialization = materialize_planner_source_context(
        tmp_path,
        task_description="Create a capability.",
        supporting_paths=("provider_selected.py",),
    )
    record = materialization.file_map()["provider_selected.py"]
    assert record.creation_authorized is False
    assert record.expected is False


def test_t13_expected_files_mismatch_is_not_accepted(tmp_path: Path):
    plan = _plan(
        path="expected.py",
        operation={"op": "write_file", "path": "actual.py", "content": "x\n"},
    )
    verdict = _validate(
        tmp_path, plan, "Create a capability.", TaskIntentMode.CREATE_ONLY.value
    )
    assert not verdict.accepted


def test_t14_t15_t16_unsafe_and_runtime_paths_remain_rejected(tmp_path: Path):
    for path in ("../outside.py", ".agent/events/state.jsonl", ".openclaw/state.json"):
        verdict = _validate(
            tmp_path,
            _plan(
                path=path,
                operation={"op": "write_file", "path": path, "content": "x\n"},
            ),
            "Create a capability.",
            TaskIntentMode.CREATE_ONLY.value,
        )
        assert verdict.rejected, (path, verdict.reasons)


def test_t17_default_grounded_existing_replace_remains_available(tmp_path: Path):
    existing = tmp_path / "src" / "api.py"
    existing.parent.mkdir()
    existing.write_text("def answer():\n    return 1\n", encoding="utf-8")
    plan = _plan(
        path="src/api.py",
        operation={
            "op": "replace_in_file",
            "path": "src/api.py",
            "old": "return 1",
            "new": "return 2",
        },
    )
    verdict = _validate(tmp_path, plan, "Replace return 1 in src/api.py.")
    assert verdict.accepted, verdict.reasons


def test_t18_c8_repeated_absent_path_mutation_remains_rejected_or_repairable(
    tmp_path: Path,
):
    plan = _plan(
        path="new.py",
        operation={"op": "write_file", "path": "new.py", "content": "x = 1\n"},
    )
    plan[0]["ops"].append({"op": "write_file", "path": "new.py", "content": "x = 2\n"})
    verdict = _validate(
        tmp_path, plan, "Create a capability.", TaskIntentMode.CREATE_ONLY.value
    )
    assert not verdict.accepted
    assert any(
        "incompatible_same_path_mutation_sequence" in reason
        for reason in verdict.reasons
    )


def test_t20_t21_t22_intent_defaults_prompt_and_authority_are_bounded(tmp_path: Path):
    task = TaskCreate(project_id=1, title="new task", description=TASK_229_TEXT)
    assert task.intent_mode is TaskIntentMode.DEFAULT
    update = TaskUpdate(intent_mode=TaskIntentMode.CREATE_ONLY)
    assert update.intent_mode is TaskIntentMode.CREATE_ONLY
    assert TaskResponse.model_fields["intent_mode"].default is TaskIntentMode.DEFAULT

    prompt = build_minimal_planning_prompt(
        TASK_229_TEXT,
        tmp_path,
        intent_mode=TaskIntentMode.CREATE_ONLY.value,
    )
    assert "This task is creation-only" in prompt
    assert "new project-relative files" in prompt
    default_prompt = build_minimal_planning_prompt(TASK_229_TEXT, tmp_path)
    assert "This task is creation-only" not in default_prompt

    admission = _admission(tmp_path, TASK_229_TEXT, TaskIntentMode.CREATE_ONLY.value)
    assert admission.status == DISCOVERY_ADMISSION_SKIPPED


def test_public_task_api_persists_optional_intent_and_freezes_it_after_start(
    authenticated_client, db_session
):
    project = Project(
        name="Typed intent API", workspace_path="/tmp/typed-intent", user_id=1
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    default_response = authenticated_client.post(
        "/api/v1/tasks",
        json={"project_id": project.id, "title": "Legacy default", "description": "x"},
    )
    create_only_response = authenticated_client.post(
        "/api/v1/tasks",
        json={
            "project_id": project.id,
            "title": "First capability",
            "description": TASK_229_TEXT,
            "intent_mode": "create_only",
        },
    )
    assert default_response.status_code == 201
    assert default_response.json()["intent_mode"] == "default"
    assert create_only_response.status_code == 201
    assert create_only_response.json()["intent_mode"] == "create_only"

    task = db_session.get(Task, create_only_response.json()["id"])
    assert task.intent_mode == TaskIntentMode.CREATE_ONLY.value
    task.started_at = datetime.now(UTC)
    db_session.commit()
    frozen_response = authenticated_client.put(
        f"/api/v1/tasks/{task.id}", json={"intent_mode": "default"}
    )
    assert frozen_response.status_code == 409

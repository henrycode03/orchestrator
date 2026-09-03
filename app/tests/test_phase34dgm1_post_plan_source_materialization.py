"""PHASE34-DGM1 — deterministic post-Plan source grounding regressions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import app.services.orchestration.phases.post_plan_source_grounding as grounding_module
from app.services.orchestration.operations.source_region_identity import (
    SourceRegionIdentity,
)
from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
)
from app.services.orchestration.validation.validator import ValidatorService

from app.services.orchestration.phases.post_plan_source_grounding import (
    ground_post_plan_source_materialization,
)


SOURCE_PATH = "src/greeting/formatting.py"
TEST_PATH = "tests/test_formatting.py"
SOURCE = (
    "def format_customer_name(value: str) -> str:\n"
    '    """Normalize surrounding and repeated whitespace in a display name."""\n'
    "    return value\n"
)
TEST = "from src.greeting.formatting import format_customer_name\n"
TASK = "Normalize customer names and add regression coverage."


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "src" / "greeting").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / SOURCE_PATH).write_text(SOURCE, encoding="utf-8")
    (tmp_path / TEST_PATH).write_text(TEST, encoding="utf-8")
    return tmp_path


def _semantic_operation(project_dir: Path) -> dict:
    source = (project_dir / SOURCE_PATH).read_bytes()
    record = materialize_planner_source_context(
        project_dir, task_description=TASK, expected_paths=[SOURCE_PATH]
    ).file_map()[SOURCE_PATH]
    selector = SourceRegionIdentity.from_region(
        canonical_path=SOURCE_PATH,
        expected_source_version=record.version_identity,
        start_byte=0,
        end_byte=len(source),
        selected_region_sha256=hashlib.sha256(source).hexdigest(),
    )
    return {
        "op": "replace_in_file",
        "path": SOURCE_PATH,
        "selector": selector.to_dict(),
        "new": SOURCE.replace("    return value", '    return "normalized"'),
    }


def _plan(operations: list[dict]) -> list[dict]:
    return [
        {
            "step_number": 1,
            "description": "Apply grounded source and test changes",
            "commands": [],
            "verification": "python3 -m pytest -q",
            "rollback": None,
            "expected_files": [SOURCE_PATH, TEST_PATH],
            "ops": operations,
        }
    ]


def test_case_b_plan_grounding_expands_materialization_and_validator_accepts(
    tmp_path,
):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir, task_description=TASK, expected_paths=[SOURCE_PATH]
    )
    source_operation = _semantic_operation(project_dir)
    plan = _plan(
        [
            source_operation,
            {"op": "append_file", "path": TEST_PATH, "content": "\n"},
        ]
    )

    result = ground_post_plan_source_materialization(
        plan,
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert result.ok, result.to_dict()
    assert result.grounded_paths == (TEST_PATH,)
    assert set(result.materialization.file_map()) == {SOURCE_PATH, TEST_PATH}
    assert result.materialization.file_map()[SOURCE_PATH] == (
        source_materialization.file_map()[SOURCE_PATH]
    )
    assert plan == json.loads(json.dumps(plan))

    validation = ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=TASK,
        execution_profile="implementation",
        project_dir=project_dir,
        source_materialization=result.materialization,
    )
    assert validation.accepted, validation.reasons


def test_already_materialized_target_is_not_reread_or_duplicated(tmp_path, monkeypatch):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir,
        task_description=TASK,
        expected_paths=[TEST_PATH],
        supporting_paths=[],
    )
    plan = _plan([{"op": "append_file", "path": TEST_PATH, "content": "\n"}])

    def unexpected_grounding(*args, **kwargs):
        raise AssertionError("already-materialized target was grounded again")

    monkeypatch.setattr(
        grounding_module, "materialize_planner_source_context", unexpected_grounding
    )
    result = grounding_module.ground_post_plan_source_materialization(
        plan,
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert result.ok, result.to_dict()
    assert result.grounded_paths == ()
    assert result.materialization == source_materialization
    assert len(result.materialization.files) == 1


def test_missing_existing_append_target_fails_closed(tmp_path):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir, task_description=TASK, expected_paths=[SOURCE_PATH]
    )
    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "append_file", "path": "tests/missing.py", "content": "\n"}]),
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert not result.ok
    assert result.failure_code == grounding_module.POST_PLAN_GROUNDING_MISSING


def test_traversal_and_absolute_paths_are_rejected_before_grounding(
    tmp_path, monkeypatch
):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir, task_description=TASK, expected_paths=[SOURCE_PATH]
    )

    def unexpected_grounding(*args, **kwargs):
        raise AssertionError("unsafe path reached source builder")

    monkeypatch.setattr(
        grounding_module, "materialize_planner_source_context", unexpected_grounding
    )
    for path in ("../secret.txt", "/tmp/secret.txt"):
        result = grounding_module.ground_post_plan_source_materialization(
            _plan([{"op": "append_file", "path": path, "content": "\n"}]),
            project_dir=project_dir,
            source_materialization=source_materialization,
        )
        assert not result.ok
        assert result.failure_code == grounding_module.POST_PLAN_GROUNDING_PATH_REJECTED


def test_protected_path_is_rejected_before_grounding(tmp_path, monkeypatch):
    project_dir = _workspace(tmp_path)
    (project_dir / ".agent").mkdir()
    (project_dir / ".agent" / "state.json").write_text("{}", encoding="utf-8")
    source_materialization = materialize_planner_source_context(
        project_dir, task_description=TASK, expected_paths=[SOURCE_PATH]
    )

    def unexpected_grounding(*args, **kwargs):
        raise AssertionError("protected path reached source builder")

    monkeypatch.setattr(
        grounding_module, "materialize_planner_source_context", unexpected_grounding
    )
    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "append_file", "path": ".agent/state.json", "content": "\n"}]),
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert not result.ok
    assert result.failure_code == grounding_module.POST_PLAN_GROUNDING_PROTECTED


def test_symlink_target_fails_closed_before_source_materialization(tmp_path):
    project_dir = _workspace(tmp_path)
    (project_dir / "tests" / "linked.py").symlink_to(project_dir / TEST_PATH)
    source_materialization = materialize_planner_source_context(
        project_dir, task_description=TASK, expected_paths=[SOURCE_PATH]
    )
    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "append_file", "path": "tests/linked.py", "content": "\n"}]),
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert not result.ok
    assert result.failure_code == grounding_module.POST_PLAN_GROUNDING_SYMLINK


def test_wrong_runtime_workspace_identity_fails_closed(tmp_path):
    runtime_one = _workspace(tmp_path / "runtime-one")
    runtime_two = _workspace(tmp_path / "runtime-two")
    source_materialization = materialize_planner_source_context(
        runtime_one, task_description=TASK, expected_paths=[SOURCE_PATH]
    )
    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "append_file", "path": TEST_PATH, "content": "\n"}]),
        project_dir=runtime_two,
        source_materialization=source_materialization,
    )

    assert not result.ok
    assert (
        result.failure_code == grounding_module.POST_PLAN_GROUNDING_INCOMPLETE_EVIDENCE
    )


def test_stale_append_target_fails_before_validator(tmp_path):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir, task_description=TASK, expected_paths=[TEST_PATH]
    )
    (project_dir / TEST_PATH).write_text(TEST + "# drift\n", encoding="utf-8")

    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "append_file", "path": TEST_PATH, "content": "\n"}]),
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert not result.ok
    assert result.failure_code == grounding_module.POST_PLAN_GROUNDING_VERSION_STALE


def test_stale_delete_target_fails_before_validator(tmp_path):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir, task_description=TASK, expected_paths=[TEST_PATH]
    )
    (project_dir / TEST_PATH).write_text(TEST + "# drift\n", encoding="utf-8")

    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "delete_file", "path": TEST_PATH}]),
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert not result.ok
    assert result.failure_code == grounding_module.POST_PLAN_GROUNDING_VERSION_STALE


def test_non_regular_target_fails_closed(tmp_path):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir,
        task_description=TASK,
        expected_paths=[SOURCE_PATH],
        supporting_paths=[],
    )
    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "append_file", "path": "tests", "content": "\n"}]),
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert not result.ok
    assert result.failure_code == grounding_module.POST_PLAN_GROUNDING_NOT_REGULAR


def test_grounding_capacity_does_not_evict_existing_records(tmp_path):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir,
        task_description=TASK,
        expected_paths=[SOURCE_PATH],
        supporting_paths=[],
    )
    bounded = replace(source_materialization, maximum_files=1)
    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "append_file", "path": TEST_PATH, "content": "\n"}]),
        project_dir=project_dir,
        source_materialization=bounded,
    )

    assert not result.ok
    assert result.failure_code == grounding_module.POST_PLAN_GROUNDING_CAPACITY_EXCEEDED
    assert tuple(item.relative_path for item in result.materialization.files) == (
        SOURCE_PATH,
    )


def test_expected_file_only_path_is_not_grounded(tmp_path):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir,
        task_description=TASK,
        expected_paths=[SOURCE_PATH],
        supporting_paths=[],
    )
    plan = _plan([])
    result = grounding_module.ground_post_plan_source_materialization(
        plan,
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert result.ok
    assert result.grounded_paths == ()
    assert TEST_PATH not in result.materialization.file_map()


def test_inspection_only_path_is_not_grounded(tmp_path):
    project_dir = _workspace(tmp_path)
    source_materialization = materialize_planner_source_context(
        project_dir,
        task_description=TASK,
        expected_paths=[SOURCE_PATH],
        supporting_paths=[],
    )
    plan = _plan([])
    plan[0]["commands"] = ["python3 tests/test_formatting.py"]
    result = grounding_module.ground_post_plan_source_materialization(
        plan,
        project_dir=project_dir,
        source_materialization=source_materialization,
    )

    assert result.ok
    assert result.grounded_paths == ()
    assert TEST_PATH not in result.materialization.file_map()


def test_runtime_workspace_is_the_only_grounding_source(tmp_path):
    product_root = _workspace(tmp_path / "product-root")
    runtime_workspace = _workspace(tmp_path / "runtime-workspace")
    (product_root / TEST_PATH).write_text("PRODUCT ROOT SENTINEL\n", encoding="utf-8")
    source_materialization = materialize_planner_source_context(
        runtime_workspace,
        task_description=TASK,
        expected_paths=[SOURCE_PATH],
        supporting_paths=[],
    )

    result = grounding_module.ground_post_plan_source_materialization(
        _plan([{"op": "append_file", "path": TEST_PATH, "content": "\n"}]),
        project_dir=runtime_workspace,
        source_materialization=source_materialization,
    )

    assert result.ok, result.to_dict()
    assert result.materialization.file_map()[TEST_PATH].content == TEST
    assert (product_root / TEST_PATH).read_text(encoding="utf-8") == (
        "PRODUCT ROOT SENTINEL\n"
    )

"""Provider-free PL16 factual observed-candidate inventory seam controls."""

from __future__ import annotations

import json

import pytest

from app.services.orchestration.planning.read_only_discovery import (
    DiscoveryObservation,
    DiscoveryRequest,
    materialize_observation_source_context,
    execute_discovery_request,
)
from app.services.orchestration.planning.semantic_target_inventory import (
    SemanticTargetContractError,
    build_semantic_target_inventory,
    normalize_provider_semantic_intents,
)
from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
    observed_candidate_paths,
)
from app.services.orchestration.validation.accepted_path_authority import (
    accepted_path_authority_from_verdict,
)
from app.services.orchestration.validation.path_authority import GrantClass, declare
from app.services.orchestration.validation.validator import ValidatorService


TASK = "Replace exact snippet `value = 1` in the relevant discovered source files."
SOURCE_A = "def value():\n    candidate_marker = True\n    value = 1\n"
SOURCE_B = "def value():\n    value = 1\n"


def _materialize(root, *, expected=(), supporting=("app/A.py", "app/B.py")):
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "app/A.py").write_text(SOURCE_A, encoding="utf-8")
    (root / "app/B.py").write_text(SOURCE_B, encoding="utf-8")
    return materialize_planner_source_context(
        root,
        task_description=TASK,
        expected_paths=expected,
        supporting_paths=supporting,
    )


def _semantic_plan(target_id: str, path: str = "app/A.py"):
    return [
        {
            "step_number": 1,
            "description": "Apply the requested bounded source change",
            "commands": [],
            "verification": "python -m py_compile app/A.py",
            "rollback": None,
            "expected_files": [path],
            "ops": [
                {
                    "op": "replace_in_file",
                    "path": path,
                    "target_id": target_id,
                    "new": "value = 2",
                }
            ],
        }
    ]


def test_observed_hit_paths_widen_inventory_without_promoting_expected_or_scope(
    tmp_path,
):
    materialization = _materialize(tmp_path)
    default_inventory = build_semantic_target_inventory(materialization)
    observed = execute_discovery_request(
        tmp_path,
        DiscoveryRequest(
            action="search_text", query="candidate_marker", paths=("app",)
        ),
    )

    assert observed.status == "completed"
    assert observed.paths == ("app",)
    assert tuple(hit.path for hit in observed.hits) == ("app/A.py",)
    assert observed.materialization_paths() == ("app/A.py",)
    assert default_inventory.handles == ()

    widened = build_semantic_target_inventory(
        materialization,
        additional_candidate_paths=observed.materialization_paths(),
    )
    assert [handle.path for handle in widened.handles] == ["app/A.py"]
    assert materialization.file_map()["app/A.py"].expected is False
    assert materialization.file_map()["app/B.py"].expected is False
    assert "target_id:" in materialization.to_prompt_block(
        provider_safe=True,
        additional_candidate_paths=observed.materialization_paths(),
    )
    assert "app/B.py\nstatus:" in materialization.to_prompt_block(
        provider_safe=True,
        additional_candidate_paths=observed.materialization_paths(),
    )
    assert build_semantic_target_inventory(materialization).handles == ()


def test_unobserved_supporting_path_and_hard_scope_stay_ineligible(tmp_path):
    materialization = _materialize(tmp_path)
    observed_paths = ("app/A.py",)

    inventory = build_semantic_target_inventory(
        materialization,
        additional_candidate_paths=observed_paths,
    )
    hard_scoped = build_semantic_target_inventory(
        materialization,
        task_scope=("app/A.py",),
        additional_candidate_paths=("app/A.py", "app/B.py"),
    )
    assert [handle.path for handle in inventory.handles] == ["app/A.py"]
    assert [handle.path for handle in hard_scoped.handles] == ["app/A.py"]
    assert build_semantic_target_inventory(materialization).handles == ()

    stop = DiscoveryObservation(action="stop", status="stopped")
    assert stop.materialization_paths() == ()
    failed = DiscoveryObservation(
        action="read_file", status="failed", paths=("app/A.py",)
    )
    assert observed_candidate_paths(failed) == ()
    assert (
        build_semantic_target_inventory(
            materialization,
            additional_candidate_paths=observed_candidate_paths(stop),
        ).handles
        == ()
    )


def test_read_file_carries_only_the_exact_observed_file(tmp_path):
    materialization = _materialize(tmp_path, supporting=())
    observation = DiscoveryObservation(
        action="read_file",
        status="completed",
        paths=("app/A.py",),
        content=SOURCE_A,
    )
    assert observation.materialization_paths() == ("app/A.py",)
    observed_materialization = materialize_observation_source_context(
        project_dir=tmp_path,
        prompt=TASK,
        planner_contract=None,
        observation=observation,
        materialize=materialize_planner_source_context,
    )
    inventory = build_semantic_target_inventory(
        observed_materialization,
        additional_candidate_paths=observation.materialization_paths(),
    )
    assert [handle.path for handle in inventory.handles] == ["app/A.py"]
    assert observed_materialization.file_map()["app/A.py"].expected is False
    assert materialization.files == ()


def test_multiple_handles_normalize_and_only_selected_path_gets_mutable_apa(
    tmp_path,
):
    materialization = _materialize(tmp_path)
    inventory = build_semantic_target_inventory(
        materialization,
        additional_candidate_paths=("app/A.py", "app/B.py"),
    )
    assert [handle.path for handle in inventory.handles] == ["app/A.py", "app/B.py"]
    assert len({handle.target_id for handle in inventory.handles}) == 2
    selected = inventory.handles[0]
    normalized = normalize_provider_semantic_intents(
        _semantic_plan(selected.target_id),
        inventory=inventory,
        project_dir=tmp_path,
        source_materialization=materialization,
    )
    assert "selector" in normalized[0]["ops"][0]

    with pytest.raises(SemanticTargetContractError) as unknown:
        normalize_provider_semantic_intents(
            _semantic_plan("tgt_unissued"),
            inventory=inventory,
            project_dir=tmp_path,
            source_materialization=materialization,
        )
    assert unknown.value.code == "unknown_target_id"

    other = inventory.handles[1]
    with pytest.raises(SemanticTargetContractError) as mismatch:
        normalize_provider_semantic_intents(
            _semantic_plan(selected.target_id, path=other.path),
            inventory=inventory,
            project_dir=tmp_path,
            source_materialization=materialization,
        )
    assert mismatch.value.code == "target_id_path_mismatch"

    outside = normalize_provider_semantic_intents(
        _semantic_plan(other.target_id, path=other.path),
        inventory=inventory,
        project_dir=tmp_path,
        source_materialization=materialization,
    )
    scope_verdict = ValidatorService.validate_plan(
        outside,
        output_text=json.dumps(outside),
        task_prompt="Only app/A.py may be modified. Replace exact snippet `value = 1`.",
        execution_profile="implementation",
        project_dir=tmp_path,
        source_materialization=materialization,
    )
    assert not scope_verdict.accepted
    assert any("task scope violation" in reason for reason in scope_verdict.reasons)

    assert json.dumps(normalized)
    verdict = ValidatorService.validate_plan(
        normalized,
        output_text=json.dumps(normalized),
        task_prompt=TASK,
        execution_profile="implementation",
        project_dir=tmp_path,
        source_materialization=materialization,
    )
    authority = accepted_path_authority_from_verdict(verdict)
    assert verdict.accepted, verdict.reasons
    assert authority is not None
    assert authority.authorizes(declare("app/A.py"), GrantClass.EXISTING_MUTABLE)
    assert not authority.authorizes(declare("app/B.py"), GrantClass.EXISTING_MUTABLE)


def test_stale_widened_handle_fails_against_current_inventory(tmp_path):
    materialization = _materialize(tmp_path)
    inventory = build_semantic_target_inventory(
        materialization,
        additional_candidate_paths=("app/A.py", "app/B.py"),
    )
    stale_id = inventory.handles[0].target_id
    (tmp_path / "app/A.py").write_text(SOURCE_A + "# V2\n", encoding="utf-8")
    current = _materialize(tmp_path)
    current_inventory = build_semantic_target_inventory(
        current,
        additional_candidate_paths=("app/A.py", "app/B.py"),
    )
    assert current_inventory.handles[0].target_id != stale_id
    with pytest.raises(SemanticTargetContractError) as stale:
        normalize_provider_semantic_intents(
            _semantic_plan(stale_id),
            inventory=current_inventory,
            project_dir=tmp_path,
            source_materialization=current,
        )
    assert stale.value.code == "unknown_target_id"

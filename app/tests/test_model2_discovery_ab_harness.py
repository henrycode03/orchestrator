"""Provider-free controls for the POST33-MODEL2 evaluation-only seam."""

from __future__ import annotations

import json

from scripts.evals import model2_discovery_ab as harness
from app.services.orchestration.planning.read_only_discovery import (
    execute_discovery_request,
    materialize_observation_source_context,
    parse_discovery_request,
)
from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
)


def test_model2_packets_reuse_production_prompt_and_isolate_profile_envelope():
    packet = harness._prompt_packet("T222")
    baseline = harness._wire_prompt(packet, harness.ARMS["A"])
    candidate = harness._wire_prompt(packet, harness.ARMS["B"])
    profile = harness._wire_prompt(packet, harness.ARMS["C"])

    assert baseline == packet["discovery_prompt"]
    assert candidate == baseline
    assert json.loads(profile)["body"] == baseline
    assert profile != baseline


def test_model2_ephemeral_identity_and_pl18_do_not_mutate_product_state(tmp_path):
    before = harness._product_state()
    before_config = harness._persistent_config_fingerprint()
    runtime_workspace = tmp_path / "runtime"
    runtime_workspace.mkdir()
    service, identity = harness._configure_ephemeral_service(
        harness.ARMS["B"], runtime_workspace
    )
    try:
        assert identity["agent_id"] == "orchestrator"
        assert identity["model"] == "ollama/qwen3-coder:30b"
        assert identity["tools"] == {"deny": ["*"]}
    finally:
        service.release_runtime_workspace_binding()
        service._evaluation_db.close()

    assert harness._persistent_config_fingerprint() == before_config
    assert harness._product_state() == before


def test_model2_response_injection_uses_production_replay_chain():
    request = parse_discovery_request(
        '{"action":"search_text","query":"rate","paths":["app/services/tasks/tool_tracking.py"]}'
    )
    observation = execute_discovery_request(harness.REPOSITORY_ROOT, request)
    materialization = materialize_observation_source_context(
        project_dir=harness.REPOSITORY_ROOT,
        prompt=harness.TASKS["T222"]["task"],
        planner_contract=None,
        observation=observation,
        materialize=materialize_planner_source_context,
        source_cache={},
    )
    assert observation.hits
    assert any(
        item.relative_path == harness.TASKS["T222"]["target_path"]
        for item in materialization.files
    )


def test_model2_order_budget_and_raw_response_artifacts_are_bounded():
    assert harness.PROVIDER_CALL_BUDGET == 9
    assert len(harness.CALL_ORDER) == 9
    assert len(set(harness.CALL_ORDER)) == 9
    for packet, arm in harness.CALL_ORDER:
        cell = next(
            path
            for path in (harness.EVIDENCE_ROOT / "cells").iterdir()
            if f"-{packet}-" in path.name
            and path.name.endswith(harness.ARMS[arm]["name"])
        )
        assert (cell / "raw-provider.stdout").exists()
        assert (cell / "raw-provider.stderr").exists()

"""Provider-free controls for the POST33-MODEL3 evaluation-only seam."""

from __future__ import annotations

import pytest

from scripts.evals import model3_discovery_generalization as harness
from app.services.orchestration.planning.read_only_discovery import (
    DiscoveryContractError,
    execute_discovery_request,
    materialize_observation_source_context,
    parse_discovery_request,
)
from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
)


def test_model3_packets_use_production_prompt_and_exclude_profile_control():
    t179 = harness._prompt_packet("T179")
    assert "app/services/observability/log_stream.py" in t179["task"]
    assert t179["discovery_prompt_bytes"] > len(t179["task"].encode())
    assert set(harness.ARMS) == {"A", "B"}
    assert all(arm != "C" for _, arm in harness.CALL_ORDER)


def test_model3_fixed_order_and_budget_are_bounded():
    assert harness.PROVIDER_CALL_BUDGET == 8
    assert len(harness.CALL_ORDER) == 8
    assert len(set(harness.CALL_ORDER)) == 8
    assert [packet for packet, _ in harness.CALL_ORDER] == [
        "T217",
        "T220",
        "T179",
        "T181",
        "T220",
        "T217",
        "T181",
        "T179",
    ]


def test_model3_production_replay_keeps_t181_creation_path_non_readable():
    packet = harness._prompt_packet("T181")
    request = parse_discovery_request(
        '{"action":"read_file","path":"app/services/observability/log_metadata.py"}'
    )
    with pytest.raises(DiscoveryContractError, match="discovery_path_missing"):
        execute_discovery_request(harness.REPOSITORY_ROOT, request)
    assert not (harness.REPOSITORY_ROOT / packet["creation_path"]).exists()


def test_model3_d1_failure_classification_preserves_envelope_distinction():
    assert (
        harness._d1_failure_cause(
            "prior conversation text", "discovery_output_not_json"
        )
        == "OPENCLAW_SESSION_CONTEXT_CONTAMINATION"
    )
    assert (
        harness._d1_failure_cause('{"action":"stop"}\n```', "discovery_output_not_json")
        == "MODEL_GENERATED_NONCANONICAL"
    )


def test_model3_exact_path_replay_uses_production_materialization():
    packet = harness._prompt_packet("T179")
    request = parse_discovery_request(
        '{"action":"read_file","path":"app/services/observability/log_stream.py"}'
    )
    observation = execute_discovery_request(harness.REPOSITORY_ROOT, request)
    materialization = materialize_observation_source_context(
        project_dir=harness.REPOSITORY_ROOT,
        prompt=packet["task"],
        planner_contract=None,
        observation=observation,
        materialize=materialize_planner_source_context,
        source_cache={},
    )
    assert observation.status == "completed"
    assert any(
        item.relative_path == packet["target_path"] for item in materialization.files
    )

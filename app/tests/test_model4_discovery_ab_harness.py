"""Provider-free controls for the POST33-MODEL4 evaluation-only seam."""

from scripts.evals import model4_discovery_ab as model4


def test_model4_has_exact_frozen_packets_and_balanced_order():
    assert tuple(model4.TASKS) == (
        "T222",
        "T218",
        "T214",
        "T217",
        "T220",
        "T179",
        "T181",
    )
    assert len(model4.CALL_ORDER) == 14
    assert model4.CALL_ORDER == (
        ("T222", "A"),
        ("T218", "B"),
        ("T217", "A"),
        ("T220", "B"),
        ("T214", "A"),
        ("T179", "B"),
        ("T181", "A"),
        ("T222", "B"),
        ("T218", "A"),
        ("T217", "B"),
        ("T220", "A"),
        ("T214", "B"),
        ("T179", "A"),
        ("T181", "B"),
    )
    assert all(
        sum(arm == arm_id for _, arm in model4.CALL_ORDER) == 7 for arm_id in ("A", "B")
    )
    assert set(model4.ARMS) == {"A", "B"}
    assert all(arm["profile"] == "openclaw_default" for arm in model4.ARMS.values())


def test_model4_scoring_contract_and_adoption_threshold_are_frozen():
    assert model4.SCORING_CONTRACT["discovery_success"].startswith("D1 PASS")
    threshold = model4.SCORING_CONTRACT["adoption_threshold"]
    assert threshold["candidate_successes_at_least"] == 4
    assert threshold["candidate_advantage_at_least"] == 2
    assert threshold["no_explicit_path_control_regression"] is True
    assert "qwen_compact_json" not in str(model4.ARMS)


def test_model4_c1_check_rejects_unexpected_hashes_without_repair():
    observed = model4._c1_check()
    assert observed["identity_match"] is True
    assert model4._c1_check({"hashes": observed["hashes"]})["identity_match"] is True
    mismatched = model4._c1_check({"hashes": {"openai-completions": "wrong"}})
    assert mismatched["identity_match"] is False
    assert "hash" in mismatched["failure_reason"]


def test_model4_prompt_packets_are_canonical_single_turn_inputs():
    packets = {
        packet_id: model4._prompt_packet(packet_id) for packet_id in model4.TASKS
    }
    assert all(packet["discovery_prompt_bytes"] > 0 for packet in packets.values())
    assert all(
        packet["discovery_prompt_hash"]
        == model4._sha256_text(packet["discovery_prompt"])
        for packet in packets.values()
    )
    assert (
        model4.TASKS["T181"]["creation_path"]
        == "app/services/observability/log_metadata.py"
    )

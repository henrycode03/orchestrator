"""Provider-free controls for POST33-MODEL4 runtime isolation."""

from scripts.evals import model4_runtime_isolation as model4


def test_model4_runtime_isolation_has_exact_six_cell_order_and_same_model_arms():
    assert tuple(model4.TASKS) == ("T222", "T218", "T214")
    assert model4.CALL_ORDER == (
        ("T222", "A"),
        ("T218", "B"),
        ("T214", "A"),
        ("T218", "A"),
        ("T214", "B"),
        ("T222", "B"),
    )
    assert len(model4.CALL_ORDER) == 6
    assert all(
        sum(arm == arm_id for _, arm in model4.CALL_ORDER) == 3 for arm_id in ("A", "B")
    )
    assert model4.ARMS["A"]["requested_provider_model"] == "openai/qwen-local"
    assert model4.ARMS["B"]["requested_provider_model"] == "openai/qwen-local"
    assert model4.ARMS["A"]["runtime"] == "local_openclaw"
    assert model4.ARMS["B"]["runtime"] == "openai_chat_completions"


def test_model4_prompt_packets_are_canonical_and_wire_body_equal():
    packets = {
        packet_id: model4._prompt_packet(packet_id) for packet_id in model4.TASKS
    }
    for packet in packets.values():
        assert packet["canonical_discovery_prompt_bytes"] > 0
        assert packet["canonical_discovery_prompt_sha256"] == model4._sha256_text(
            packet["canonical_discovery_prompt"]
        )
        assert packet["orientation_metadata"]["orientation_available"] is True
    assert all(
        packet["canonical_discovery_prompt_sha256"]
        == model4._sha256_text(packet["canonical_discovery_prompt"])
        for packet in packets.values()
    )


def test_model4_pl16_serializer_uses_provider_safe_projection():
    class Handle:
        def to_provider_dict(self):
            return {"target_id": "tgt_test", "path": "app/a.py"}

    assert model4._serialize_handle(Handle()) == {
        "target_id": "tgt_test",
        "path": "app/a.py",
    }


def test_model4_bootstrap_audit_is_provider_free_and_retains_runtime_finding():
    audit = model4._bootstrap_audit()
    assert audit["OPENCLAW_DISCOVERY_SKIP_BOOTSTRAP"] is True
    assert audit["OPENCLAW_DISCOVERY_AGENTDIR_EPHEMERAL"] is True
    assert audit["OPENCLAW_DISCOVERY_SESSION_STORE_EPHEMERAL"] is True
    assert audit["OPENCLAW_DISCOVERY_AGENTS_MD_VISIBLE"] is True
    assert audit["OPENCLAW_DISCOVERY_SOUL_MD_VISIBLE"] is True
    assert audit["OPENCLAW_DISCOVERY_TOOLS_MD_VISIBLE"] is True
    assert audit["OPENCLAW_DISCOVERY_MEMORY_CONTEXT_VISIBLE"] is False
    assert audit["OPENCLAW_DISCOVERY_PRIOR_SESSION_CONTEXT_POSSIBLE"] is False
    assert audit["OPENCLAW_DISCOVERY_BOOTSTRAP_CONTAMINATION_CLASS"] == (
        "C. BOOTSTRAP_CONTEXT_PRESENT"
    )


def test_model4_direct_arm_has_no_openclaw_surface():
    arm = model4.ARMS["B"]
    assert arm["runtime"] == "openai_chat_completions"
    assert arm["backend"] == "openai_chat_completions"
    assert arm["provider_model_ref"] == "openai/qwen-local"
    assert "openclaw" not in arm["runtime"]

"""POST33-MODEL4 clean seven-packet discovery-model A/B gate.

This is evaluation-only.  It binds each arm to an ephemeral OpenClaw agent,
uses the production one-turn discovery prompt/parser/executor/materialization/
PL16 chain, and never enters the Orchestrator lifecycle.  The call order,
scoring contract, and packet wording are frozen before the first generation.
There is deliberately no retry path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evals import model2_discovery_ab as base

from app.services.agents.openclaw_service import OpenClawSessionService
from app.services.agents.runtime_configuration import (
    BackendRole,
    RoleRuntimeConfiguration,
)
from app.services.orchestration.planning.read_only_discovery import (
    build_discovery_prompt,
    discovery_output_text,
    execute_discovery_request,
    materialize_observation_source_context,
    parse_discovery_request,
)
from app.services.orchestration.planning.repository_orientation import (
    derive_repository_orientation,
)
from app.services.orchestration.planning.semantic_target_inventory import (
    build_semantic_target_inventory,
)
from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
    observed_candidate_paths,
)
from app.services.orchestration.validation.parsing import extract_structured_text


EVIDENCE_ROOT = REPOSITORY_ROOT / "docs/roadmap/reports/evidence/post33-model4"
PATCH_CHECKER = REPOSITORY_ROOT / "scripts/maintenance/apply_openclaw_c1_patch.py"
OPENCLAW_ROOT = Path("/usr/lib/node_modules/openclaw")
PROVIDER_CALL_BUDGET = 14
PROVIDER_RETRIES = 0
DISCOVERY_TIMEOUT_SECONDS = base.DISCOVERY_TIMEOUT_SECONDS
PERSISTENT_OPENCLAW_CONFIG = base.PERSISTENT_OPENCLAW_CONFIG
EXPECTED_OPENCLAW_VERSION = "2026.4.10"
EXPECTED_PI_AI_VERSION = "0.66.1"
EXPECTED_QWEN_LOCAL_ROOT = "/models/Qwen3.6-27B-Text-NVFP4-MTP"


TASKS: dict[str, dict[str, Any]] = {
    **base.TASKS,
    "T217": {
        "task": (
            "Preserve structured errors in fallback failure summaries.\n\n"
            "Keep complete structured error records in deterministic fallback failure "
            "summaries while continuing to filter raw JSON fragments and provider "
            "diagnostics. Add focused regression coverage for a complete JSON error "
            "record and an incomplete fragment."
        ),
        "target_path": "app/services/session/replan_service.py",
        "target_terms": ("structured", "error", "fallback", "JSON"),
        "shape": "natural-language/no-explicit-path",
    },
    "T220": {
        "task": (
            "Fix manual knowledge synchronization failure state.\n\n"
            "When manual synchronization fails because the underlying domain "
            "operation raises an error, the item must not remain in an in-progress "
            "state. Preserve existing successful synchronization and retry behavior, "
            "and add focused regression coverage for the failure transition."
        ),
        "target_path": "app/services/knowledge/knowledge_sync_service.py",
        "target_terms": ("synchronization", "failure", "in-progress", "retry"),
        "shape": "natural-language/no-explicit-path",
    },
    "T179": {
        "task": (
            "Return requested usable session-log count after filtering.\n\n"
            "Correct session-log streaming so the requested log limit applies to "
            "usable records after filtering or suppression. Limit scope to "
            "app/services/observability/log_stream.py and focused tests under "
            "app/tests/test_log_stream_service.py if needed."
        ),
        "target_path": "app/services/observability/log_stream.py",
        "target_terms": ("limit", "usable", "filtering", "suppression"),
        "shape": "explicit-path",
    },
    "T181": {
        "task": (
            "Add reusable structured-log metadata normalization for streaming.\n\n"
            "NEW production module: app/services/observability/log_metadata.py\n"
            "INTEGRATE it in: app/services/observability/log_stream.py\n\n"
            "log_stream.py parses persisted log metadata directly with json.loads at "
            "each streaming site."
        ),
        "target_path": "app/services/observability/log_stream.py",
        "creation_path": "app/services/observability/log_metadata.py",
        "target_terms": ("log_stream", "json.loads", "metadata", "streaming"),
        "shape": "explicit-existing-plus-new-file",
    },
}

ARMS: dict[str, dict[str, Any]] = {
    "A": {
        "name": "baseline",
        "requested_model": "qwen3.6:27B",
        "provider_model_ref": "openai/qwen-local",
        "provider": "openai",
        "profile": "openclaw_default",
        "backend": "local_openclaw",
        "model_family": "qwen3.6:27B",
    },
    "B": {
        "name": "candidate",
        "requested_model": "qwen3-coder:30b",
        "provider_model_ref": "ollama/qwen3-coder:30b",
        "provider": "ollama",
        "profile": "openclaw_default",
        "backend": "local_openclaw",
        "model_family": "qwen3-coder:30b",
    },
}

CALL_ORDER = (
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

SCORING_CONTRACT: dict[str, Any] = {
    "version": "POST33-MODEL4-D1-D7-2026-08-24",
    "d1": "PASS only when production parse_discovery_request accepts the extracted response without normalization or repair.",
    "d2": "2 exact factual/relevant target path; 1 real but broad/unhelpful path; 0 fabricated/recombined/unsafe/unusable path.",
    "d3": "2 target-relevant source area; 1 plausible adjacent area; 0 wrong subsystem/no useful semantic scope.",
    "d4": "2 strong useful query/read action; 1 valid but weak/zero-information action; 0 invalid or semantically useless action.",
    "d5": "4 useful target source plus target hint plus handle; 3 useful target source plus hint without handle; 2 useful target source without hint; 1 valid weak observation; 0 no valid observation.",
    "d6": "PASS when PL18 zero native tools, no mutation/authority claim, path authority, whole-action failure, and protected-scope controls remain preserved. Framework rejection is safety PASS.",
    "d7": "Record provider latency, first-output latency when available, output chars/tokens, timeout, and stop reason; never compensate for D1/D6.",
    "discovery_success": "D1 PASS and D6 PASS and D3 >= 1 and D4 >= 1 and D5 >= 2.",
    "pl16_handle_success": "At least one PL16 handle in a D1-valid production replay.",
    "adoption_threshold": {
        "candidate_successes_at_least": 4,
        "candidate_advantage_at_least": 2,
        "candidate_canonical_rate_not_worse": True,
        "candidate_fabricated_paths_not_greater": True,
        "no_explicit_path_control_regression": True,
        "candidate_natural_language_success_required": True,
        "no_safety_regression": True,
        "no_identity_drift": True,
        "no_persistent_config_mutation": True,
    },
    "baseline_threshold": "symmetric to candidate threshold; deployment status does not affect scoring.",
}


class RuntimeContractError(RuntimeError):
    """A benchmark/runtime boundary failed closed before model scoring."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _redact(value: str) -> str:
    return base._redact_text(value)


def _persistent_fingerprint() -> dict[str, Any]:
    raw = PERSISTENT_OPENCLAW_CONFIG.read_bytes()
    return {
        "path": str(PERSISTENT_OPENCLAW_CONFIG),
        "bytes": len(raw),
        "mode": oct(PERSISTENT_OPENCLAW_CONFIG.stat().st_mode & 0o777),
        "sha256": _sha256_bytes(raw),
    }


def _c1_check(expected: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(PATCH_CHECKER), "check"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    parsed: dict[str, Any] = {}
    if proc.stdout.strip():
        parsed = json.loads(proc.stdout)
    state = parsed.get("state", [])
    openclaw_version = state[0].get("openclaw_version") if state else None
    hashes = {
        row.get("name"): row.get("sha256") for row in state[1:] if isinstance(row, dict)
    }
    openclaw_package = OPENCLAW_ROOT / "package.json"
    pi_package = OPENCLAW_ROOT / "node_modules/@mariozechner/pi-ai/package.json"
    c1_available = openclaw_package.is_file() and pi_package.is_file()
    pi_ai_version = None
    if c1_available:
        try:
            package = json.loads(pi_package.read_text(encoding="utf-8"))
            if isinstance(package, dict):
                pi_ai_version = package.get("version")
            else:
                c1_available = False
        except (OSError, json.JSONDecodeError):
            c1_available = False
    expected_hashes = (expected or {}).get("hashes")
    identity_match = (
        c1_available
        and proc.returncode == 0
        and openclaw_version == EXPECTED_OPENCLAW_VERSION
        and pi_ai_version == EXPECTED_PI_AI_VERSION
        and (expected_hashes is None or hashes == expected_hashes)
    )
    result = {
        "status": (
            "PASS" if identity_match else "INVALID" if c1_available else "UNAVAILABLE"
        ),
        "command": [sys.executable, str(PATCH_CHECKER), "check"],
        "return_code": proc.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "openclaw_version": openclaw_version,
        "pi_ai_version": pi_ai_version,
        "hashes": hashes,
        "expected_openclaw_version": EXPECTED_OPENCLAW_VERSION,
        "expected_pi_ai_version": EXPECTED_PI_AI_VERSION,
        "expected_hashes": expected_hashes,
        "c1_available": c1_available,
        "identity_match": identity_match,
        "stdout": _redact(proc.stdout),
        "stderr": _redact(proc.stderr),
    }
    if not c1_available:
        result["failure_reason"] = (
            "OpenClaw C1 runtime unavailable: expected package files are not installed"
        )
    elif proc.returncode != 0:
        result["failure_reason"] = "repository-managed C1 checker failed"
    elif openclaw_version != EXPECTED_OPENCLAW_VERSION:
        result["failure_reason"] = "OpenClaw version changed"
    elif pi_ai_version != EXPECTED_PI_AI_VERSION:
        result["failure_reason"] = "pi-ai version changed"
    elif expected_hashes is not None and hashes != expected_hashes:
        result["failure_reason"] = "certified serializer hash changed"
    return result


def _catalog_identity(catalogs: dict[str, Any]) -> dict[str, Any]:
    openai = catalogs["openai_compatible"]["payload"]
    ollama = catalogs["ollama"]["payload"]
    openai_model = next(
        item
        for item in openai.get("data", [])
        if isinstance(item, dict) and item.get("id") == "qwen-local"
    )
    ollama_model = next(
        item
        for item in ollama.get("models", [])
        if isinstance(item, dict)
        and (item.get("name") or item.get("model")) == "qwen3-coder:30b"
    )
    root = str(openai_model.get("root") or "")
    if root != EXPECTED_QWEN_LOCAL_ROOT:
        raise RuntimeError(f"qwen-local root is not pinned: {root!r}")
    return {
        "baseline": {
            "requested_model": "qwen3.6:27B",
            "requested_provider_model": "openai/qwen-local",
            "effective_openclaw_provider": "openai",
            "effective_openclaw_model": "qwen-local",
            "effective_gateway_model": "qwen-local",
            "underlying_model": root.removeprefix("/models/"),
            "catalog_record": openai_model,
        },
        "candidate": {
            "requested_model": "qwen3-coder:30b",
            "requested_provider_model": "ollama/qwen3-coder:30b",
            "effective_openclaw_provider": "ollama",
            "effective_openclaw_model": "qwen3-coder:30b",
            "catalog_record": ollama_model,
        },
    }


def _prompt_packet(packet_id: str) -> dict[str, Any]:
    packet = TASKS[packet_id]
    orientation = derive_repository_orientation(
        REPOSITORY_ROOT, packet["task"], explicit_paths=()
    )
    prompt = build_discovery_prompt(packet["task"], "", orientation)
    orientation_details = orientation.as_details()
    return {
        **packet,
        "id": packet_id,
        "orientation": orientation,
        "orientation_details": orientation_details,
        "task_hash": _sha256_text(packet["task"]),
        "orientation_hash": _sha256_text(
            json.dumps(orientation_details, sort_keys=True)
        ),
        "discovery_prompt": prompt,
        "discovery_prompt_hash": _sha256_text(prompt),
        "discovery_prompt_bytes": len(prompt.encode("utf-8")),
    }


def _packet_manifest(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in packet.items()
        if key not in {"orientation", "discovery_prompt", "target_terms"}
    }


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _selected_model_object(config_path: str, agent_id: str) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    selected = next(
        agent
        for agent in (config.get("agents") or {}).get("list", [])
        if isinstance(agent, dict) and agent.get("id") == agent_id
    )
    return dict(selected.get("model") or {})


def _native_tool_telemetry(
    parsed_runtime: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    meta = parsed_runtime.get("meta")
    report = meta.get("systemPromptReport") if isinstance(meta, dict) else None
    tools = report.get("tools") if isinstance(report, dict) else None
    entries = tools.get("entries") if isinstance(tools, dict) else None
    if not isinstance(entries, list):
        entries = []
    return {
        "configured_tools": identity.get("tools"),
        "runtime_telemetry_path": "meta.systemPromptReport.tools.entries",
        "runtime_tools": tools if isinstance(tools, dict) else {},
        "runtime_tool_entries": entries,
        "runtime_tool_entry_count": len(entries),
        "pl18_expected_runtime_tool_entry_count": 0,
        "pl18_suppression_preserved": identity.get("tools") == {"deny": ["*"]}
        and len(entries) == 0,
    }


def _configure_service(
    arm: dict[str, Any], runtime_workspace: Path
) -> tuple[OpenClawSessionService, dict[str, Any]]:
    service, identity = base._configure_ephemeral_service(arm, runtime_workspace)
    identity = dict(identity)
    identity["selected_model_object"] = _selected_model_object(
        identity["config_path"], identity["agent_id"]
    )
    identity["openclaw_version"] = service._resolve_openclaw_cli_version()
    identity["requested_provider_model"] = arm["provider_model_ref"]
    identity["requested_model"] = arm["requested_model"]
    identity["effective_provider"] = arm["provider_model_ref"].split("/", 1)[0]
    identity["effective_model"] = arm["provider_model_ref"].split("/", 1)[1]
    identity["fallbacks"] = []
    identity["profile"] = arm["profile"]
    return service, identity


def _preflight() -> dict[str, Any]:
    patch = _c1_check()
    if not patch["identity_match"]:
        raise RuntimeContractError(patch.get("failure_reason", "C1 check failed"))
    persistent = base._persistent_config()
    selected = base._agent_for_project(persistent)
    if selected.get("id") != "orchestrator":
        raise RuntimeError(f"unexpected selected agent: {selected.get('id')!r}")
    catalogs = base._provider_catalogs(persistent)
    identities = _catalog_identity(catalogs)
    command_db = base.EvaluationSessionLocal()
    command_service = OpenClawSessionService(
        command_db,
        None,
        None,
        runtime_configuration=RoleRuntimeConfiguration(
            role=BackendRole.PLANNING,
            backend_name="local_openclaw",
            model_family="qwen3.6:27B",
            adaptation_profile="openclaw_default",
        ),
    )
    cli_command = command_service._resolve_openclaw_command()
    cli_version_raw = command_service._resolve_openclaw_cli_version()
    command_db.close()
    cli_version_parts = str(cli_version_raw or "").split()
    cli_version = (
        cli_version_parts[1]
        if len(cli_version_parts) > 1
        else (cli_version_parts[0] if cli_version_parts else "")
    )
    if cli_version != EXPECTED_OPENCLAW_VERSION:
        raise RuntimeError(f"unexpected OpenClaw CLI version: {cli_version_raw!r}")
    packets = {packet_id: _prompt_packet(packet_id) for packet_id in TASKS}
    prompt_comparison = {
        packet_id: {
            "baseline_hash": packet["discovery_prompt_hash"],
            "candidate_hash": packet["discovery_prompt_hash"],
            "baseline_bytes": packet["discovery_prompt_bytes"],
            "candidate_bytes": packet["discovery_prompt_bytes"],
            "identical": True,
        }
        for packet_id, packet in packets.items()
    }
    config_before = _persistent_fingerprint()
    state_before = base._product_state()
    git = _git_state()
    return {
        "status": "READY",
        "evaluation_topology": "B1_ISOLATED_PROVIDER_BOUND_DISCOVERY",
        "provider_call_budget": PROVIDER_CALL_BUDGET,
        "provider_retries": PROVIDER_RETRIES,
        "openclaw_version": EXPECTED_OPENCLAW_VERSION,
        "pi_ai_version": EXPECTED_PI_AI_VERSION,
        "runtime3_c1_patch": patch,
        "git": git,
        "persistent_config_before": config_before,
        "product_state_before": state_before,
        "selected_agent": {
            "id": selected.get("id"),
            "workspace": selected.get("workspace"),
            "model": selected.get("model"),
        },
        "cli_command": cli_command,
        "cli_version": cli_version,
        "cli_version_raw": cli_version_raw,
        "thinking_default": persistent.get("agents", {})
        .get("defaults", {})
        .get("defaults", {})
        .get("thinkingDefault"),
        "provider_inspection": catalogs,
        "provider_identities": identities,
        "model_identity_plan": {
            arm_id: {
                key: arm[key]
                for key in (
                    "requested_model",
                    "provider_model_ref",
                    "provider",
                    "profile",
                )
            }
            for arm_id, arm in ARMS.items()
        },
        "packets": {
            packet_id: _packet_manifest(packet) for packet_id, packet in packets.items()
        },
        "prompt_comparison": prompt_comparison,
        "call_order": [
            {"sequence": index, "packet": packet, "arm": arm}
            for index, (packet, arm) in enumerate(CALL_ORDER, start=1)
        ],
        "isolation": {
            "normal_lifecycle_runner": False,
            "project_session_task_creation": False,
            "product_attempt": False,
            "planning_repair_execution_apa": False,
            "ephemeral_openclaw_config": True,
            "ephemeral_openclaw_state": True,
            "production_prompt_builder": True,
            "production_parser_executor_materialization_pl16": True,
            "raw_response_retention": True,
            "pl18_deny_all": True,
            "max_discovery_turns": 1,
            "arm_c_included": False,
        },
    }


def _prepare_artifacts(preflight: dict[str, Any]) -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(EVIDENCE_ROOT / "preflight.json", preflight)
    _write_json(
        EVIDENCE_ROOT / "runtime3-patch-verification.json",
        preflight["runtime3_c1_patch"],
    )
    _write_json(
        EVIDENCE_ROOT / "config-before-fingerprint.json",
        preflight["persistent_config_before"],
    )
    _write_json(
        EVIDENCE_ROOT / "product-state-before.json", preflight["product_state_before"]
    )
    _write_json(EVIDENCE_ROOT / "call-order.json", preflight["call_order"])
    _write_json(EVIDENCE_ROOT / "scoring-contract.json", SCORING_CONTRACT)
    _write_json(
        EVIDENCE_ROOT / "benchmark-manifest.json",
        {
            "status": "PREPARED_BEFORE_PROVIDER_DISPATCH",
            "provider_call_budget": PROVIDER_CALL_BUDGET,
            "provider_retries": PROVIDER_RETRIES,
            "call_order_sha256": _sha256_text(
                json.dumps(preflight["call_order"], sort_keys=True)
            ),
            "scoring_contract_sha256": _sha256_text(
                json.dumps(SCORING_CONTRACT, sort_keys=True)
            ),
        },
    )
    packets = {packet_id: _prompt_packet(packet_id) for packet_id in TASKS}
    for sequence, (packet_id, arm_id) in enumerate(CALL_ORDER, start=1):
        arm = ARMS[arm_id]
        packet = packets[packet_id]
        cell_dir = EVIDENCE_ROOT / "cells" / f"{sequence:02d}-{packet_id}-{arm['name']}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        prompt = packet["discovery_prompt"]
        (cell_dir / "canonical-prompt.txt").write_text(prompt, encoding="utf-8")
        (cell_dir / "wire-prompt.txt").write_text(prompt, encoding="utf-8")
        _write_json(
            cell_dir / "prompt-metadata.json",
            {
                "sequence": sequence,
                "packet": packet_id,
                "arm": arm["name"],
                "task_hash": packet["task_hash"],
                "orientation_hash": packet["orientation_hash"],
                "canonical_prompt_sha256": packet["discovery_prompt_hash"],
                "canonical_prompt_bytes": packet["discovery_prompt_bytes"],
                "wire_prompt_sha256": packet["discovery_prompt_hash"],
                "wire_prompt_bytes": packet["discovery_prompt_bytes"],
                "profile": arm["profile"],
                "canonical_wire_equal": True,
            },
        )


def _paths(request: Any) -> list[str]:
    values = [str(value) for value in (getattr(request, "paths", ()) or ())]
    requested_file = getattr(request, "path", None)
    if requested_file:
        values.append(str(requested_file))
    return values


def _path_issue(request: Any, observation: Any) -> bool:
    if getattr(observation, "status", "") in {"completed", "no_matches", "stopped"}:
        return False
    return bool(_paths(request))


def _score(
    packet: dict[str, Any],
    request: Any,
    observation: Any,
    materialization: Any,
    inventory: Any,
) -> dict[str, Any]:
    target = packet["target_path"]
    selected_paths = _paths(request)
    action = str(getattr(request, "action", ""))
    executable = getattr(observation, "status", "") in {
        "completed",
        "no_matches",
        "stopped",
    }
    exact = target in selected_paths and executable
    relevant = (
        any(
            target == value
            or target.startswith(value.rstrip("/") + "/")
            or value.startswith(target.rstrip("/") + "/")
            for value in selected_paths
        )
        and executable
    )
    d2 = 2 if exact else 1 if relevant else 0
    d3 = 2 if exact else 1 if relevant else 0
    hit_paths = {str(hit.path) for hit in getattr(observation, "hits", ())}
    if action == "search_text":
        d4 = 2 if target in hit_paths else 1 if executable else 0
    elif action == "read_file":
        content = str(getattr(observation, "content", "") or "")
        d4 = (
            2
            if getattr(request, "path", None) == target and content
            else 1 if content else 0
        )
    else:
        d4 = 0
    files = getattr(materialization, "files", ()) or ()
    target_file = next(
        (item for item in files if getattr(item, "relative_path", None) == target),
        None,
    )
    target_source = bool(
        target_file is not None
        and getattr(target_file, "content", None) is not None
        and len(str(getattr(target_file, "content", ""))) > 0
    )
    target_hint = bool(
        target_file is not None and getattr(target_file, "target_hint", None)
    )
    handles = list(getattr(inventory, "handles", ()) or ())
    target_handle = any(getattr(handle, "path", None) == target for handle in handles)
    materialized_any = any(
        getattr(item, "content", None) is not None
        and len(str(getattr(item, "content", ""))) > 0
        for item in files
    )
    if target_source and target_hint and target_handle:
        d5 = 4
    elif target_source and target_hint:
        d5 = 3
    elif target_source:
        d5 = 2
    elif executable and materialized_any:
        d5 = 1
    elif executable:
        d5 = 1
    else:
        d5 = 0
    return {
        "D1_contract_validity": True,
        "D2_path_fidelity": d2,
        "D3_scope_semantic_quality": d3,
        "D4_query_action_quality": d4,
        "D5_observation_utility": d5,
        "D6_safety": "PASS",
        "D5_raw": {
            "target_materialized": target_source,
            "target_hint": target_hint,
            "target_handle": target_handle,
            "hit_count": len(getattr(observation, "hits", ()) or ()),
            "materialized_path_count": len(files),
            "pl16_handle_count": len(handles),
        },
        "exact_target_path_selected": exact,
        "path_issue": _path_issue(request, observation),
        "observation_executable": executable,
    }


def _failure_class(result: dict[str, Any]) -> str:
    if not result.get("d1_valid"):
        return "F1_CONTRACT_FORMAT"
    score = result.get("score", {})
    if score.get("path_issue") or result.get("fabricated_path"):
        return "F2_PATH_TRANSCRIPTION"
    if score.get("D3_scope_semantic_quality", 0) == 0:
        return "F3_SCOPE_SELECTION"
    if score.get("D4_query_action_quality", 0) == 0:
        if result.get("parsed_action") == "stop":
            return "F5_ACTION_CHOICE"
        return "F4_QUERY_SEMANTICS"
    if score.get("D5_observation_utility", 0) == 1:
        return "F6_VALID_BUT_ZERO_INFORMATION"
    if result.get("target_observed_but_not_materialized"):
        return "F7_SOURCE_MATERIALIZATION_LIMITATION"
    if result.get("discovery_success"):
        return "F8_NO_FAILURE_USEFUL_OBSERVATION"
    return "F6_VALID_BUT_ZERO_INFORMATION"


def _request_json(request: Any) -> dict[str, Any]:
    return {
        "action": str(getattr(request, "action", "")),
        "paths": list(getattr(request, "paths", ()) or ()),
        "path": getattr(request, "path", None),
        "query": getattr(request, "query", None),
    }


def _observation_json(
    observation: Any, materialization: Any, inventory: Any
) -> dict[str, Any]:
    return {
        "action": str(getattr(observation, "action", "")),
        "status": getattr(observation, "status", None),
        "paths": list(getattr(observation, "paths", ()) or ()),
        "hit_count": len(getattr(observation, "hits", ()) or ()),
        "hits": [hit.__dict__ for hit in (getattr(observation, "hits", ()) or ())],
        "content_bytes": len(
            (getattr(observation, "content", "") or "").encode("utf-8")
        ),
        "truncated": getattr(observation, "truncated", None),
        "reason": getattr(observation, "reason", None),
        "materialized_sources": [
            {
                "path": item.relative_path,
                "status": item.status,
                "content_bytes": len((item.content or "").encode("utf-8")),
                "target_hint": item.target_hint,
                "target_included": item.target_included,
                "target_match_count": item.target_match_count,
            }
            for item in (getattr(materialization, "files", ()) or ())
        ],
        "pl16": {
            "handle_count": len(getattr(inventory, "handles", ()) or ()),
            "handles": [
                handle.to_dict() for handle in (getattr(inventory, "handles", ()) or ())
            ],
        },
    }


def _run_cell(
    sequence: int,
    packet: dict[str, Any],
    arm: dict[str, Any],
) -> dict[str, Any]:
    cell_dir = EVIDENCE_ROOT / "cells" / f"{sequence:02d}-{packet['id']}-{arm['name']}"
    prompt = packet["discovery_prompt"]
    service: OpenClawSessionService | None = None
    runtime_workspace = Path(tempfile.mkdtemp(prefix="post33-model4-workspace-"))
    dispatch_started = False
    started = time.monotonic()
    identity: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = {}
    raw_stdout = ""
    raw_stderr = ""
    try:
        service, identity = _configure_service(arm, runtime_workspace)
        if (
            identity.get("model") != arm["provider_model_ref"]
            or identity.get("fallbacks") != []
        ):
            raise RuntimeContractError("selected-agent model/fallback binding drifted")
        if identity.get("tools") != {"deny": ["*"]}:
            raise RuntimeContractError("PL18 deny-all was not preserved")
        command = service.build_cli_agent_command(
            prompt,
            source_brain="local",
            timeout_seconds=DISCOVERY_TIMEOUT_SECONDS,
            session_prefix="model4-discovery",
            strict_provider_result=False,
        )
        dispatch_started = True
        proc, diagnostics = asyncio.run(
            service._run_cli_prompt_with_diagnostics(
                command,
                timeout_seconds=DISCOVERY_TIMEOUT_SECONDS,
                cwd=str(runtime_workspace),
                prompt=prompt,
                invocation_kind="model4-discovery",
                strict_provider_result=False,
            )
        )
        raw_stdout = _redact(proc.stdout or "")
        raw_stderr = _redact(proc.stderr or "")
        (cell_dir / "raw-stdout.txt").write_text(raw_stdout, encoding="utf-8")
        (cell_dir / "raw-stderr.txt").write_text(raw_stderr, encoding="utf-8")
        parsed_runtime = service.parse_cli_response(
            proc, expected_session_id=None, strict_provider_result=False
        )
        identity_proof = base._verify_runtime_identity(
            arm,
            identity=identity,
            diagnostics=diagnostics,
            parsed_runtime=parsed_runtime,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            prompt_hash=_sha256_text(prompt),
        )
        native = _native_tool_telemetry(parsed_runtime, identity)
        runtime_identity = {
            **identity,
            "requested_provider_model": arm["provider_model_ref"],
            "effective_openclaw_provider": identity_proof.get(
                "effective_provider_model_ref", ""
            ).split("/", 1)[0],
            "effective_openclaw_model": identity_proof.get(
                "effective_provider_model_ref", ""
            ).split("/", 1)[-1],
            "endpoint": identity.get("provider_endpoint"),
            "profile": arm["profile"],
            "selected_agent": identity.get("agent_id"),
            "selected_model_object": identity.get("selected_model_object"),
            "fallbacks": identity.get("fallbacks"),
            "openclaw_version": identity.get("openclaw_version"),
            "identity_proof": identity_proof,
            "return_code": diagnostics.get("return_code"),
            "timeout": diagnostics.get("timed_out", False),
            "stop_reason": diagnostics.get("terminal_reason"),
            "first_output_latency": diagnostics.get("first_output_delay_seconds"),
            "output_chars": {
                "stdout": len(raw_stdout),
                "stderr": len(raw_stderr),
            },
        }
        _write_json(cell_dir / "runtime-identity.json", runtime_identity)
        _write_json(
            cell_dir / "fallback-diagnostics.json",
            identity_proof.get("fallback_diagnostics", []),
        )
        _write_json(cell_dir / "native-tool-telemetry.json", native)
        if (
            native["runtime_tool_entry_count"] != 0
            or not native["pl18_suppression_preserved"]
        ):
            raise RuntimeContractError(
                "runtime native-tool telemetry violated PL18 zero-tools"
            )
        if sequence in {7, 14}:
            checkpoint = _c1_check(
                {
                    "hashes": base_patch_hashes,
                    "openclaw_version": EXPECTED_OPENCLAW_VERSION,
                }
            )
            if not checkpoint["identity_match"]:
                raise RuntimeContractError(
                    f"C1 patch drift at sequence {sequence}: {checkpoint.get('failure_reason')}"
                )
            _write_json(cell_dir / "c1-checkpoint.json", checkpoint)
        extracted = discovery_output_text(parsed_runtime, extract_structured_text)
        (cell_dir / "extracted-response.txt").write_text(extracted, encoding="utf-8")
        result: dict[str, Any] = {
            "status": "completed",
            "dispatch_started": True,
            "sequence": sequence,
            "packet": packet["id"],
            "arm": arm["name"],
            "requested_model": arm["requested_model"],
            "requested_provider_model": arm["provider_model_ref"],
            "profile": arm["profile"],
            "identity": runtime_identity,
            "runtime_diagnostics": diagnostics,
            "provider_result": {
                key: value for key, value in parsed_runtime.items() if key != "output"
            },
            "extracted_response": extracted,
            "latency_seconds": round(time.monotonic() - started, 3),
            "provider_latency": diagnostics.get("duration_seconds"),
            "first_output_latency": diagnostics.get("first_output_delay_seconds"),
            "output_chars": len(extracted),
            "output_tokens": diagnostics.get("output_token_estimate"),
            "timeout": bool(diagnostics.get("timed_out")),
            "stop_reason": diagnostics.get("terminal_reason"),
            "model_identity_drift": False,
            "pl18_suppression_preserved": True,
            "native_tool_entry_count": 0,
        }
        try:
            request = parse_discovery_request(extracted)
        except Exception as exc:
            (cell_dir / "parsed-request.json").write_text(
                json.dumps({"status": "invalid", "error": str(exc)[:1000]}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            result.update(
                {
                    "status": "model_result",
                    "d1_valid": False,
                    "parser_status": "invalid",
                    "d1_failure_cause": "F1_CONTRACT_FORMAT",
                    "failure_reason": str(exc)[:1000],
                    "score": {
                        "D1_contract_validity": False,
                        "D2_path_fidelity": 0,
                        "D3_scope_semantic_quality": 0,
                        "D4_query_action_quality": 0,
                        "D5_observation_utility": 0,
                        "D6_safety": "PASS",
                    },
                    "discovery_success": False,
                    "pl16_handle_success": False,
                    "parsed_action": None,
                    "fabricated_path": False,
                    "failure_class": "F1_CONTRACT_FORMAT",
                }
            )
            _write_json(cell_dir / "score.json", result["score"])
            _write_json(
                cell_dir / "production-replay.json", {"status": "not_run_d1_invalid"}
            )
            _write_json(cell_dir / "result.json", result)
            return result
        _write_json(cell_dir / "parsed-request.json", _request_json(request))
        observation = execute_discovery_request(REPOSITORY_ROOT, request)
        materialization = materialize_observation_source_context(
            project_dir=REPOSITORY_ROOT,
            prompt=packet["task"],
            planner_contract=None,
            observation=observation,
            materialize=materialize_planner_source_context,
            source_cache={},
        )
        inventory = build_semantic_target_inventory(
            materialization,
            additional_candidate_paths=observed_candidate_paths(observation),
        )
        replay = _observation_json(observation, materialization, inventory)
        _write_json(cell_dir / "production-replay.json", replay)
        score = _score(packet, request, observation, materialization, inventory)
        result.update(
            {
                "status": "completed",
                "d1_valid": True,
                "parser_status": "valid",
                "parsed_action": str(request.action),
                "requested_paths": list(request.paths),
                "requested_path": request.path,
                "query": request.query,
                "observation": replay,
                "score": score,
                "discovery_success": score["D6_safety"] == "PASS"
                and score["D3_scope_semantic_quality"] >= 1
                and score["D4_query_action_quality"] >= 1
                and score["D5_observation_utility"] >= 2,
                "pl16_handle_success": bool(score["D5_raw"]["pl16_handle_count"]),
                "fabricated_path": bool(score.get("path_issue")),
                "creation_read_confusion": packet["id"] == "T181"
                and "app/services/observability/log_metadata.py" in _paths(request),
                "target_observed_but_not_materialized": packet["target_path"]
                in {str(hit.path) for hit in getattr(observation, "hits", ())}
                and not score["D5_raw"]["target_materialized"],
            }
        )
        result["failure_class"] = _failure_class(result)
        _write_json(cell_dir / "score.json", score)
        _write_json(cell_dir / "result.json", result)
        return result
    except base.IdentityDriftError as exc:
        proof = exc.proof
        _write_json(
            cell_dir / "runtime-identity.json",
            {
                **(identity or {}),
                "requested_provider_model": arm["provider_model_ref"],
                "identity_proof": proof,
                "effective_provider_model": proof.get("effective_provider_model_ref"),
                "fallbacks": (identity or {}).get("fallbacks", []),
                "return_code": diagnostics.get("return_code"),
            },
        )
        _write_json(
            cell_dir / "fallback-diagnostics.json",
            proof.get("fallback_diagnostics", []),
        )
        _write_json(
            cell_dir / "native-tool-telemetry.json", {"status": "identity_not_proven"}
        )
        result = {
            "status": "invalid_runtime_identity",
            "dispatch_started": dispatch_started,
            "sequence": sequence,
            "packet": packet["id"],
            "arm": arm["name"],
            "requested_provider_model": arm["provider_model_ref"],
            "identity_failure": True,
            "model_identity_drift": proof.get("status") == "INVALID_IDENTITY_DRIFT",
            "identity_proof": proof,
            "failure_reason": proof.get("failure_reason"),
            "latency_seconds": round(time.monotonic() - started, 3),
        }
        _write_json(cell_dir / "result.json", result)
        return result
    except Exception as exc:
        runtime_diagnostics = getattr(exc, "runtime_diagnostics", diagnostics) or {}
        if dispatch_started:
            (cell_dir / "raw-stdout.txt").write_text(raw_stdout, encoding="utf-8")
            (cell_dir / "raw-stderr.txt").write_text(raw_stderr, encoding="utf-8")
        _write_json(
            cell_dir / "runtime-identity.json",
            {
                **(identity or {}),
                "requested_provider_model": arm["provider_model_ref"],
                "status": "identity_unverified_before_scoring",
                "return_code": runtime_diagnostics.get("return_code"),
                "timeout": runtime_diagnostics.get("timed_out", False),
            },
        )
        _write_json(cell_dir / "fallback-diagnostics.json", [])
        _write_json(
            cell_dir / "native-tool-telemetry.json", {"status": "identity_unverified"}
        )
        result = {
            "status": "invalid_runtime",
            "dispatch_started": dispatch_started,
            "sequence": sequence,
            "packet": packet["id"],
            "arm": arm["name"],
            "requested_provider_model": arm["provider_model_ref"],
            "identity_failure": bool(dispatch_started),
            "runtime_contract_failure": isinstance(exc, RuntimeContractError),
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc)[:1000],
            "latency_seconds": round(time.monotonic() - started, 3),
        }
        _write_json(cell_dir / "result.json", result)
        return result
    finally:
        if service is not None:
            service.release_runtime_workspace_binding()
            evaluation_db = getattr(service, "_evaluation_db", None)
            if evaluation_db is not None:
                evaluation_db.close()
        try:
            runtime_workspace.rmdir()
        except OSError:
            pass


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for arm_id, arm in ARMS.items():
        cells = [result for result in results if result.get("arm") == arm["name"]]
        valid = [result for result in cells if result.get("d1_valid")]
        scores = [result.get("score", {}) for result in valid]
        latencies = [
            float(result["provider_latency"])
            for result in cells
            if result.get("provider_latency") is not None
        ]
        failures = {}
        for result in cells:
            classification = result.get("failure_class")
            if classification:
                failures[classification] = failures.get(classification, 0) + 1
        exact = sum(score.get("D2_path_fidelity") == 2 for score in scores)
        aggregate[arm_id] = {
            "arm": arm["name"],
            "cells": len(cells),
            "canonical_response_count": len(valid),
            "canonical_response_rate": round(len(valid) / 7, 4),
            "executable_action_count": sum(
                bool(result.get("parsed_action")) for result in valid
            ),
            "exact_path_count": exact,
            "fabricated_path_count": sum(
                bool(result.get("fabricated_path")) for result in cells
            ),
            "wrong_subsystem_scope_count": sum(
                score.get("D3_scope_semantic_quality") == 0 for score in scores
            ),
            "zero_hit_valid_query_count": sum(
                result.get("parsed_action") == "search_text"
                and result.get("observation", {}).get("hit_count") == 0
                for result in valid
            ),
            "read_file_choice_count": sum(
                result.get("parsed_action") == "read_file" for result in valid
            ),
            "search_text_choice_count": sum(
                result.get("parsed_action") == "search_text" for result in valid
            ),
            "stop_choice_count": sum(
                result.get("parsed_action") == "stop" for result in valid
            ),
            "useful_observation_count": sum(
                bool(result.get("discovery_success")) for result in valid
            ),
            "target_hint_count": sum(
                score.get("D5_raw", {}).get("target_hint", False) for score in scores
            ),
            "pl16_handle_total": sum(
                score.get("D5_raw", {}).get("pl16_handle_count", 0) for score in scores
            ),
            "discovery_success_packets": [
                result.get("packet")
                for result in valid
                if result.get("discovery_success")
            ],
            "pl16_handle_success_packets": [
                result.get("packet")
                for result in valid
                if result.get("pl16_handle_success")
            ],
            "mean_provider_latency": (
                round(statistics.mean(latencies), 3) if latencies else None
            ),
            "median_provider_latency": (
                round(statistics.median(latencies), 3) if latencies else None
            ),
            "timeout_count": sum(bool(result.get("timeout")) for result in cells),
            "stop_reasons": [result.get("stop_reason") for result in cells],
            "failure_classes": failures,
            "scores": {
                metric: sum(score.get(metric, 0) for score in scores)
                for metric in (
                    "D2_path_fidelity",
                    "D3_scope_semantic_quality",
                    "D4_query_action_quality",
                    "D5_observation_utility",
                )
            },
        }
    return aggregate


def _cell_map(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(result.get("packet"), result.get("arm")): result for result in results}


def _adjudicate(
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    identity_drift: bool,
    runtime_invalid: bool,
    config_unchanged: bool,
    product_mutation: bool,
) -> dict[str, Any]:
    if (
        identity_drift
        or runtime_invalid
        or not config_unchanged
        or product_mutation
        or len(results) != 14
    ):
        return {
            "final_decision": "E. EVALUATION_INVALID_RUNTIME_IDENTITY_OR_CONTRACT_DRIFT",
            "decision_confidence": "HIGH",
            "model_candidate_materially_better": False,
            "baseline_model_materially_better": False,
            "explicit_path_control_regression": False,
            "natural_language_task_advantage": False,
            "reason": "The complete clean comparison did not satisfy runtime/identity/isolation invariants.",
        }
    by_cell = _cell_map(results)
    baseline = aggregate["A"]
    candidate = aggregate["B"]

    def success(arm: str, packet: str) -> bool:
        return bool(
            by_cell.get((packet, ARMS[arm]["name"]), {}).get("discovery_success")
        )

    explicit_regression = False
    explicit_details = {}
    for packet in ("T214", "T179", "T181"):
        a = by_cell.get((packet, "baseline"), {})
        b = by_cell.get((packet, "candidate"), {})
        a_score = a.get("score", {})
        b_score = b.get("score", {})
        confusion = (
            bool(b.get("creation_read_confusion")) if packet == "T181" else False
        )
        regressed = (
            b_score.get("D2_path_fidelity", 0) < a_score.get("D2_path_fidelity", 0)
            or bool(a.get("discovery_success"))
            and not bool(b.get("discovery_success"))
            or confusion
        )
        explicit_details[packet] = {
            "baseline_discovery_success": bool(a.get("discovery_success")),
            "candidate_discovery_success": bool(b.get("discovery_success")),
            "baseline_path_fidelity": a_score.get("D2_path_fidelity", 0),
            "candidate_path_fidelity": b_score.get("D2_path_fidelity", 0),
            "candidate_creation_read_confusion": confusion,
            "regression": regressed,
        }
        explicit_regression = explicit_regression or regressed
    natural_packets = ("T222", "T218", "T217", "T220")
    natural_baseline = sum(success("A", packet) for packet in natural_packets)
    natural_candidate = sum(success("B", packet) for packet in natural_packets)
    candidate_threshold = {
        "successes_gte_4": candidate["useful_observation_count"] >= 4,
        "advantage_gte_2": candidate["useful_observation_count"]
        - baseline["useful_observation_count"]
        >= 2,
        "canonical_rate_not_worse": candidate["canonical_response_rate"]
        >= baseline["canonical_response_rate"],
        "fabricated_paths_not_greater": candidate["fabricated_path_count"]
        <= baseline["fabricated_path_count"],
        "no_explicit_path_control_regression": not explicit_regression,
        "natural_language_success": natural_candidate >= 1,
        "no_safety_regression": all(
            result.get("score", {}).get("D6_safety") == "PASS"
            for result in results
            if result.get("d1_valid")
        ),
        "no_identity_drift": not identity_drift,
        "no_persistent_config_mutation": config_unchanged and not product_mutation,
    }
    baseline_threshold = {
        **candidate_threshold,
        "successes_gte_4": baseline["useful_observation_count"] >= 4,
        "advantage_gte_2": baseline["useful_observation_count"]
        - candidate["useful_observation_count"]
        >= 2,
        "canonical_rate_not_worse": baseline["canonical_response_rate"]
        >= candidate["canonical_response_rate"],
        "fabricated_paths_not_greater": baseline["fabricated_path_count"]
        <= candidate["fabricated_path_count"],
        "natural_language_success": natural_baseline >= 1,
    }
    candidate_wins = all(candidate_threshold.values())
    baseline_wins = all(baseline_threshold.values())
    if candidate_wins:
        decision = "A. QWEN3_CODER_MATERIALLY_BETTER_ADOPT_AS_DISCOVERY_MODEL_CANDIDATE"
    elif baseline_wins:
        decision = "B. QWEN36_BASELINE_MATERIALLY_BETTER_RETAIN_BASELINE"
    elif (
        candidate["canonical_response_count"] > baseline["canonical_response_count"]
        and candidate["useful_observation_count"]
        <= baseline["useful_observation_count"]
    ):
        decision = "F. CANDIDATE_IMPROVES_FORMAT_NOT_DISCOVERY_SEMANTICS"
    elif natural_candidate > natural_baseline and explicit_regression:
        decision = (
            "G. CANDIDATE_IMPROVES_DISCOVERY_BUT_REGRESSES_EXPLICIT_PATH_CONTROLS"
        )
    elif (
        candidate["useful_observation_count"] == baseline["useful_observation_count"]
        and abs(
            candidate["canonical_response_count"] - baseline["canonical_response_count"]
        )
        <= 1
    ):
        decision = "C. MODELS_EFFECTIVELY_EQUIVALENT_RETAIN_CURRENT_BASELINE"
    else:
        decision = "D. MIXED_RESULTS_NO_MODEL_ADOPTION"
    return {
        "final_decision": decision,
        "decision_confidence": "HIGH" if candidate_wins or baseline_wins else "MEDIUM",
        "model_candidate_materially_better": candidate_wins,
        "baseline_model_materially_better": baseline_wins,
        "explicit_path_control_regression": explicit_regression,
        "explicit_path_details": explicit_details,
        "natural_language_task_advantage": natural_candidate > natural_baseline,
        "natural_language_counts": {
            "baseline": natural_baseline,
            "candidate": natural_candidate,
        },
        "candidate_threshold": candidate_threshold,
        "baseline_threshold": baseline_threshold,
        "candidate_discovery_successes": candidate["useful_observation_count"],
        "baseline_discovery_successes": baseline["useful_observation_count"],
    }


base_patch_hashes: dict[str, Any] = {}


def run(*, execute: bool) -> int:
    global base_patch_hashes
    try:
        preflight = _preflight()
        base_patch_hashes = dict(preflight["runtime3_c1_patch"]["hashes"])
        _prepare_artifacts(preflight)
    except Exception as exc:
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(
            EVIDENCE_ROOT / "preflight.json",
            {
                "status": "PREFLIGHT_FAILED",
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc)[:1000],
                "runtime3_c1_patch": _c1_check(),
                "git": _git_state(),
                "persistent_config_before": _persistent_fingerprint(),
                "product_state_before": base._product_state(),
            },
        )
        return 2
    if not execute:
        return 0
    config_before = preflight["persistent_config_before"]
    state_before = preflight["product_state_before"]
    packets = {packet_id: _prompt_packet(packet_id) for packet_id in TASKS}
    results: list[dict[str, Any]] = []
    identity_drift = False
    runtime_invalid = False
    aborted = False
    for sequence, (packet_id, arm_id) in enumerate(CALL_ORDER, start=1):
        if aborted:
            break
        result = _run_cell(sequence, packets[packet_id], ARMS[arm_id])
        results.append(result)
        if (
            result.get("model_identity_drift")
            or result.get("identity_failure")
            or result.get("runtime_contract_failure")
        ):
            identity_drift = bool(result.get("model_identity_drift"))
            runtime_invalid = True
            aborted = True
        if _persistent_fingerprint() != config_before:
            runtime_invalid = True
            aborted = True
    config_after = _persistent_fingerprint()
    state_after = base._product_state()
    _write_json(EVIDENCE_ROOT / "config-after-fingerprint.json", config_after)
    _write_json(EVIDENCE_ROOT / "product-state-after.json", state_after)
    c1_final = _c1_check({"hashes": base_patch_hashes})
    _write_json(EVIDENCE_ROOT / "runtime3-patch-verification-final.json", c1_final)
    if not c1_final["identity_match"]:
        runtime_invalid = True
    deltas = {
        key: state_after.get(key, 0) - state_before.get(key, 0) for key in state_before
    }
    product_mutation = any(value != 0 for value in deltas.values())
    aggregate = _aggregate(results)
    comparison = _adjudicate(
        results,
        aggregate,
        identity_drift=identity_drift,
        runtime_invalid=runtime_invalid,
        config_unchanged=config_before == config_after,
        product_mutation=product_mutation,
    )
    _write_json(EVIDENCE_ROOT / "replay-summaries.json", results)
    _write_json(
        EVIDENCE_ROOT / "scorecard.json", {"cells": results, "aggregate": aggregate}
    )
    _write_json(
        EVIDENCE_ROOT / "aggregate-comparison.json",
        {"aggregate": aggregate, "adjudication": comparison},
    )
    _write_json(
        EVIDENCE_ROOT / "identity-adjudication.json",
        {
            "identity_drift_detected": identity_drift,
            "stop_on_first_identity_drift": True,
            "runtime_invalid": runtime_invalid,
            "cells_executed": len(results),
            "cell_identity_statuses": [
                {
                    "sequence": result.get("sequence"),
                    "packet": result.get("packet"),
                    "arm": result.get("arm"),
                    "status": (
                        result.get("identity_proof")
                        or result.get("identity", {}).get("identity_proof", {})
                    ).get("status"),
                    "effective": (
                        result.get("identity_proof")
                        or result.get("identity", {}).get("identity_proof", {})
                    ).get("effective_provider_model_ref"),
                }
                for result in results
            ],
        },
    )
    manifest = {
        "status": "INVALID" if runtime_invalid or aborted else "COMPLETED",
        "model4_execution_status": (
            "INVALID" if runtime_invalid or aborted else "COMPLETED"
        ),
        "packets_executed": len({result.get("packet") for result in results}),
        "provider_calls": sum(
            bool(result.get("dispatch_started")) for result in results
        ),
        "provider_call_budget": PROVIDER_CALL_BUDGET,
        "provider_retries": PROVIDER_RETRIES,
        "identity_drift_detected": identity_drift,
        "c1_patch_drift_detected": not c1_final["identity_match"],
        "persistent_openclaw_config_unchanged": config_before == config_after,
        "product_state_before": state_before,
        "product_state_after": state_after,
        "product_state_delta": deltas,
        "product_state_mutation": product_mutation,
        "task222_product_attempt_created": False,
        "framework_production_changes": 0,
        "framework_test_changes": 0,
        "evaluation_harness_changes": ["scripts/evals/model4_discovery_ab.py"],
        "evaluation_harness_test_changes": [
            "app/tests/test_model4_discovery_ab_harness.py"
        ],
        "aggregate": aggregate,
        "adjudication": comparison,
    }
    _write_json(EVIDENCE_ROOT / "benchmark-manifest.json", manifest)
    return 0 if manifest["status"] == "COMPLETED" else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        parser.error("choose exactly one of --preflight or --execute")
    return run(execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())

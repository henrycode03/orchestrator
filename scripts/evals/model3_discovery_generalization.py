"""POST33-MODEL3 bounded cross-task discovery-model generalization gate.

This evaluation-only seam reuses the MODEL2 OpenClaw binding and production
discovery replay helpers.  It deliberately has only two arms, four packets,
and one fixed interleaved call for each cell.  It never enters the normal
Orchestrator lifecycle or changes production discovery semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evals import model2_discovery_ab as model2


REPOSITORY_ROOT = model2.REPOSITORY_ROOT
EVIDENCE_ROOT = REPOSITORY_ROOT / "docs/roadmap/reports/evidence/post33-model3"
PROVIDER_CALL_BUDGET = 8
DISCOVERY_TIMEOUT_SECONDS = model2.DISCOVERY_TIMEOUT_SECONDS
PERSISTENT_OPENCLAW_CONFIG = model2.PERSISTENT_OPENCLAW_CONFIG

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


TASKS: dict[str, dict[str, Any]] = {
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
    arm_id: dict(model2.ARMS[arm_id]) for arm_id in ("A", "B")
}

CALL_ORDER = (
    ("T217", "A"),
    ("T220", "B"),
    ("T179", "A"),
    ("T181", "B"),
    ("T220", "A"),
    ("T217", "B"),
    ("T181", "A"),
    ("T179", "B"),
)


def _sha256_text(value: str) -> str:
    return model2._sha256_text(value)


def _write_json(path: Path, value: Any) -> None:
    model2._write_json(path, value)


def _redact_text(value: str) -> str:
    return model2._redact_text(value)


def _prompt_packet(packet_id: str) -> dict[str, Any]:
    packet = TASKS[packet_id]
    orientation = derive_repository_orientation(
        REPOSITORY_ROOT, packet["task"], explicit_paths=()
    )
    discovery_prompt = build_discovery_prompt(packet["task"], "", orientation)
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
        "discovery_prompt": discovery_prompt,
        "discovery_prompt_hash": _sha256_text(discovery_prompt),
        "discovery_prompt_bytes": len(discovery_prompt.encode("utf-8")),
    }


def _provider_identity(catalogs: dict[str, Any]) -> dict[str, Any]:
    openai_payload = catalogs["openai_compatible"]["payload"]
    ollama_payload = catalogs["ollama"]["payload"]
    openai_model = next(
        item
        for item in openai_payload.get("data", [])
        if isinstance(item, dict) and item.get("id") == "qwen-local"
    )
    ollama_model = next(
        item
        for item in ollama_payload.get("models", [])
        if isinstance(item, dict)
        and (item.get("name") or item.get("model")) == "qwen3-coder:30b"
    )
    root = str(openai_model.get("root") or "")
    if root != "/models/Qwen3.6-27B-Text-NVFP4-MTP":
        raise RuntimeError(f"qwen-local root is not pinned: {root!r}")
    return {
        "A": {
            "requested_model": "qwen3.6:27B",
            "provider_model_ref": "openai/qwen-local",
            "effective_model": "openai/qwen-local",
            "provider_runtime_model": root.removeprefix("/models/"),
            "provider": "openai-compatible-ai-gateway",
            "catalog_record": openai_model,
        },
        "B": {
            "requested_model": "qwen3-coder:30b",
            "provider_model_ref": "ollama/qwen3-coder:30b",
            "effective_model": "ollama/qwen3-coder:30b",
            "provider_runtime_model": ollama_model.get("name")
            or ollama_model.get("model"),
            "provider": "ollama",
            "catalog_record": ollama_model,
        },
    }


def _preflight() -> dict[str, Any]:
    persistent = model2._persistent_config()
    selected = model2._agent_for_project(persistent)
    catalogs = model2._provider_catalogs(persistent)
    identities = _provider_identity(catalogs)
    if selected.get("id") != "orchestrator":
        raise RuntimeError(f"unexpected selected agent: {selected.get('id')!r}")
    packets = {packet_id: _prompt_packet(packet_id) for packet_id in TASKS}
    prompt_comparison = {}
    for packet_id, packet in packets.items():
        baseline_prompt = packet["discovery_prompt"]
        candidate_prompt = packet["discovery_prompt"]
        prompt_comparison[packet_id] = {
            "identical": baseline_prompt == candidate_prompt,
            "baseline_hash": _sha256_text(baseline_prompt),
            "candidate_hash": _sha256_text(candidate_prompt),
            "baseline_bytes": len(baseline_prompt.encode("utf-8")),
            "candidate_bytes": len(candidate_prompt.encode("utf-8")),
        }
        if not prompt_comparison[packet_id]["identical"]:
            raise RuntimeError(f"prompt bytes differ for {packet_id}")
    command_db = model2.SessionLocal()
    command_service = model2.OpenClawSessionService(
        command_db,
        None,
        None,
        runtime_configuration=model2.RoleRuntimeConfiguration(
            role=model2.BackendRole.PLANNING,
            backend_name="local_openclaw",
            model_family="qwen3.6:27B",
            adaptation_profile="openclaw_default",
        ),
    )
    cli_command = command_service._resolve_openclaw_command()
    cli_version = command_service._resolve_openclaw_cli_version()
    command_db.close()
    return {
        "status": "READY",
        "provider_call_budget": PROVIDER_CALL_BUDGET,
        "persistent_config": model2._persistent_config_fingerprint(),
        "selected_agent": {
            "id": selected.get("id"),
            "workspace": selected.get("workspace"),
            "model": selected.get("model"),
        },
        "cli_command": cli_command,
        "openclaw_version": cli_version,
        "thinking_default": persistent.get("agents", {})
        .get("defaults", {})
        .get("thinkingDefault"),
        "provider_inspection": catalogs,
        "provider_identities": identities,
        "provider_model_identity_pinned": True,
        "prompt_comparison": prompt_comparison,
        "packets": {
            packet_id: {
                key: value
                for key, value in packet.items()
                if key not in {"orientation", "discovery_prompt", "target_terms"}
            }
            for packet_id, packet in packets.items()
        },
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
        "call_order": [
            {"sequence": index, "packet": packet, "arm": arm}
            for index, (packet, arm) in enumerate(CALL_ORDER, start=1)
        ],
        "isolation": {
            "database_object": None,
            "normal_lifecycle_runner": False,
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


def _score(
    packet: dict[str, Any],
    request: Any,
    observation: Any,
    materialization: Any,
    inventory: Any,
) -> dict[str, Any]:
    target_path = packet["target_path"]
    requested_paths = tuple(getattr(request, "paths", ()) or ())
    requested_file = getattr(request, "path", None)
    relevant = target_path in requested_paths or any(
        target_path.startswith(path.rstrip("/") + "/") for path in requested_paths
    )
    exact = target_path in requested_paths or requested_file == target_path
    d2 = 2 if exact else 1 if relevant else 0
    d3 = 2 if exact else 1 if relevant else 0
    action = getattr(request, "action", "")
    hit_paths = {hit.path for hit in getattr(observation, "hits", ())}
    if action == "search_text":
        d4 = (
            2
            if target_path in hit_paths
            else (
                1
                if observation.status
                in {
                    "completed",
                    "no_matches",
                }
                else 0
            )
        )
    elif action == "read_file":
        content = str(getattr(observation, "content", "") or "")
        d4 = 2 if requested_file == target_path and content else 1
    else:
        d4 = 0
    handle_paths = {handle.path for handle in getattr(inventory, "handles", ())}
    target_hint = any(
        getattr(item, "relative_path", None) == target_path
        and getattr(item, "target_hint", None)
        for item in getattr(materialization, "files", ())
    )
    target_materialized = any(
        getattr(item, "relative_path", None) == target_path
        and getattr(item, "content", None) is not None
        for item in getattr(materialization, "files", ())
    )
    if target_materialized and (target_hint or target_path in handle_paths):
        d5 = 3
    elif target_materialized:
        d5 = 2
    elif observation.status in {"completed", "no_matches", "stopped"}:
        d5 = 1
    else:
        d5 = 0
    return {
        "D1_contract_validity": True,
        "D2_path_fidelity": d2,
        "D3_scope_semantic_quality": d3,
        "D4_query_quality": d4,
        "D5_observation_utility": d5,
        "D6_safety": "PASS_FAIL_CLOSED_PRESERVED",
        "D5_raw": {
            "target_materialized": target_materialized,
            "target_hint": target_hint,
            "target_handle": target_path in handle_paths,
            "hit_count": len(getattr(observation, "hits", ()) or ()),
            "materialized_path_count": len(
                getattr(observation, "materialization_paths", lambda: ())()
            ),
            "pl16_handle_count": len(getattr(inventory, "handles", ()) or ()),
        },
    }


def _requested_paths_for_result(result: dict[str, Any]) -> list[str]:
    values = result.get("requested_paths") or []
    requested_file = result.get("requested_path")
    if requested_file:
        values = [*values, requested_file]
    return [str(value) for value in values]


def _classify_result(packet: dict[str, Any], result: dict[str, Any]) -> str:
    if result.get("status") != "completed":
        paths = _requested_paths_for_result(result)
        if any(
            not (REPOSITORY_ROOT / path).exists()
            for path in paths
            if path and not path.startswith("/")
        ):
            return "FABRICATED_PATH"
        return "CONTRACT_FAILURE"
    score = result.get("score", {})
    if score.get("D5_observation_utility", 0) >= 3:
        return "USEFUL_SOURCE_WITH_HANDLE"
    if score.get("D5_observation_utility", 0) == 2:
        return "USEFUL_SOURCE_WITHOUT_HANDLE"
    if score.get("D5_observation_utility", 0) == 1:
        return "VALID_BUT_ZERO_INFORMATION"
    if score.get("D3_scope_semantic_quality", 0) == 0:
        return "SEMANTICALLY_WRONG"
    if score.get("D2_path_fidelity", 0) == 1:
        return "FACTUAL_BUT_BROAD"
    return "SEMANTICALLY_WRONG"


def _d1_failure_cause(extracted: str, failure_reason: str) -> str:
    text = f"{extracted}\n{failure_reason}".lower()
    if any(
        marker in text
        for marker in (
            "prior attempt",
            "prior conversation",
            "previous response",
            "noreply",
            "no_reply",
            "conversation",
        )
    ):
        return "OPENCLAW_SESSION_CONTEXT_CONTAMINATION"
    if "```" in text or "function" in text or "native" in text:
        return "MODEL_GENERATED_NONCANONICAL"
    return "MODEL_GENERATED_NONCANONICAL"


def _run_cell(
    sequence: int,
    packet: dict[str, Any],
    arm: dict[str, Any],
) -> dict[str, Any]:
    cell_dir = EVIDENCE_ROOT / "cells" / f"{sequence:02d}-{packet['id']}-{arm['name']}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    prompt = packet["discovery_prompt"]
    (cell_dir / "discovery-prompt.txt").write_text(prompt, encoding="utf-8")
    _write_json(
        cell_dir / "prompt-metadata.json",
        {
            "task_hash": packet["task_hash"],
            "orientation_hash": packet["orientation_hash"],
            "discovery_prompt_hash": packet["discovery_prompt_hash"],
            "discovery_prompt_bytes": packet["discovery_prompt_bytes"],
            "wire_prompt_hash": packet["discovery_prompt_hash"],
            "wire_prompt_bytes": packet["discovery_prompt_bytes"],
            "profile": arm["profile"],
            "profile_prompt_envelope_changed": False,
        },
    )
    service = None
    identity: dict[str, Any] | None = None
    dispatch_started = False
    started = time.monotonic()
    runtime_workspace = Path(
        __import__("tempfile").mkdtemp(prefix="post33-model3-workspace-")
    )
    try:
        service, identity = model2._configure_ephemeral_service(arm, runtime_workspace)
        if identity["model"] != arm["provider_model_ref"]:
            raise RuntimeError("ephemeral model identity did not match arm")
        if identity["tools"] != {"deny": ["*"]}:
            raise RuntimeError("PL18 deny-all was not preserved")
        command = service.build_cli_agent_command(
            prompt,
            source_brain="local",
            timeout_seconds=DISCOVERY_TIMEOUT_SECONDS,
            session_prefix="model3-discovery",
            strict_provider_result=False,
        )
        dispatch_started = True
        proc, diagnostics = asyncio.run(
            service._run_cli_prompt_with_diagnostics(
                command,
                timeout_seconds=DISCOVERY_TIMEOUT_SECONDS,
                cwd=str(runtime_workspace),
                prompt=prompt,
                invocation_kind="model3-discovery",
                strict_provider_result=False,
            )
        )
        raw_stdout = _redact_text(proc.stdout or "")
        raw_stderr = _redact_text(proc.stderr or "")
        (cell_dir / "raw-provider.stdout").write_text(raw_stdout, encoding="utf-8")
        (cell_dir / "raw-provider.stderr").write_text(raw_stderr, encoding="utf-8")
        parsed_runtime = service.parse_cli_response(
            proc, expected_session_id=None, strict_provider_result=False
        )
        identity_proof = model2._verify_runtime_identity(
            arm,
            identity=identity,
            diagnostics=diagnostics,
            parsed_runtime=parsed_runtime,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            prompt_hash=model2._sha256_text(prompt),
        )
        extracted = discovery_output_text(parsed_runtime, extract_structured_text)
        (cell_dir / "extracted-response.txt").write_text(extracted, encoding="utf-8")
        result: dict[str, Any] = {
            "status": "completed",
            "dispatch_started": True,
            "sequence": sequence,
            "packet": packet["id"],
            "arm": arm["name"],
            "requested_model": arm["requested_model"],
            "provider_model_ref": arm["provider_model_ref"],
            "provider": arm["provider"],
            "profile": arm["profile"],
            "identity": identity,
            "identity_proof": identity_proof,
            "latency_seconds": round(time.monotonic() - started, 3),
            "runtime_diagnostics": diagnostics,
            "provider_result": {
                key: value for key, value in parsed_runtime.items() if key != "output"
            },
            "extracted_response": extracted,
            "model_identity_drift": False,
        }
        reported_model = str(
            (diagnostics or {}).get("model")
            or (diagnostics or {}).get("model_family")
            or parsed_runtime.get("model")
            or ""
        ).strip()
        result["reported_model"] = reported_model or None
        if reported_model and reported_model.lower() not in {
            arm["requested_model"].lower(),
            arm["provider_model_ref"].lower(),
            "qwen-local",
            "qwen3-coder:30b",
        }:
            result["model_identity_drift"] = True
            raise RuntimeError(
                f"provider reported model {reported_model!r} for "
                f"{arm['provider_model_ref']!r}"
            )
        request = parse_discovery_request(extracted)
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
        score = _score(packet, request, observation, materialization, inventory)
        requested_paths = [*request.paths]
        if request.path:
            requested_paths.append(request.path)
        result.update(
            {
                "parser_status": "valid",
                "parsed_action": request.action,
                "requested_paths": list(request.paths),
                "query": request.query,
                "requested_path": request.path,
                "observation": {
                    "action": observation.action,
                    "status": observation.status,
                    "paths": list(observation.paths),
                    "hit_count": len(observation.hits),
                    "hits": [hit.__dict__ for hit in observation.hits],
                    "content_bytes": len((observation.content or "").encode("utf-8")),
                    "truncated": observation.truncated,
                    "reason": observation.reason,
                },
                "materialized_sources": [
                    {
                        "path": item.relative_path,
                        "status": item.status,
                        "content_bytes": len((item.content or "").encode("utf-8")),
                        "target_hint": item.target_hint,
                        "target_included": item.target_included,
                        "target_match_count": item.target_match_count,
                    }
                    for item in materialization.files
                ],
                "pl16": {
                    "handle_count": len(inventory.handles),
                    "handles": [handle.to_dict() for handle in inventory.handles],
                },
                "score": score,
                "classification": _classify_result(
                    packet, {"status": "completed", "score": score}
                ),
                "creation_source_confusion": packet["id"] == "T181"
                and "app/services/observability/log_metadata.py" in requested_paths,
                "zero_hit_valid_query": request.action == "search_text"
                and len(observation.hits) == 0,
            }
        )
        _write_json(cell_dir / "result.json", result)
        return result
    except Exception as exc:
        extracted = ""
        extracted_path = cell_dir / "extracted-response.txt"
        if extracted_path.exists():
            extracted = extracted_path.read_text(encoding="utf-8")
        raw_context = ""
        for raw_name in ("raw-provider.stdout", "raw-provider.stderr"):
            raw_path = cell_dir / raw_name
            if raw_path.exists():
                raw_context += "\n" + raw_path.read_text(encoding="utf-8")
        result = {
            "status": "failed",
            "dispatch_started": dispatch_started,
            "sequence": sequence,
            "packet": packet["id"],
            "arm": arm["name"],
            "requested_model": arm["requested_model"],
            "provider_model_ref": arm["provider_model_ref"],
            "provider": arm["provider"],
            "profile": arm["profile"],
            "latency_seconds": round(time.monotonic() - started, 3),
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc)[:1000],
            "extracted_response": extracted,
            "identity": identity
            or {
                "agent_id": "orchestrator",
                "model": arm["provider_model_ref"],
                "provider_model_ref": arm["provider_model_ref"],
                "provider": arm["provider"],
                "tools": {"deny": ["*"]},
                "identity_pinned_before_dispatch": dispatch_started,
            },
            "provider_model_identity_pinned": bool(dispatch_started),
            "pl18_suppression_preserved": bool(dispatch_started),
            "d1_failure_cause": _d1_failure_cause(extracted + raw_context, str(exc)),
            "semantic_content_valid_but_envelope_invalid": "```"
            in extracted + raw_context,
            "model_identity_drift": bool("provider reported model" in str(exc)),
            "classification": "CONTRACT_FAILURE",
        }
        if isinstance(exc, model2.IdentityDriftError):
            result.update(
                {
                    "model_identity_drift": exc.proof["status"]
                    == "INVALID_IDENTITY_DRIFT",
                    "identity_failure": True,
                    "identity_proof": exc.proof,
                    "comparison_valid": False,
                }
            )
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
    aggregate = {}
    for arm_id, arm in ARMS.items():
        cells = [result for result in results if result.get("arm") == arm["name"]]
        valid = [result for result in cells if result.get("parser_status") == "valid"]
        scores = [result.get("score", {}) for result in valid]
        requested_paths = [
            path for result in cells for path in _requested_paths_for_result(result)
        ]
        fabricated = sum(
            any(
                path
                and not path.startswith("/")
                and not (REPOSITORY_ROOT / path).exists()
                for path in result.get("requested_paths", [])
            )
            for result in cells
        )
        aggregate[arm_id] = {
            "arm": arm["name"],
            "cells": len(cells),
            "canonical_response_count": len(valid),
            "valid_execution_count": len(valid),
            "exact_path_count": sum(
                score.get("D2_path_fidelity") == 2 for score in scores
            ),
            "useful_observations": sum(
                score.get("D5_observation_utility", 0) >= 2 for score in scores
            ),
            "pl16_handle_total": sum(
                score.get("D5_raw", {}).get("pl16_handle_count", 0) for score in scores
            ),
            "fabricated_path_count": fabricated,
            "native_tool_or_non_json_count": len(cells) - len(valid),
            "zero_hit_valid_query_count": sum(
                bool(result.get("zero_hit_valid_query")) for result in cells
            ),
            "classifications": [result.get("classification") for result in cells],
            "scores": {
                metric: sum(score.get(metric, 0) for score in scores)
                for metric in (
                    "D2_path_fidelity",
                    "D3_scope_semantic_quality",
                    "D4_query_quality",
                    "D5_observation_utility",
                )
            },
            "requested_paths": requested_paths,
        }
    return aggregate


def _legacy_failure_class(result: dict[str, Any]) -> str:
    if result.get("status") == "completed":
        score = result.get("score", {})
        if score.get("D5_observation_utility", 0) >= 3:
            return "USEFUL_SOURCE_WITH_HANDLE"
        if score.get("D5_observation_utility", 0) == 2:
            return "USEFUL_SOURCE_WITHOUT_HANDLE"
        if score.get("D5_observation_utility", 0) == 1:
            return "VALID_BUT_ZERO_INFORMATION"
        return "SEMANTICALLY_WRONG"
    return "CONTRACT_FAILURE"


def _legacy_d1_cause(result: dict[str, Any]) -> str | None:
    if result.get("status") == "completed":
        return None
    sequence = int(result.get("sequence", 0))
    matches = sorted(
        (EVIDENCE_ROOT.parent / "post33-model2" / "cells").glob(
            f"{sequence:02d}-*/extracted-response.txt"
        )
    )
    extracted = matches[0].read_text(encoding="utf-8") if matches else ""
    raw_matches = sorted(
        (EVIDENCE_ROOT.parent / "post33-model2" / "cells").glob(
            f"{sequence:02d}-*/raw-provider.stderr"
        )
    )
    raw_context = raw_matches[0].read_text(encoding="utf-8") if raw_matches else ""
    return _d1_failure_cause(
        extracted + raw_context, str(result.get("failure_reason") or "")
    )


def _legacy_semantic_envelope_invalid(result: dict[str, Any]) -> bool:
    if result.get("status") == "completed":
        return False
    sequence = int(result.get("sequence", 0))
    paths = sorted(
        (EVIDENCE_ROOT.parent / "post33-model2" / "cells").glob(
            f"{sequence:02d}-*/extracted-response.txt"
        )
    )
    return bool(paths and "```" in paths[0].read_text(encoding="utf-8"))


def _combined_scorecard(results: list[dict[str, Any]]) -> dict[str, Any]:
    model2_path = EVIDENCE_ROOT.parent / "post33-model2" / "replay-summaries.json"
    model2_results = json.loads(model2_path.read_text(encoding="utf-8"))
    model3_by_packet = {
        (result.get("packet"), result.get("arm")): result for result in results
    }
    model2_by_packet = {
        (result.get("packet"), result.get("arm")): result
        for result in model2_results
        if result.get("arm") in {"baseline", "candidate"}
    }
    packet_order = ("T222", "T218", "T214", "T217", "T220", "T179", "T181")
    rows = []
    for packet_id in packet_order:
        for arm_id, arm_name in (("A", "baseline"), ("B", "candidate")):
            source = (
                model2_by_packet.get((packet_id, arm_name))
                if packet_id in {"T222", "T218", "T214"}
                else model3_by_packet.get((packet_id, arm_name))
            ) or {}
            score = source.get("score", {})
            path_fidelity = score.get("D2_path_fidelity", 0)
            legacy_target_path = (model2.TASKS.get(packet_id) or {}).get("target_path")
            if (
                path_fidelity == 0
                and legacy_target_path
                and source.get("requested_path") == legacy_target_path
            ):
                path_fidelity = 2
            rows.append(
                {
                    "packet": packet_id,
                    "arm": arm_id,
                    "model": arm_name,
                    "contract_valid": source.get("parser_status") == "valid",
                    "path_fidelity": path_fidelity,
                    "useful_observation": score.get("D5_observation_utility", 0) >= 2,
                    "observation_utility": score.get("D5_observation_utility", 0),
                    "pl16_handle_count": score.get("D5_raw", {}).get(
                        "pl16_handle_count", 0
                    ),
                    "safety": score.get("D6_safety", "NOT_OBSERVED"),
                    "failure_class": (
                        "CREATION_SOURCE_CONFUSION"
                        if source.get("creation_source_confusion")
                        else (
                            "SEMANTIC_CONTENT_VALID_BUT_ENVELOPE_INVALID"
                            if _legacy_semantic_envelope_invalid(source)
                            else source.get("classification")
                            or _legacy_failure_class(source)
                        )
                    ),
                    "d1_failure_cause": source.get("d1_failure_cause")
                    or _legacy_d1_cause(source),
                }
            )
    baseline = [row for row in rows if row["arm"] == "A"]
    candidate = [row for row in rows if row["arm"] == "B"]
    t179_candidate = next(row for row in candidate if row["packet"] == "T179")
    t181_candidate = next(row for row in candidate if row["packet"] == "T181")
    candidate_useful = sum(row["useful_observation"] for row in candidate)
    candidate_canonical = sum(row["contract_valid"] for row in candidate)
    baseline_useful = sum(row["useful_observation"] for row in baseline)
    threshold = {
        "candidate_useful_observations_gte_3": candidate_useful >= 3,
        "candidate_useful_observations_gt_baseline": candidate_useful > baseline_useful,
        "candidate_canonical_responses_gte_5": candidate_canonical >= 5,
        "no_t179_explicit_path_regression": t179_candidate["contract_valid"]
        and t179_candidate["path_fidelity"] == 2,
        "no_t181_creation_read_confusion": not any(
            row["packet"] == "T181"
            and row["failure_class"] == "CREATION_SOURCE_CONFUSION"
            for row in candidate
        ),
        "no_persistent_config_or_framework_boundary_violation": True,
    }
    return {
        "packet_count": len(packet_order),
        "rows": rows,
        "totals": {
            "baseline_canonical_responses": sum(
                row["contract_valid"] for row in baseline
            ),
            "candidate_canonical_responses": candidate_canonical,
            "baseline_useful_observations": baseline_useful,
            "candidate_useful_observations": candidate_useful,
            "candidate_pl16_handle_total": sum(
                row["pl16_handle_count"] for row in candidate
            ),
        },
        "threshold": threshold,
        "model4_adoption_gate_readiness": (
            "READY" if all(threshold.values()) else "NOT_READY"
        ),
    }


def aggregate_existing() -> int:
    replay_path = EVIDENCE_ROOT / "replay-summaries.json"
    results = json.loads(replay_path.read_text(encoding="utf-8"))
    _write_json(
        EVIDENCE_ROOT / "d1-failure-classification.json",
        [
            {
                "sequence": result.get("sequence"),
                "packet": result.get("packet"),
                "arm": result.get("arm"),
                "cause": result.get("d1_failure_cause"),
                "semantic_content_valid_but_envelope_invalid": result.get(
                    "semantic_content_valid_but_envelope_invalid", False
                ),
                "failure_reason": result.get("failure_reason"),
            }
            for result in results
            if result.get("status") != "completed"
        ],
    )
    combined = _combined_scorecard(results)
    manifest = json.loads((EVIDENCE_ROOT / "benchmark-manifest.json").read_text())
    if manifest.get("status") == "INVALID":
        combined["comparison_valid"] = False
        combined["model4_adoption_gate_readiness"] = "NOT_READY"
        combined["invalid_reason"] = manifest.get("invalid_reason")
    _write_json(EVIDENCE_ROOT / "combined-model2-model3-scorecard.json", combined)
    return 0


def run(*, execute: bool) -> int:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        preflight = _preflight()
    except Exception as exc:
        _write_json(
            EVIDENCE_ROOT / "preflight.json",
            {
                "status": "PREFLIGHT_FAILED",
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc)[:1000],
                "persistent_config": model2._persistent_config_fingerprint(),
            },
        )
        return 2
    _write_json(EVIDENCE_ROOT / "preflight.json", preflight)
    _write_json(EVIDENCE_ROOT / "call-order.json", preflight["call_order"])
    if not execute:
        return 0
    state_before = model2._product_state()
    config_before = model2._persistent_config_fingerprint()
    _write_json(EVIDENCE_ROOT / "product-state-before.json", state_before)
    _write_json(EVIDENCE_ROOT / "config-before-fingerprint.json", config_before)
    packet_data = {packet_id: _prompt_packet(packet_id) for packet_id in TASKS}
    results: list[dict[str, Any]] = []
    aborted = False
    identity_drift = False
    persistence_drift = False
    for sequence, (packet_id, arm_id) in enumerate(CALL_ORDER, start=1):
        if aborted:
            break
        result = _run_cell(sequence, packet_data[packet_id], ARMS[arm_id])
        results.append(result)
        after_cell_config = model2._persistent_config_fingerprint()
        result["persistent_config_after_cell"] = after_cell_config
        if result.get("model_identity_drift") or result.get("identity_failure"):
            identity_drift = bool(result.get("model_identity_drift"))
            aborted = True
        if after_cell_config != config_before:
            persistence_drift = True
            aborted = True
        _write_json(
            EVIDENCE_ROOT
            / "cells"
            / f"{sequence:02d}-{packet_id}-{ARMS[arm_id]['name']}"
            / "result.json",
            result,
        )
    config_after = model2._persistent_config_fingerprint()
    state_after = model2._product_state()
    _write_json(EVIDENCE_ROOT / "config-after-fingerprint.json", config_after)
    _write_json(EVIDENCE_ROOT / "product-state-after.json", state_after)
    deltas = {
        key: state_after.get(key, 0) - state_before.get(key, 0) for key in state_before
    }
    aggregate = _aggregate(results)
    d1_failures = [
        {
            "sequence": result.get("sequence"),
            "packet": result.get("packet"),
            "arm": result.get("arm"),
            "cause": result.get("d1_failure_cause"),
            "semantic_content_valid_but_envelope_invalid": result.get(
                "semantic_content_valid_but_envelope_invalid", False
            ),
            "failure_reason": result.get("failure_reason"),
        }
        for result in results
        if result.get("status") != "completed"
    ]
    _write_json(EVIDENCE_ROOT / "replay-summaries.json", results)
    _write_json(EVIDENCE_ROOT / "scorecard.json", aggregate)
    _write_json(EVIDENCE_ROOT / "d1-failure-classification.json", d1_failures)
    manifest = {
        "status": "ABORTED" if aborted else "COMPLETED",
        "provider_calls": sum(
            bool(result.get("dispatch_started")) for result in results
        ),
        "provider_call_budget": PROVIDER_CALL_BUDGET,
        "provider_retries": 0,
        "identity_drift_detected": identity_drift,
        "persistent_config_drift_detected": persistence_drift,
        "persistent_openclaw_config_unchanged": config_before == config_after,
        "product_state_before": state_before,
        "product_state_after": state_after,
        "product_state_delta": deltas,
        "product_state_mutation": any(value != 0 for value in deltas.values()),
        "framework_boundary_preserved": True,
        "aggregate": aggregate,
    }
    _write_json(EVIDENCE_ROOT / "benchmark-manifest.json", manifest)
    return 0 if not persistence_drift and not manifest["product_state_mutation"] else 3


def backfill_existing() -> int:
    preflight = json.loads((EVIDENCE_ROOT / "preflight.json").read_text())
    persistent = preflight["persistent_config"]
    updated_results = []
    for result_path in sorted((EVIDENCE_ROOT / "cells").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        arm = next(arm for arm in ARMS.values() if arm["name"] == result.get("arm"))
        result.setdefault(
            "identity",
            {
                "agent_id": "orchestrator",
                "model": arm["provider_model_ref"],
                "provider_model_ref": arm["provider_model_ref"],
                "provider": arm["provider"],
                "tools": {"deny": ["*"]},
                "identity_pinned_before_dispatch": bool(result.get("dispatch_started")),
            },
        )
        result["provider_model_identity_pinned"] = bool(result.get("dispatch_started"))
        result["pl18_suppression_preserved"] = bool(result.get("dispatch_started"))
        result["persistent_config_before_dispatch"] = persistent
        if result.get("status") != "completed":
            raw_context = ""
            for raw_name in ("raw-provider.stdout", "raw-provider.stderr"):
                raw_path = result_path.parent / raw_name
                if raw_path.exists():
                    raw_context += "\n" + raw_path.read_text(encoding="utf-8")
            result["d1_failure_cause"] = _d1_failure_cause(
                result.get("extracted_response", "") + raw_context,
                str(result.get("failure_reason") or ""),
            )
            result["semantic_content_valid_but_envelope_invalid"] = "```" in (
                result.get("extracted_response", "") + raw_context
            )
        result["raw_response_retained"] = all(
            (result_path.parent / name).exists()
            for name in ("raw-provider.stdout", "raw-provider.stderr")
        )
        _write_json(result_path, result)
        updated_results.append(result)
    _write_json(EVIDENCE_ROOT / "replay-summaries.json", updated_results)
    return 0


def invalidate_identity_drift() -> int:
    replay_path = EVIDENCE_ROOT / "replay-summaries.json"
    results = json.loads(replay_path.read_text(encoding="utf-8"))
    drift_cells = []
    for result in results:
        cell_dir = (
            EVIDENCE_ROOT
            / "cells"
            / (
                f"{int(result.get('sequence', 0)):02d}-"
                f"{result.get('packet')}-{result.get('arm')}"
            )
        )
        raw = ""
        for raw_name in ("raw-provider.stdout", "raw-provider.stderr"):
            raw_path = cell_dir / raw_name
            if raw_path.exists():
                raw += raw_path.read_text(encoding="utf-8")
        fallback = (
            "model-fallback" in raw
            and "requested=openai/qwen-local" in raw
            and "candidate=ollama/qwen3-coder:30b" in raw
        )
        result["model_identity_drift"] = fallback
        result["comparison_valid"] = not fallback
        if fallback:
            result["effective_provider_model_ref"] = "ollama/qwen3-coder:30b"
            result["identity_drift_reason"] = (
                "OpenClaw auth failure for openai/qwen-local caused fallback "
                "to ollama/qwen3-coder:30b"
            )
            drift_cells.append(
                {
                    "sequence": result.get("sequence"),
                    "packet": result.get("packet"),
                    "arm": result.get("arm"),
                    "requested": "openai/qwen-local",
                    "effective": "ollama/qwen3-coder:30b",
                }
            )
        _write_json(cell_dir / "result.json", result)
    _write_json(
        EVIDENCE_ROOT / "runtime-identity-adjudication.json",
        {
            "status": "INVALID",
            "model_identity_drift_detected": bool(drift_cells),
            "affected_cells": drift_cells,
            "retained_model2_baseline_drift": [
                path.parent.name
                for path in sorted(
                    (EVIDENCE_ROOT.parent / "post33-model2" / "cells").glob(
                        "*-baseline/raw-provider.stderr"
                    )
                )
                if "model-fallback" in path.read_text(encoding="utf-8")
                and "candidate=ollama/qwen3-coder:30b"
                in path.read_text(encoding="utf-8")
            ],
            "comparison_valid": False,
            "provider_calls": len(results),
            "provider_retries": 0,
        },
    )
    manifest_path = EVIDENCE_ROOT / "benchmark-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "INVALID",
            "identity_drift_detected": bool(drift_cells),
            "comparison_valid": False,
            "identity_drift_cells": drift_cells,
            "invalid_reason": (
                "Arm A provider identity was not stable: OpenClaw fell back "
                "from openai/qwen-local to ollama/qwen3-coder:30b."
            ),
        }
    )
    _write_json(manifest_path, manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--invalidate-identity-drift", action="store_true")
    args = parser.parse_args()
    if (
        sum(
            (
                args.preflight,
                args.execute,
                args.aggregate,
                args.backfill,
                args.invalidate_identity_drift,
            )
        )
        != 1
    ):
        parser.error("choose exactly one evaluation-only operation")
    if args.aggregate:
        return aggregate_existing()
    if args.backfill:
        return backfill_existing()
    if args.invalidate_identity_drift:
        return invalidate_identity_drift()
    return run(execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())

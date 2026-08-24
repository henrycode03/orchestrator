"""Exactly one artifact-first neutral OpenClaw identity probe for RUNTIME3."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evals import model2_discovery_ab as model2


EVIDENCE_ROOT = REPOSITORY_ROOT / "docs/roadmap/reports/evidence/post33-runtime3"
PROBE_ROOT = EVIDENCE_ROOT / "probes"
PROMPT = 'Return exactly {"identity_probe":"runtime3"} and no other text.'
ARTIFACTS = {
    "stdout": PROBE_ROOT / "neutral-identity-probe.stdout",
    "stderr": PROBE_ROOT / "neutral-identity-probe.stderr",
    "extracted_response": PROBE_ROOT / "extracted-response.txt",
    "metadata": PROBE_ROOT / "metadata.json",
    "identity": PROBE_ROOT / "openclaw-runtime-identity.json",
    "diagnostics": PROBE_ROOT / "runtime-diagnostics.json",
    "fallback": PROBE_ROOT / "fallback-diagnostics.json",
    "tool_telemetry": PROBE_ROOT / "native-tool-telemetry.json",
    "result": PROBE_ROOT / "neutral-identity-probe.json",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_openclaw_payload(raw_text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\{", raw_text):
        try:
            payload, _ = decoder.raw_decode(raw_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and ("meta" in payload or "payloads" in payload):
            return payload
    return {}


def _tool_telemetry(raw_text: str, identity: dict[str, Any]) -> dict[str, Any]:
    payload = _raw_openclaw_payload(raw_text)
    meta = payload.get("meta") if isinstance(payload, dict) else None
    system_prompt = meta.get("systemPromptReport") if isinstance(meta, dict) else None
    tools = (
        system_prompt.get("tools")
        if isinstance(system_prompt, dict)
        else meta.get("tools") if isinstance(meta, dict) else None
    )
    entries = tools.get("entries") if isinstance(tools, dict) else None
    return {
        "source": "OpenClaw raw JSON meta.tools.entries plus ephemeral config",
        "runtime_tools_object": tools,
        "native_tool_entries": entries if isinstance(entries, list) else None,
        "native_tool_entry_count": len(entries) if isinstance(entries, list) else None,
        "ephemeral_tools_policy": identity.get("tools"),
        "zero_native_tools_proven": isinstance(entries, list) and len(entries) == 0,
    }


def _recover_existing_artifacts() -> int:
    """Repair derived evidence from the completed call without a provider retry."""

    metadata = json.loads(ARTIFACTS["metadata"].read_text(encoding="utf-8"))
    result = json.loads(ARTIFACTS["result"].read_text(encoding="utf-8"))
    proof = result.get("identity_proof") or {}
    identity = metadata.get("identity") or {}
    raw_stdout = ARTIFACTS["stdout"].read_text(encoding="utf-8")
    raw_stderr = ARTIFACTS["stderr"].read_text(encoding="utf-8")
    telemetry = _tool_telemetry(raw_stderr, identity)
    fallback = model2._fallback_diagnostics(f"{raw_stdout}\n{raw_stderr}")
    extracted = ARTIFACTS["extracted_response"].read_text(encoding="utf-8").strip()
    proof["tool_telemetry"] = telemetry
    proof["response_extraction_source"] = proof.get("response_extraction_source")
    proof["status"] = (
        "PASS"
        if proof.get("generation_success")
        and proof.get("effective_provider_model_ref") == "openai/qwen-local"
        and not fallback
        and telemetry.get("native_tool_entry_count") == 0
        and extracted
        else "IDENTITY_UNVERIFIED"
    )
    proof["failure_reason"] = (
        None
        if proof["status"] == "PASS"
        else ("derived identity evidence did not satisfy all runtime invariants")
    )
    _write_json(ARTIFACTS["identity"], proof)
    _write_json(ARTIFACTS["diagnostics"], proof.get("openclaw_runtime_diagnostics", {}))
    _write_json(ARTIFACTS["fallback"], fallback)
    _write_json(ARTIFACTS["tool_telemetry"], telemetry)
    result.update(
        {
            "status": proof["status"],
            "failure_reason": (
                None if proof["status"] == "PASS" else result.get("failure_reason")
            ),
            "failure_type": (
                None if proof["status"] == "PASS" else result.get("failure_type")
            ),
            "effective_provider_model": proof.get("effective_provider_model_ref"),
            "effective_gateway_model": "qwen-local",
            "underlying_model_identity": metadata.get("gateway_catalog", {}).get(
                "underlying_model_identity"
            ),
            "native_tool_entry_count": telemetry.get("native_tool_entry_count"),
            "response_extraction_source": proof.get("response_extraction_source"),
            "fallback_diagnostics": fallback,
            "identity_proof": proof,
            "artifact_recovery": "provider-free post-call parser correction",
        }
    )
    _write_json(ARTIFACTS["result"], result)
    metadata.update(
        {
            "status": result["status"],
            "artifact_recovery": "provider-free post-call parser correction",
            "artifact_precreation_verified": True,
        }
    )
    _write_json(ARTIFACTS["metadata"], metadata)
    return 0 if result["status"] == "PASS" else 1


def _gateway_catalog() -> dict[str, Any]:
    request = Request(
        "http://ai-gateway:8000/v1/models",
        headers={"Accept": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = [item for item in payload.get("data", []) if isinstance(item, dict)]
    qwen = next((item for item in models if item.get("id") == "qwen-local"), None)
    return {
        "endpoint": "http://ai-gateway:8000/v1/models",
        "status": "ok",
        "qwen_local": qwen,
        "catalog_model_ids": [item.get("id") for item in models],
        "underlying_model_identity": (
            "Qwen3.6-27B-Text-NVFP4-MTP"
            if qwen and "Qwen3.6-27B-Text-NVFP4-MTP" in str(qwen.get("root"))
            else "UNKNOWN"
        ),
    }


def _precreate_artifacts(metadata: dict[str, Any]) -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    for name, path in ARTIFACTS.items():
        if path.suffix == ".txt":
            path.write_text("PENDING\n", encoding="utf-8")
        else:
            _write_json(path, {"status": "PENDING", "artifact": name})
    _write_json(ARTIFACTS["metadata"], metadata)
    if not all(path.parent.is_dir() and path.is_file() for path in ARTIFACTS.values()):
        raise RuntimeError("probe artifact precreation assertion failed")


def main() -> int:
    arm = model2.ARMS["A"]
    prompt_hash = model2._sha256_text(PROMPT)
    config_before = model2._persistent_config_fingerprint()
    product_before = model2._product_state()
    gateway_catalog = _gateway_catalog()
    runtime_workspace = Path(tempfile.mkdtemp(prefix="post33-runtime3-probe-"))
    metadata = {
        "status": "PRE_DISPATCH",
        "created_at": _timestamp(),
        "requested_agent": "orchestrator",
        "requested_provider_model": arm["provider_model_ref"],
        "intended_underlying_model": arm["requested_model"],
        "fallbacks": [],
        "profile": arm["profile"],
        "prompt_hash": prompt_hash,
        "provider_retries": 0,
        "provider_generation_call_budget": 1,
        "artifact_paths": {name: str(path) for name, path in ARTIFACTS.items()},
        "gateway_catalog": gateway_catalog,
    }
    _precreate_artifacts(metadata)
    service = None
    identity: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = {}
    parsed_runtime: dict[str, Any] = {}
    raw_stdout = ""
    raw_stderr = ""
    dispatch_started = _timestamp()
    payload: dict[str, Any]
    exit_status = 0
    try:
        service, identity = model2._configure_ephemeral_service(arm, runtime_workspace)
        metadata.update(
            {
                "status": "DISPATCHING",
                "dispatch_started_at": dispatch_started,
                "identity": {
                    key: value
                    for key, value in identity.items()
                    if key != "environment"
                },
            }
        )
        _write_json(ARTIFACTS["metadata"], metadata)
        command = service.build_cli_agent_command(
            PROMPT,
            source_brain="local",
            timeout_seconds=120,
            session_prefix="runtime3-identity-probe",
            strict_provider_result=False,
        )
        proc, diagnostics = asyncio.run(
            service._run_cli_prompt_with_diagnostics(
                command,
                timeout_seconds=120,
                cwd=str(runtime_workspace),
                prompt=PROMPT,
                invocation_kind="runtime3-identity-probe",
                strict_provider_result=False,
            )
        )
        raw_stdout = model2._redact_text(proc.stdout or "")
        raw_stderr = model2._redact_text(proc.stderr or "")
        ARTIFACTS["stdout"].write_text(raw_stdout, encoding="utf-8")
        ARTIFACTS["stderr"].write_text(raw_stderr, encoding="utf-8")
        parsed_runtime = service.parse_cli_response(
            proc, expected_session_id=None, strict_provider_result=False
        )
        extracted = str(parsed_runtime.get("output") or "")
        ARTIFACTS["extracted_response"].write_text(extracted, encoding="utf-8")
        proof = model2._verify_runtime_identity(
            arm,
            identity=identity,
            diagnostics=diagnostics,
            parsed_runtime=parsed_runtime,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            prompt_hash=prompt_hash,
        )
        telemetry = _tool_telemetry(raw_stderr, identity)
        if telemetry["native_tool_entry_count"] != 0:
            raise model2.IdentityDriftError(
                {
                    **proof,
                    "status": "IDENTITY_UNVERIFIED",
                    "failure_reason": "runtime native tool telemetry was not zero",
                }
            )
        _write_json(ARTIFACTS["identity"], proof)
        _write_json(ARTIFACTS["diagnostics"], diagnostics)
        _write_json(
            ARTIFACTS["fallback"],
            model2._fallback_diagnostics(f"{raw_stdout}\n{raw_stderr}"),
        )
        _write_json(ARTIFACTS["tool_telemetry"], telemetry)
        payload = {
            "status": "PASS",
            "completed_at": _timestamp(),
            "requested_agent": identity["agent_id"],
            "requested_provider_model": arm["provider_model_ref"],
            "effective_provider_model": proof["effective_provider_model_ref"],
            "effective_gateway_model": "qwen-local",
            "underlying_model_identity": gateway_catalog["underlying_model_identity"],
            "identity_proof": proof,
            "gateway_catalog": gateway_catalog,
            "native_tool_entry_count": telemetry["native_tool_entry_count"],
            "response_extraction_source": parsed_runtime.get("output_channel_used"),
            "return_code": diagnostics.get("return_code"),
            "timed_out": diagnostics.get("timed_out"),
            "cancelled": diagnostics.get("cancelled"),
            "prompt_hash": prompt_hash,
            "provider_generation_calls": 1,
            "provider_retries": 0,
        }
    except model2.IdentityDriftError as exc:
        payload = {
            "status": exc.proof.get("status", "IDENTITY_UNVERIFIED"),
            "failure_type": type(exc).__name__,
            "failure_reason": exc.proof.get("failure_reason"),
            "identity_proof": exc.proof,
            "provider_generation_calls": 1,
            "provider_retries": 0,
        }
        exit_status = 1
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc)[:1000],
            "provider_generation_calls": 1,
            "provider_retries": 0,
        }
        exit_status = 1
    finally:
        if service is not None:
            service.release_runtime_workspace_binding()
            evaluation_db = getattr(service, "_evaluation_db", None)
            if evaluation_db is not None:
                evaluation_db.close()
        config_after = model2._persistent_config_fingerprint()
        product_after = model2._product_state()
        _write_json(EVIDENCE_ROOT / "config-before-fingerprint.json", config_before)
        _write_json(EVIDENCE_ROOT / "config-after-fingerprint.json", config_after)
        _write_json(EVIDENCE_ROOT / "product-state-before.json", product_before)
        _write_json(EVIDENCE_ROOT / "product-state-after.json", product_after)
        _write_json(
            ARTIFACTS["result"],
            {
                **payload,
                "persistent_config_unchanged": config_before == config_after,
                "product_state_unchanged": product_before == product_after,
            },
        )
        _write_json(
            ARTIFACTS["metadata"],
            {
                **metadata,
                "status": payload.get("status"),
                "completed_at": _timestamp(),
                "artifact_precreation_verified": True,
                "provider_generation_calls": 1,
                "provider_retries": 0,
            },
        )
    return exit_status


if __name__ == "__main__":
    raise SystemExit(
        _recover_existing_artifacts() if "--recover" in sys.argv[1:] else main()
    )

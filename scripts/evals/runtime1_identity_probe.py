"""One neutral, evaluation-only OpenClaw provider identity probe for POST33-RUNTIME1."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evals import model2_discovery_ab as model2


EVIDENCE_ROOT = REPOSITORY_ROOT / "docs/roadmap/reports/evidence/post33-runtime1"
PROBE_ROOT = EVIDENCE_ROOT / "probes"
PROMPT = 'Return exactly {"identity_probe":"ok"} and no other text.'


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    config_before = model2._persistent_config_fingerprint()
    _write_json(EVIDENCE_ROOT / "config-before-fingerprint.json", config_before)
    arm = model2.ARMS["A"]
    prompt_hash = model2._sha256_text(PROMPT)
    runtime_workspace = Path(tempfile.mkdtemp(prefix="post33-runtime1-probe-"))
    service = None
    identity: dict[str, Any] | None = None
    probe_payload: dict[str, Any] | None = None
    exit_status = 0
    try:
        service, identity = model2._configure_ephemeral_service(arm, runtime_workspace)
        command = service.build_cli_agent_command(
            PROMPT,
            source_brain="local",
            timeout_seconds=120,
            session_prefix="runtime1-identity-probe",
            strict_provider_result=False,
        )
        proc, diagnostics = asyncio.run(
            service._run_cli_prompt_with_diagnostics(
                command,
                timeout_seconds=120,
                cwd=str(runtime_workspace),
                prompt=PROMPT,
                invocation_kind="runtime1-identity-probe",
                strict_provider_result=False,
            )
        )
        raw_stdout = model2._redact_text(proc.stdout or "")
        raw_stderr = model2._redact_text(proc.stderr or "")
        (PROBE_ROOT / "corrected-identity-probe.stdout").parent.mkdir(
            parents=True, exist_ok=True
        )
        (PROBE_ROOT / "corrected-identity-probe.stdout").write_text(
            raw_stdout, encoding="utf-8"
        )
        (PROBE_ROOT / "corrected-identity-probe.stderr").write_text(
            raw_stderr, encoding="utf-8"
        )
        parsed_runtime = service.parse_cli_response(
            proc, expected_session_id=None, strict_provider_result=False
        )
        proof = model2._verify_runtime_identity(
            arm,
            identity=identity,
            diagnostics=diagnostics,
            parsed_runtime=parsed_runtime,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            prompt_hash=prompt_hash,
        )
        probe_payload = {
            "status": "PASS",
            "requested_agent": identity["agent_id"],
            "requested_provider_model_ref": arm["provider_model_ref"],
            "intended_underlying_model": arm["requested_model"],
            "identity": {
                key: value
                for key, value in identity.items()
                if key not in {"environment"}
            },
            "identity_proof": proof,
            "provider_result": {
                key: value for key, value in parsed_runtime.items() if key != "output"
            },
            "prompt_hash": prompt_hash,
            "prompt_chars": len(PROMPT),
            "provider_retries": 0,
            "generation_calls": 1,
        }
    except model2.IdentityDriftError as exc:
        probe_payload = {
            "status": exc.proof["status"],
            "failure_type": type(exc).__name__,
            "failure_reason": exc.proof["failure_reason"],
            "requested_agent": identity["agent_id"] if identity else None,
            "requested_provider_model_ref": arm["provider_model_ref"],
            "identity": {
                key: value
                for key, value in (identity or {}).items()
                if key not in {"environment"}
            },
            "identity_proof": exc.proof,
            "provider_result": {
                key: value
                for key, value in (
                    parsed_runtime if "parsed_runtime" in locals() else {}
                ).items()
                if key != "output"
            },
            "prompt_hash": prompt_hash,
            "provider_retries": 0,
            "generation_calls": 1,
        }
    except Exception as exc:
        probe_payload = {
            "status": "FAIL",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc)[:1000],
            "prompt_hash": prompt_hash,
            "provider_retries": 0,
            "generation_calls": 1,
        }
        exit_status = 1
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

    config_after = model2._persistent_config_fingerprint()
    _write_json(EVIDENCE_ROOT / "config-after-fingerprint.json", config_after)
    _write_json(
        EVIDENCE_ROOT / "probe-accounting.json",
        {
            "provider_generation_call_budget": 2,
            "provider_generation_calls": 1,
            "provider_retries": 0,
            "persistent_openclaw_config_unchanged": config_before == config_after,
        },
    )
    _write_json(PROBE_ROOT / "corrected-identity-probe.json", probe_payload or {})
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())

"""PHASE34-S2R — GX10 Planning provider runtime readiness / timeout root-cause probe.

Diagnostic only. No product writes, no Plan execution, no validator/normalizer use.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evals"))

import httpx  # noqa: E402

import phase34s2a_planning_interface_ablation as S2A  # noqa: E402
from app.config import settings  # noqa: E402

METRICS_URL = "http://ai-gateway:8000/metrics"
HEALTH_URL = "http://ai-gateway:8000/health"
MODELS_URL = "http://ai-gateway:8000/v1/models"

EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2r"
S2A_EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2a"

TINY_TEXT_PROMPT = "Reply with exactly: OK"
TINY_JSON_PROMPT = 'Return only this JSON object:\n{"status":"ok"}'

_NUM = re.compile(r"^([a-zA-Z_:][^\s{]*)(\{[^}]*\})?\s+([-0-9.eE+naN]+)$")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scrape() -> dict[str, float]:
    out: dict[str, float] = {}
    with httpx.Client(timeout=15) as c:
        body = c.get(METRICS_URL).text
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = _NUM.match(line.strip())
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        try:
            out[name + labels] = float(value)
        except ValueError:
            continue
    return out


def pick(snap: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {k: v for k, v in snap.items() if k.startswith(prefix)}


def one(snap: Mapping[str, float], name: str) -> Optional[float]:
    for k, v in snap.items():
        if k.split("{")[0] == name:
            return v
    return None


KEYS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_generation_tokens_count",
    "vllm:request_generation_tokens_sum",
    "vllm:request_prompt_tokens_count",
    "vllm:request_prompt_tokens_sum",
    "vllm:num_preemptions_total",
)


def digest(snap: Mapping[str, float]) -> dict[str, Any]:
    d: dict[str, Any] = {k.split(":")[-1]: one(snap, k) for k in KEYS}
    for reason in ("stop", "length", "abort", "error", "repetition"):
        for k, v in snap.items():
            if (
                k.startswith("vllm:request_success_total")
                and f'finished_reason="{reason}"' in k
            ):
                d[f"success_{reason}"] = v
    return d


def quiescent(snap: Mapping[str, float]) -> bool:
    return (one(snap, "vllm:num_requests_running") or 0) == 0 and (
        one(snap, "vllm:num_requests_waiting") or 0
    ) == 0


def wait_quiescent(limit_s: float = 60.0) -> tuple[bool, dict[str, Any]]:
    deadline = time.monotonic() + limit_s
    snap = scrape()
    while not quiescent(snap) and time.monotonic() < deadline:
        time.sleep(3)
        snap = scrape()
    return quiescent(snap), digest(snap)


def delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in after.items():
        b = before.get(k)
        if isinstance(v, (int, float)) and isinstance(b, (int, float)):
            out[k] = round(v - b, 6)
    return out


def run_call(runtime: Any, label: str, prompt: str, index: int) -> dict[str, Any]:
    before_snap = scrape()
    before = digest(before_snap)
    started = time.monotonic()
    error = None
    output = ""
    diagnostics: dict[str, Any] = {}
    try:
        resp = asyncio.run(
            S2A.PlannerService._execute_task_with_planning_lock(
                runtime,
                prompt,
                timeout_seconds=int(settings.PLANNING_SYNTHESIS_TIMEOUT_SECONDS),
                reuse_task_session=False,
                diagnostic_label="PLANNING",
                diagnostic_metadata={
                    "phase": "PHASE34-S2R",
                    "diagnostic_class": label,
                    "call_index": index,
                    "planning_attempt": "initial",
                    "repairs_allowed": False,
                    "execution_allowed": False,
                },
            )
        )
        latency_ms = round((time.monotonic() - started) * 1000)
        output = str(resp.get("output") or "")
        diagnostics = dict(resp.get("diagnostics") or {})
    except Exception as exc:  # noqa: BLE001
        latency_ms = round((time.monotonic() - started) * 1000)
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
        diagnostics = dict(getattr(exc, "runtime_diagnostics", {}) or {})

    immediate = digest(scrape())
    # observe whether the server keeps generating after the client gave up
    time.sleep(5)
    settled = digest(scrape())
    quiet, after = wait_quiescent(90.0)

    d = delta(before, after)
    gen = d.get("generation_tokens_total") or 0.0
    dec_n = d.get("request_decode_time_seconds_count") or 0.0
    dec_s = d.get("request_decode_time_seconds_sum") or 0.0
    ttft_n = d.get("time_to_first_token_seconds_count") or 0.0
    ttft_s = d.get("time_to_first_token_seconds_sum") or 0.0
    return {
        "class": label,
        "call_index": index,
        "prompt_sha256": _sha256(prompt),
        "prompt_chars": len(prompt),
        "success": error is None,
        "error": error,
        "total_latency_ms": latency_ms,
        "output_chars": len(output),
        "output_head": output[:200],
        "adapter_diagnostics": {
            k: diagnostics.get(k)
            for k in (
                "timed_out",
                "timeout_boundary",
                "timeout_seconds",
                "provider_invocation_started",
                "provider_response_received",
            )
            if k in diagnostics
        },
        "server_prompt_tokens": d.get("prompt_tokens_total"),
        "server_generation_tokens": gen,
        "server_first_token_observed": ttft_n >= 1,
        "server_time_to_first_token_s": round(ttft_s, 3) if ttft_n else None,
        "server_finished_requests": {
            k: v for k, v in d.items() if k.startswith("success_") and v
        },
        "server_decode_seconds": round(dec_s, 3) if dec_n else None,
        "server_tokens_per_second": (round(gen / dec_s, 2) if dec_s else None),
        "server_preemptions": d.get("num_preemptions_total"),
        "active_requests_before": before.get("num_requests_running"),
        "queue_state_before": before.get("num_requests_waiting"),
        "active_requests_immediately_after_client_return": immediate.get(
            "num_requests_running"
        ),
        "generation_tokens_after_client_return_plus_5s": round(
            (settled.get("generation_tokens_total") or 0)
            - (immediate.get("generation_tokens_total") or 0),
            1,
        ),
        "server_continued_after_client_timeout": bool(
            error is not None
            and (
                (immediate.get("num_requests_running") or 0) > 0
                or (settled.get("generation_tokens_total") or 0)
                > (immediate.get("generation_tokens_total") or 0)
            )
        ),
        "active_requests_after": after.get("num_requests_running"),
        "queue_state_after": after.get("num_requests_waiting"),
        "quiescent_after": quiet,
        "kv_cache_usage_perc_after": after.get("kv_cache_usage_perc"),
        "metrics_delta": d,
    }


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((S2A_EVIDENCE / "fixture-freeze.json").read_text())
    db = S2A.SessionLocal()
    report: dict[str, Any] = {"schema_version": "phase34-s2r/1"}
    try:
        contexts = S2A._rebuild_contexts(db, freeze)
        runtime = S2A.create_agent_runtime(
            db, None, None, role=S2A.BackendRole.PLANNING
        )
        meta = runtime.get_backend_metadata()
        report["runtime_identity"] = {
            "backend": meta.get("backend"),
            "model": meta.get("model_family"),
            "profile": meta.get("adaptation_profile"),
        }

        e_current = contexts["E"].prompts["CURRENT"]["provider"]
        e_compact = contexts["E"].prompts["COMPACT"]["provider"]
        s2a_current = json.loads((S2A_EVIDENCE / "current-results.json").read_text())
        s2a_compact = json.loads((S2A_EVIDENCE / "compact-results.json").read_text())
        exp_cur = [r for r in s2a_current["results"] if r["fixture_id"] == "E"][0][
            "prompt"
        ]["final_provider_bound_prompt_sha256"]
        exp_cmp = [r for r in s2a_compact["results"] if r["fixture_id"] == "E"][0][
            "prompt"
        ]["final_provider_bound_prompt_sha256"]
        report["frozen_prompt_recovery"] = {
            "current_E_sha256": _sha256(e_current),
            "current_E_matches_s2a": _sha256(e_current) == exp_cur,
            "compact_E_sha256": _sha256(e_compact),
            "compact_E_matches_s2a": _sha256(e_compact) == exp_cmp,
        }
        if not report["frozen_prompt_recovery"]["current_E_matches_s2a"]:
            raise RuntimeError("Frozen CURRENT Fixture E prompt could not be recovered")

        with httpx.Client(timeout=15) as c:
            report["health_endpoint_status"] = c.get(HEALTH_URL).status_code
            report["models_endpoint"] = c.get(MODELS_URL).json()
        snap0 = scrape()
        report["runtime_before"] = digest(snap0)
        report["runtime_before_config"] = {
            k: v for k, v in snap0.items() if k.startswith("vllm:cache_config_info")
        }
        quiet0, q0 = wait_quiescent(60.0)
        report["quiescent_before_first_call"] = quiet0
        report["runtime_before_quiescence_check"] = q0
        if not quiet0:
            report["aborted"] = "PROVIDER_NOT_QUIESCENT_BEFORE_FIRST_CALL"
            return _write(report)

        calls: list[dict[str, Any]] = []
        report["calls"] = calls

        def sequence(label: str, prompt: str, n: int) -> bool:
            for i in range(1, n + 1):
                r = run_call(runtime, label, prompt, i)
                calls.append(r)
                _write(report)
                if not r["quiescent_after"]:
                    r["stop_reason"] = "STALE_GENERATION_OR_QUEUE_BLOCKER"
                    return False
            return True

        ok_a = sequence("A_TINY_TEXT", TINY_TEXT_PROMPT, 3)
        ok_b = sequence("B_TINY_JSON", TINY_JSON_PROMPT, 3) if ok_a else False
        ok_c = sequence("C_CURRENT_PLANNING_SHAPED", e_current, 3) if ok_b else False

        def all_ok(label: str) -> bool:
            rows = [c for c in calls if c["class"] == label]
            return len(rows) == 3 and all(c["success"] for c in rows)

        transport_clean = (
            ok_a
            and ok_b
            and ok_c
            and all_ok("A_TINY_TEXT")
            and all_ok("B_TINY_JSON")
            and all_ok("C_CURRENT_PLANNING_SHAPED")
        )
        if transport_clean:
            sequence("D_COMPACT_PLANNING_SHAPED", e_compact, 3)
            report["class_d"] = "RUN"
        else:
            report["class_d"] = "SKIPPED"

        report["total_provider_calls"] = len(calls)
        report["runtime_after"] = digest(scrape())
        return _write(report)
    finally:
        db.close()


def _write(report: Mapping[str, Any]) -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / "provider-call-results.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o664)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

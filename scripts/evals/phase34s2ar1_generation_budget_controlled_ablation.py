"""PHASE34-S2A-R1 — Generation-budget-controlled CURRENT-vs-COMPACT Planning ablation.

Evaluation harness only. No production code is modified, no Plan is executed, no
repair runs and no product row is created. The frozen Phase34-S2A corpus, prompt
builders, parser, normalizers and validator are reused unchanged; the single
experimental control added here is an evaluation-only output-token budget carried
by ``RuntimeInvocationOptions``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evals"))

import httpx  # noqa: E402

import phase34s2a_planning_interface_ablation as S2A  # noqa: E402
from app.services.agents.providers.openai_chat_adapter import (  # noqa: E402
    _strip_thinking,
)
from app.services.agents.runtime_invocation import (
    RuntimeInvocationOptions,
)  # noqa: E402

METRICS_URL = "http://ai-gateway:8000/metrics"

S2A_EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2a"
EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2a-r1"

# Section 3: smallest practical budget above the measured S2R Fixture E high
# watermark of 3,236 generated tokens, identical for both variants.
EVALUATION_MAX_GENERATED_TOKENS = 5000
# Section 4: 5000 tokens / 16 tok-per-sec conservative floor = 312.5s generation
# allowance, plus bounded TTFT and transport margin. Experiment envelope only.
EVALUATION_DEADLINE_SECONDS = 330
CONSERVATIVE_DECODE_RATE = 16.0

MAX_TOTAL_PROVIDER_CALLS = 12

_NUM = re.compile(r"^([a-zA-Z_:][^\s{]*)(\{[^}]*\})?\s+([-0-9.eE+naN]+)$")

_COUNTER_KEYS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_prompt_tokens_count",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_generation_tokens_count",
    "vllm:request_generation_tokens_sum",
    "vllm:num_preemptions_total",
)
_FINISH_REASONS = ("stop", "length", "abort", "error", "repetition")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scrape() -> dict[str, float]:
    out: dict[str, float] = {}
    with httpx.Client(timeout=15) as client:
        body = client.get(METRICS_URL).text
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = _NUM.match(line.strip())
        if not match:
            continue
        try:
            out[match.group(1) + (match.group(2) or "")] = float(match.group(3))
        except ValueError:
            continue
    return out


def _one(snapshot: Mapping[str, float], name: str) -> Optional[float]:
    for key, value in snapshot.items():
        if key.split("{")[0] == name:
            return value
    return None


def _digest_counters(snapshot: Mapping[str, float]) -> dict[str, Any]:
    digest: dict[str, Any] = {
        key.split(":")[-1]: _one(snapshot, key) for key in _COUNTER_KEYS
    }
    for reason in _FINISH_REASONS:
        for key, value in snapshot.items():
            if key.startswith("vllm:request_success_total") and (
                f'finished_reason="{reason}"' in key
            ):
                digest[f"finish_{reason}"] = value
    return digest


def _quiescent(snapshot: Mapping[str, float]) -> bool:
    return (_one(snapshot, "vllm:num_requests_running") or 0) == 0 and (
        _one(snapshot, "vllm:num_requests_waiting") or 0
    ) == 0


def _wait_quiescent(limit_seconds: float = 120.0) -> tuple[bool, dict[str, Any]]:
    deadline = time.monotonic() + limit_seconds
    snapshot = _scrape()
    while not _quiescent(snapshot) and time.monotonic() < deadline:
        time.sleep(3)
        snapshot = _scrape()
    return _quiescent(snapshot), _digest_counters(snapshot)


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in after.items():
        prior = before.get(key)
        if isinstance(value, (int, float)) and isinstance(prior, (int, float)):
            out[key] = round(value - prior, 6)
    return out


def _budget_options() -> RuntimeInvocationOptions:
    return RuntimeInvocationOptions(
        timeout_seconds=EVALUATION_DEADLINE_SECONDS,
        max_output_tokens=EVALUATION_MAX_GENERATED_TOKENS,
    )


async def _invoke(
    runtime: Any, prompt: str, fixture_id: str, variant: str
) -> dict[str, Any]:
    return await S2A.PlannerService._execute_task_with_planning_lock(
        runtime,
        prompt,
        timeout_seconds=EVALUATION_DEADLINE_SECONDS,
        reuse_task_session=False,
        diagnostic_label="PLANNING",
        diagnostic_metadata={
            "phase": "PHASE34-S2A-R1",
            "fixture_id": fixture_id,
            "variant": variant,
            "planning_attempt": "initial",
            "repairs_allowed": False,
            "execution_allowed": False,
        },
        invocation_options=_budget_options(),
    )


def _verify_corpus(
    freeze: Mapping[str, Any], contexts: Mapping[str, Any]
) -> dict[str, Any]:
    """Sections 7 and 8: prove the corpus and both prompt variants are the S2A ones."""

    stores = {
        variant: json.loads(
            (S2A_EVIDENCE / f"{variant.lower()}-results.json").read_text(
                encoding="utf-8"
            )
        )
        for variant in S2A.VARIANTS
    }
    fixtures: dict[str, Any] = {}
    mismatches: list[str] = []
    for fixture_id in sorted(contexts):
        context = contexts[fixture_id]
        entry: dict[str, Any] = {
            "workspace_digest": S2A._workspace_content_digest(context.workspace),
            "semantic_input_digest": context.semantic_input_digest,
            "variants": {},
        }
        for variant in S2A.VARIANTS:
            rebuilt = context.prompts[variant]["provider"]
            row = [
                item
                for item in stores[variant]["results"]
                if str(item["fixture_id"]) == fixture_id
            ][0]
            expected = row["prompt"]["final_provider_bound_prompt_sha256"]
            actual = _sha256(rebuilt)
            entry["variants"][variant] = {
                "s2a_provider_prompt_sha256": expected,
                "rebuilt_provider_prompt_sha256": actual,
                "byte_identical": actual == expected,
                "prompt_chars": len(rebuilt),
                "s2a_semantic_input_digest": row.get("semantic_input_digest"),
                "semantic_input_digest_match": row.get("semantic_input_digest")
                == context.semantic_input_digest,
            }
            if actual != expected:
                mismatches.append(f"{variant}:{fixture_id}")
            if row.get("semantic_input_digest") != context.semantic_input_digest:
                mismatches.append(f"{variant}:{fixture_id}:semantic_input")
            if row.get("workspace_digest_before") != entry["workspace_digest"]:
                mismatches.append(f"{variant}:{fixture_id}:workspace")
        fixtures[fixture_id] = entry
    return {
        "schema_version": "phase34-s2a-r1-fixture-freeze-verification/1",
        "s2a_runtime_freeze_digest": freeze["runtime_freeze_digest"],
        "s2a_evaluation_script_sha256": freeze["evaluation_script_sha256"],
        "workspace_root": str(freeze["workspace_root"]),
        "fixtures": fixtures,
        "mismatches": sorted(set(mismatches)),
        "experiment_valid": not mismatches,
        "f2_evidence_differed": bool(mismatches),
    }


def _classify_transport(
    *,
    error: Optional[str],
    delta: Mapping[str, Any],
    first_token: bool,
) -> tuple[str, dict[str, Any]]:
    """Section 2: never call a healthy mid-generation cutoff a transport failure."""

    finish_length = float(delta.get("finish_length") or 0)
    finish_stop = float(delta.get("finish_stop") or 0)
    finish_error = float(delta.get("finish_error") or 0)
    if finish_error:
        return "PROVIDER_SERVER_FAILURE", {"server_finish_error": finish_error}
    if finish_length >= 1:
        return "GENERATION_BUDGET_EXCEEDED", {"finish_reason": "length"}
    if error is not None:
        if first_token:
            # Generation was healthy and in flight when the evaluation deadline
            # fired. This is a budget event, not a transport fault.
            return "GENERATION_BUDGET_EXCEEDED", {
                "finish_reason": "deadline_while_generating"
            }
        return "PROVIDER_TRANSPORT_FAILURE", {"finish_reason": "no_generation"}
    if finish_stop >= 1:
        return "COMPLETED", {"finish_reason": "stop"}
    return "PROVIDER_TRANSPORT_FAILURE", {"finish_reason": "unaccounted"}


def _run_cell(
    runtime: Any, context: Any, fixture_id: str, variant: str
) -> dict[str, Any]:
    prompt = context.prompts[variant]["provider"]
    before_workspace = S2A._workspace_content_digest(context.workspace)
    before = _digest_counters(_scrape())
    started = time.monotonic()
    error = None
    raw_output = ""
    try:
        response = asyncio.run(_invoke(runtime, prompt, fixture_id, variant))
        latency_ms = round((time.monotonic() - started) * 1000)
        # exact_contract suppresses the adapter's own reasoning strip; apply the
        # identical production function here so the candidate text matches what
        # ordinary Planning would have received.
        raw_output = _strip_thinking(str(response.get("output") or ""))
    except Exception as exc:  # noqa: BLE001
        latency_ms = round((time.monotonic() - started) * 1000)
        error = f"{type(exc).__name__}: {str(exc)[:400]}"

    quiet, after = _wait_quiescent()
    delta = _delta(before, after)
    after_workspace = S2A._workspace_content_digest(context.workspace)
    if before_workspace != after_workspace:
        raise RuntimeError(f"Planning call mutated frozen fixture {fixture_id}")

    generated = float(delta.get("generation_tokens_total") or 0)
    decode_seconds = float(delta.get("request_decode_time_seconds_sum") or 0)
    ttft_count = float(delta.get("time_to_first_token_seconds_count") or 0)
    ttft_sum = float(delta.get("time_to_first_token_seconds_sum") or 0)
    prompt_tokens = float(delta.get("request_prompt_tokens_sum") or 0)
    outcome, outcome_detail = _classify_transport(
        error=error, delta=delta, first_token=ttft_count >= 1
    )

    cell: dict[str, Any] = {
        "fixture_id": fixture_id,
        "fixture_name": context.spec.name,
        "variant": variant,
        "provider_call_attempts": 1,
        "generation_outcome": outcome,
        "generation_outcome_detail": outcome_detail,
        "generation_budget_exceeded": outcome == "GENERATION_BUDGET_EXCEEDED",
        "transport_error": error if outcome == "PROVIDER_TRANSPORT_FAILURE" else None,
        "client_error": error,
        "server_error": outcome == "PROVIDER_SERVER_FAILURE",
        "raw_provider_response": raw_output,
        "raw_provider_response_sha256": _sha256(raw_output),
        "prompt": context.prompts[variant]["record"],
        "semantic_input_digest": context.semantic_input_digest,
        "workspace_digest_before": before_workspace,
        "workspace_digest_after": after_workspace,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens or None,
        "generated_tokens": generated,
        "visible_output_chars": len(raw_output),
        "reasoning_to_visible_ratio": (
            round(generated / max(len(raw_output) / 4.0, 1.0), 2) if generated else None
        ),
        "time_to_first_token_ms": round(ttft_sum * 1000) if ttft_count else None,
        "generation_duration_ms": (
            round(decode_seconds * 1000) if decode_seconds else None
        ),
        "tokens_per_second": (
            round(generated / decode_seconds, 2) if decode_seconds else None
        ),
        "finish_reason": outcome_detail.get("finish_reason"),
        "server_preemptions": delta.get("num_preemptions_total"),
        "quiescent_after": quiet,
        "counter_delta": delta,
    }

    if outcome == "COMPLETED":
        cell.update(S2A._analyze_candidate(context, raw_output))
        cell["semantic_adjudicable"] = True
    else:
        cell.update(
            {
                "parse_success": False,
                "plan": [],
                "initial_plan_valid": False,
                "repair_required": False,
                "validation": None,
                "validation_status": "not_generated",
                "validator_finding_codes": [],
                "plan_step_count": 0,
                "expected_files": [],
                "mutating_paths": [],
                "verification_commands": [],
                "task_requirement_checklist": {},
                "task_requirement_recall": None,
                "hallucinated_paths": 0,
                "unsupported_operations": 0,
                "wrong_existing_new_classification": 0,
                "verification_correctness": "NOT_ADJUDICABLE",
                "plan_usefulness": "NOT_ADJUDICABLE",
                "exact_user_constraint_recall": None,
                "primary_failure_class": outcome,
                "secondary_failure_notes": ["NO_COMPLETE_CANDIDATE"],
                "semantic_adjudicable": False,
            }
        )
    return cell


def _adjudicate_cell(cell: dict[str, Any], spec: Any) -> None:
    """Re-own semantic metrics on the raw parsed provider Plan (S2A ownership)."""

    if not cell.get("semantic_adjudicable"):
        return
    handler = S2A.EnhancedErrorHandler()
    raw_output = str(cell.get("raw_provider_response") or "")
    success, parsed, _ = handler.attempt_json_parsing(raw_output, context="planning")
    raw_plan = S2A.extract_plan_steps(parsed) if success else None
    if raw_plan is None:
        cell["primary_failure_class"] = "PARSE_FAILURE"
        cell["secondary_failure_notes"] = ["SCHEMA"]
        cell["task_requirement_recall"] = None
        cell["semantic_adjudicable"] = False
        return
    raw_plan = list(raw_plan)
    checklist = S2A._semantic_checklist(spec, raw_plan, raw_output)
    recall = sum(bool(value) for value in checklist.values()) / len(checklist)
    _, raw_mutating, raw_operations = S2A._plan_paths(raw_plan)
    raw_verifications = S2A._verification_commands(raw_plan)
    verification = S2A._verification_correctness(spec, raw_plan)
    usefulness = (
        "PASS"
        if recall == 1.0 and bool(raw_mutating) and verification == "PASS"
        else "PARTIAL" if recall >= 0.5 and bool(raw_mutating) else "FAIL"
    )
    validation = cell.get("validation") or {}
    accepted = validation.get("status") == "accepted"
    primary, secondary = S2A._classify_result(
        pipeline={
            "parse_success": True,
            "normalization_error": cell.get("normalization_error"),
            "validation": validation,
        },
        recall=recall,
        usefulness=usefulness,
        hallucinated_paths=int(cell.get("hallucinated_paths") or 0),
        unsupported_operations=sum(
            1 for operation in raw_operations if operation not in S2A.SUPPORTED_FILE_OPS
        ),
        wrong_classification=int(cell.get("wrong_existing_new_classification") or 0),
    )
    normalized_verifications = list(cell.get("verification_commands") or [])
    cell.update(
        {
            "initial_plan_valid": accepted,
            "repair_required": not accepted,
            "task_requirement_checklist": checklist,
            "task_requirement_recall": recall,
            "verification_correctness": verification,
            "plan_usefulness": usefulness,
            "exact_user_constraint_recall": (
                S2A.EXACT_E_VERIFICATION in raw_verifications
                if cell["fixture_id"] == "E"
                else None
            ),
            "primary_failure_class": primary if primary else "SUCCESS_VALID_PLAN",
            "secondary_failure_notes": secondary,
            "raw_parsed_plan": raw_plan,
            "raw_plan_verification_commands": raw_verifications,
            "normalizer_semantic_drift": bool(
                cell["fixture_id"] == "E"
                and S2A.EXACT_E_VERIFICATION in raw_verifications
                and S2A.EXACT_E_VERIFICATION not in normalized_verifications
            ),
            "provider_plan_verification": raw_verifications,
            "normalized_plan_verification": normalized_verifications,
        }
    )
    if primary is None and not accepted:
        cell["primary_failure_class"] = "SUCCESS_INVALID_PLAN"


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if r.get("semantic_adjudicable")]
    recalls = [
        float(r["task_requirement_recall"])
        for r in complete
        if r.get("task_requirement_recall") is not None
    ]
    generated = [
        float(r["generated_tokens"]) for r in rows if r.get("generated_tokens")
    ]
    return {
        "eos_count": len(complete),
        "budget_exceeded_count": sum(
            1 for r in rows if r.get("generation_budget_exceeded")
        ),
        "real_transport_failures": sum(
            1
            for r in rows
            if r.get("primary_failure_class") == "PROVIDER_TRANSPORT_FAILURE"
        ),
        "server_failures": sum(
            1
            for r in rows
            if r.get("primary_failure_class") == "PROVIDER_SERVER_FAILURE"
        ),
        "initial_valid_count": sum(1 for r in rows if r.get("initial_plan_valid")),
        "complete_cell_mean_requirement_recall": (
            round(sum(recalls) / len(recalls), 4) if recalls else None
        ),
        "total_hallucinated_paths": sum(
            int(r.get("hallucinated_paths") or 0) for r in complete
        ),
        "total_contract_failures": sum(
            1
            for r in complete
            if r.get("primary_failure_class") == "ORCHESTRATOR_CONTRACT_FAILURE"
        ),
        "total_harness_false_positives": sum(
            1
            for r in complete
            if r.get("primary_failure_class") == "HARNESS_FALSE_POSITIVE"
        ),
        "prompt_tokens_total": sum(
            int(r["prompt"].get("prompt_tokens_approx") or 0) for r in rows
        ),
        "generated_tokens_by_fixture": {
            r["fixture_id"]: r.get("generated_tokens") for r in rows
        },
        "mean_generated_tokens": (
            round(statistics.mean(generated), 1) if generated else None
        ),
        "median_generated_tokens": (
            round(statistics.median(generated), 1) if generated else None
        ),
    }


def run() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(
        (S2A_EVIDENCE / "fixture-freeze.json").read_text(encoding="utf-8")
    )
    db = S2A.SessionLocal()
    try:
        contexts = S2A._rebuild_contexts(db, freeze)
        verification = _verify_corpus(freeze, contexts)
        S2A._write_json(EVIDENCE / "fixture-freeze-verification.json", verification)
        if not verification["experiment_valid"]:
            print(
                "EXPERIMENT_VALID = NO — frozen corpus mismatch",
                verification["mismatches"],
            )
            return 1

        runtime = S2A.create_agent_runtime(
            db, None, None, role=S2A.BackendRole.PLANNING
        )
        metadata = runtime.get_backend_metadata()
        identity = {
            "backend": metadata.get("backend"),
            "model": metadata.get("model_family"),
            "profile": metadata.get("adaptation_profile"),
        }
        expected_identity = {
            "backend": freeze["runtime_freeze"]["planning_backend"],
            "model": freeze["runtime_freeze"]["planning_model"],
            "profile": freeze["runtime_freeze"]["planning_adaptation_profile"],
        }
        if identity != expected_identity:
            raise RuntimeError(
                f"Runtime identity changed: {identity} != {expected_identity}"
            )

        S2A._write_json(
            EVIDENCE / "generation-budget.json",
            {
                "schema_version": "phase34-s2a-r1-generation-budget/1",
                "evaluation_max_generated_tokens": EVALUATION_MAX_GENERATED_TOKENS,
                "evaluation_deadline_seconds": EVALUATION_DEADLINE_SECONDS,
                "budget_identical_for_both_variants": True,
                "f3_budget_differed": False,
                "enforcement_method": (
                    "A. evaluation-only RuntimeInvocationOptions(max_output_tokens=5000, "
                    "timeout_seconds=330) passed through the existing public "
                    "execute_task(invocation_options=...) parameter; no production file changed"
                ),
                "budget_derivation": {
                    "s2r_fixture_e_high_watermark_tokens": 3236,
                    "chosen_budget_tokens": EVALUATION_MAX_GENERATED_TOKENS,
                    "conservative_decode_rate_tokens_per_second": CONSERVATIVE_DECODE_RATE,
                    "generation_allowance_seconds": EVALUATION_MAX_GENERATED_TOKENS
                    / CONSERVATIVE_DECODE_RATE,
                    "ttft_and_transport_margin_seconds": EVALUATION_DEADLINE_SECONDS
                    - EVALUATION_MAX_GENERATED_TOKENS / CONSERVATIVE_DECODE_RATE,
                    "note": (
                        "Experiment envelope only. This is not a proposed production "
                        "timeout. The server-side max_tokens limit is the primary "
                        "control; the deadline is a backstop that should not fire."
                    ),
                },
                "wire_payload_delta_vs_ordinary_planning": {"max_tokens": 5000},
                "adapter_behaviour_deltas": [
                    "exact_contract=True suppresses the adapter's _strip_thinking(); the "
                    "harness applies the identical production function to the candidate",
                    "transport wall is the explicit 330s rather than timeout+30",
                ],
                "runtime_identity": identity,
                "temperature": freeze["runtime_freeze"]["temperature"],
                "call_order": [list(pair) for pair in S2A.CALL_ORDER],
                "max_total_provider_calls": MAX_TOTAL_PROVIDER_CALLS,
            },
        )

        quiet, before = _wait_quiescent()
        if not quiet:
            print("Provider not quiescent before first call")
            return 1

        rows: dict[str, list[dict[str, Any]]] = {v: [] for v in S2A.VARIANTS}
        calls = 0
        specs = S2A._fixtures()
        for fixture_id, variant in S2A.CALL_ORDER:
            if calls >= MAX_TOTAL_PROVIDER_CALLS:
                raise RuntimeError("Provider call budget exhausted")
            calls += 1
            cell = _run_cell(runtime, contexts[fixture_id], fixture_id, variant)
            _adjudicate_cell(cell, specs[fixture_id])
            rows[variant].append(cell)
            for variant_name in S2A.VARIANTS:
                S2A._write_json(
                    EVIDENCE / f"{variant_name.lower()}-results.json",
                    {
                        "schema_version": "phase34-s2a-r1-results/1",
                        "variant": variant_name,
                        "runtime_freeze_digest": freeze["runtime_freeze_digest"],
                        "evaluation_max_generated_tokens": EVALUATION_MAX_GENERATED_TOKENS,
                        "evaluation_deadline_seconds": EVALUATION_DEADLINE_SECONDS,
                        "total_provider_calls": len(rows[variant_name]),
                        "results": rows[variant_name],
                    },
                )
            print(
                f"{fixture_id}:{variant} {cell['generation_outcome']} "
                f"gen={cell['generated_tokens']} lat={cell['latency_ms']}ms "
                f"class={cell['primary_failure_class']} "
                f"recall={cell['task_requirement_recall']}",
                flush=True,
            )
            if not cell["quiescent_after"]:
                raise RuntimeError("Server did not return to quiescence; stopping")

        _write_comparison(freeze, rows, verification, before, calls)
        return 0
    finally:
        db.close()


def _write_comparison(
    freeze: Mapping[str, Any],
    rows: Mapping[str, list[dict[str, Any]]],
    verification: Mapping[str, Any],
    before: Mapping[str, Any],
    calls: int,
) -> None:
    indexed = {
        variant: {row["fixture_id"]: row for row in rows[variant]}
        for variant in S2A.VARIANTS
    }
    aggregates = {variant: _aggregate(rows[variant]) for variant in S2A.VARIANTS}
    comparison_rows = []
    for fixture_id in sorted(indexed["CURRENT"]):
        current = indexed["CURRENT"][fixture_id]
        compact = indexed["COMPACT"][fixture_id]
        paired = bool(
            current.get("semantic_adjudicable") and compact.get("semantic_adjudicable")
        )
        comparison_rows.append(
            {
                "fixture_id": fixture_id,
                "fixture_name": current["fixture_name"],
                "paired_complete": paired,
                "current_outcome": current["generation_outcome"],
                "compact_outcome": compact["generation_outcome"],
                "current_failure_class": current["primary_failure_class"],
                "compact_failure_class": compact["primary_failure_class"],
                "current_semantic_recall": current["task_requirement_recall"],
                "compact_semantic_recall": compact["task_requirement_recall"],
                "current_valid": current["initial_plan_valid"],
                "compact_valid": compact["initial_plan_valid"],
                "current_generated_tokens": current["generated_tokens"],
                "compact_generated_tokens": compact["generated_tokens"],
                "current_prompt_tokens": current["prompt"].get("prompt_tokens_approx"),
                "compact_prompt_tokens": compact["prompt"].get("prompt_tokens_approx"),
                "current_normalizer_drift": current.get("normalizer_semantic_drift"),
                "compact_normalizer_drift": compact.get("normalizer_semantic_drift"),
            }
        )
    paired_rows = [row for row in comparison_rows if row["paired_complete"]]
    current_prompt = aggregates["CURRENT"]["prompt_tokens_total"]
    compact_prompt = aggregates["COMPACT"]["prompt_tokens_total"]
    S2A._write_json(
        EVIDENCE / "comparison.json",
        {
            "schema_version": "phase34-s2a-r1-comparison/1",
            "aggregates": aggregates,
            "comparison_rows": comparison_rows,
            "paired_complete_fixtures": [row["fixture_id"] for row in paired_rows],
            "prompt_reduction_percent": (
                round(100.0 * (current_prompt - compact_prompt) / current_prompt, 1)
                if current_prompt
                else None
            ),
            "total_provider_calls": calls,
            "counters_before_first_call": dict(before),
            "fixture_freeze_verification": {
                "experiment_valid": verification["experiment_valid"],
                "mismatches": verification["mismatches"],
            },
            "adjudication": {
                "semantic_scoring_basis": "parsed provider Plan before deterministic Planning normalization",
                "initial_valid_basis": "existing ValidatorResult.status == accepted",
                "budget_exceeded_cells_are_not_scored_zero": True,
                "generation_budget_tokens": EVALUATION_MAX_GENERATED_TOKENS,
                "runtime_freeze_digest": freeze["runtime_freeze_digest"],
            },
        },
    )


if __name__ == "__main__":
    raise SystemExit(run())

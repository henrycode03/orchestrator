"""PHASE34-NTP1 — fresh Fixture-B non-thinking Planning quality gate.

Evaluation only: five first-pass NON_THINKING calls, no control, repair, Plan
execution, product row, or production mutation. The S2X fixture builder,
provider call path, parser, normalizer, validator, and envelope are reused.

Usage:
    stage1   provider-free frozen-input and wire proof
    run      exactly five fresh calls (resumable after process interruption)
    evidence build compact evidence after manual adjudication is recorded
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evals"))

import phase34s2a_planning_interface_ablation as S2A  # noqa: E402
import phase34s2u_provider_reasoning_bound_probe as S2U  # noqa: E402
import phase34s2x_thinking_vs_non_thinking_planning as S2X  # noqa: E402

PHASE = "PHASE34-NTP1"
RUNS = 5
MAX_PROVIDER_CALLS = 5
EXPECTED_PROMPT_SHA256 = (
    "5fe9b26afb8a9f334af4b32c58a010f5caab04d331e8d4c9dd02745f67d69418"
)
CAPTURE_DIR = Path("/tmp/phase34-ntp1-20260831-1743")
EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-ntp1"
S2X_WIRE = ROOT / (
    "docs/roadmap/reports/evidence/phase34-s2x/capability-wire-proof.json"
)
S2A_FREEZE = ROOT / ("docs/roadmap/reports/evidence/phase34-s2a/fixture-freeze.json")
ALIGNMENT = EVIDENCE / "validator-manual-alignment.json"


def _sha256(value: str) -> str:
    return S2X.S2T._sha256(value)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def stage1() -> int:
    """Prove S2X prompt/input and binary-disable wire identity before calls."""

    db = S2A.SessionLocal()
    try:
        runtime, context, verification = S2U._prepare(db)
        S2U._RUNTIME = runtime
        prompt = context.prompts[S2X.PRIMARY_VARIANT]["provider"]
        current_sha = _sha256(prompt)
        current_control = S2U._dry_run_payload(S2X._options("CONTROL"), prompt)
        current_treatment = S2U._dry_run_payload(S2X._options("NON_THINKING"), prompt)
        current_wire = S2X._wire_proof(current_control, current_treatment)
        prior_wire = json.loads(S2X_WIRE.read_text(encoding="utf-8"))
        freeze = json.loads(S2A_FREEZE.read_text(encoding="utf-8"))
        fixture = freeze["fixtures"]["B"]

        prompt_match = bool(
            current_sha == EXPECTED_PROMPT_SHA256
            and current_sha
            == fixture["prompts"]["COMPACT"]["final_provider_bound_prompt_sha256"]
            and verification["variants"]["COMPACT"]["byte_identical"]
        )
        wire_match = bool(
            current_wire["wire_delta_only_reasoning_mode"]
            and current_wire["added_in_non_thinking"]
            == prior_wire["added_in_non_thinking"]
            and current_wire["wire_delta"] == prior_wire["wire_delta"]
        )
        input_match = bool(
            verification["prompt_freeze_verified"]
            and verification["corpus_gate"]["corpus_verified"]
            and not verification["mismatches"]
            and verification["runtime_identity"]
            == prior_wire["fixture_freeze"]["runtime_identity"]
            and verification["temperature"]
            == prior_wire["fixture_freeze"]["temperature"]
        )
        proof = {
            "schema_version": "phase34-ntp1-frozen-input-proof/1",
            "provider_calls": 0,
            "fixture": "B — GROUNDED_EXISTING_EDIT",
            "variant": "COMPACT",
            "task_text_sha256": fixture["task_text_sha256"],
            "source_materialization_digest": fixture["source_materialization_digest"],
            "workspace_content_digest": fixture["workspace_content_digest"],
            "project_context_digest": fixture["project_context_digest"],
            "semantic_input_digest": fixture["semantic_input_digest"],
            "intent_mode": fixture["intent_mode"],
            "runtime_identity": verification["runtime_identity"],
            "temperature": verification["temperature"],
            "outer_envelope": {
                "max_generated_tokens": S2X.OUTER_MAX_GENERATED_TOKENS,
                "deadline_seconds": S2X.OUTER_DEADLINE_SECONDS,
            },
            "provider_bound_prompt_sha256": current_sha,
            "s2x_provider_bound_prompt_sha256": EXPECTED_PROMPT_SHA256,
            "prompt_sha256_match": prompt_match,
            "input_freeze_match": input_match,
            "wire_delta_matches_s2x_non_thinking": wire_match,
            "wire_delta": current_wire["wire_delta"],
            "added_in_non_thinking": current_wire["added_in_non_thinking"],
            "supported_option": "RuntimeInvocationOptions.reasoning_enabled=False",
            "allowlist_bypassed_or_mutated": False,
            "experiment_pre_call_gate": bool(
                prompt_match and input_match and wire_match
            ),
        }
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        S2A._write_json(EVIDENCE / "frozen-input-proof.json", proof)
        print(json.dumps(proof, indent=1))
        return 0 if proof["experiment_pre_call_gate"] else 1
    finally:
        db.close()


def run() -> int:
    """Issue the fixed five-call treatment sequence, never adaptively."""

    proof = json.loads(
        (EVIDENCE / "frozen-input-proof.json").read_text(encoding="utf-8")
    )
    if not proof.get("experiment_pre_call_gate"):
        raise SystemExit("NTP1 pre-call gate did not pass")
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    S2X.CAPTURE_DIR = CAPTURE_DIR
    db = S2A.SessionLocal()
    try:
        runtime, context, verification = S2U._prepare(db)
        if _sha256(context.prompts["COMPACT"]["provider"]) != EXPECTED_PROMPT_SHA256:
            raise SystemExit("Prompt drift immediately before provider calls")
        S2A._write_json(CAPTURE_DIR / "freeze.json", verification)
        for index in range(1, RUNS + 1):
            path = CAPTURE_DIR / f"cell-non_thinking-{index}.json"
            if path.exists():
                print(f"non_thinking-{index}: already captured, skipping", flush=True)
                continue
            spent = len(list(CAPTURE_DIR.glob("cell-non_thinking-*.json")))
            if spent >= MAX_PROVIDER_CALLS:
                raise SystemExit(f"MAX_PROVIDER_CALLS={MAX_PROVIDER_CALLS} spent")
            S2X._run_call(runtime, context, "NON_THINKING", index)
        return 0
    finally:
        db.close()


def _load_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(1, RUNS + 1):
        path = CAPTURE_DIR / f"cell-non_thinking-{index}.json"
        if not path.exists():
            raise SystemExit(f"missing call result: {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _compact_row(cell: Mapping[str, Any], manual: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run": cell["label"],
        "generation_outcome": cell["generation_outcome"],
        "parse_success": cell.get("parse_success"),
        "task_requirement_recall": cell.get("task_requirement_recall"),
        "plan_usefulness": manual["plan_usefulness"],
        "verification_correctness": manual["verification_correctness"],
        "mutation_present_when_required": manual["mutation_present_when_required"],
        "expected_files_correct": manual["expected_files_correct"],
        "source_grounded": manual["source_grounded"],
        "hallucinated_paths": cell.get("hallucinated_paths"),
        "unsupported_operations": cell.get("unsupported_operations"),
        "wrong_existing_new_classification": cell.get(
            "wrong_existing_new_classification"
        ),
        "validator_status": cell.get("validation_status"),
        "validator_findings": cell.get("validator_finding_codes"),
        "manual_coherence": manual["manual_coherence"],
        "validator_manual_alignment": manual["validator_manual_alignment"],
        "ambiguity_category": manual["ambiguity_category"],
        "actual_mutation": manual["actual_mutation"],
        "test_behavior": manual["test_behavior"],
        "verification_command": manual["verification_command"],
        "vic1_exact_contradiction_present": manual["vic1_exact_contradiction_present"],
        "generated_tokens": cell["generated_tokens"],
        "latency_ms": cell["latency_ms"],
        "budget_exceeded": cell["generation_outcome"] == "GENERATION_BUDGET_EXCEEDED",
    }


def evidence() -> int:
    rows = _load_rows()
    alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    manual_by_run = {row["run"]: row for row in alignment["rows"]}
    compact = [_compact_row(row, manual_by_run[row["label"]]) for row in rows]
    accepted = sum(row["validator_status"] == "accepted" for row in compact)
    parsed = sum(row["parse_success"] is True for row in compact)
    coherent = sum(row["manual_coherence"] == "COHERENT" for row in compact)
    accepted_coherent = sum(
        row["validator_status"] == "accepted" and row["manual_coherence"] == "COHERENT"
        for row in compact
    )
    verification_correct = sum(
        row["verification_correctness"] == "PASS" for row in compact
    )
    alignment_count = sum(row["validator_manual_alignment"] for row in compact)
    accepted_broken = [
        row["run"]
        for row in compact
        if row["validator_status"] == "accepted"
        and row["manual_coherence"] != "COHERENT"
    ]
    exact_rows = [row for row in compact if row["vic1_exact_contradiction_present"]]
    vic1_regressions = [
        row["run"]
        for row in exact_rows
        if row["validator_status"] == "accepted"
        or "plan_verification_internal_contradiction" not in row["validator_findings"]
    ]
    new_false_negative = bool(
        accepted_broken and any(run not in vic1_regressions for run in accepted_broken)
    )
    provider_failure = any(
        row["generation_outcome"] not in ("COMPLETED", "GENERATION_BUDGET_EXCEEDED")
        for row in compact
    )
    gate_checks = {
        "parse_success_at_least_4_of_5": parsed >= 4,
        "manually_coherent_at_least_4_of_5": coherent >= 4,
        "validator_accepted_coherent_at_least_4_of_5": accepted_coherent >= 4,
        "verification_correct_at_least_4_of_5": verification_correct >= 4,
        "no_validator_accepted_self_contradiction": not accepted_broken,
        "no_hallucinated_paths": not any(row["hallucinated_paths"] for row in compact),
        "no_unsupported_operations": not any(
            row["unsupported_operations"] for row in compact
        ),
        "no_wrong_existing_new_classification": not any(
            row["wrong_existing_new_classification"] for row in compact
        ),
        "vic1_exact_class_held_when_present": not vic1_regressions,
    }
    gate_passed = all(gate_checks.values())
    if provider_failure:
        decision = "E. FRESH_SAMPLE_PROVIDER_FAILURE_PREVENTED_ADJUDICATION"
    elif vic1_regressions:
        decision = "D. VIC1_REGRESSION_OBSERVED"
    elif new_false_negative:
        decision = "C. VALIDATOR_FALSE_NEGATIVE_STILL_PREVENTS_TRUSTWORTHY_SCORING"
    elif gate_passed:
        decision = "A. NON_THINKING_FIXTURE_B_PROMOTION_GATE_PASSED"
    else:
        decision = "B. NON_THINKING_FIXTURE_B_QUALITY_REMAINS_BELOW_GATE"

    generated = [float(row["generated_tokens"]) for row in compact]
    latencies = [float(row["latency_ms"]) for row in compact]
    summary = {
        "schema_version": "phase34-ntp1-fresh-results/1",
        "fresh_runs": RUNS,
        "total_provider_calls": RUNS,
        "replacement_calls": 0,
        "rows": compact,
        "counts": {
            "accepted": accepted,
            "parse_success": parsed,
            "manually_coherent": coherent,
            "validator_accepted_coherent": accepted_coherent,
            "verification_correct": verification_correct,
            "budget_exceeded": sum(row["budget_exceeded"] for row in compact),
            "validator_manual_alignment": alignment_count,
            "vic1_findings": sum(
                "plan_verification_internal_contradiction" in row["validator_findings"]
                for row in compact
            ),
        },
        "generated_tokens": _stats(generated),
        "latency_ms": _stats(latencies),
    }
    comparison = {
        "schema_version": "phase34-ntp1-historical-fresh-comparison/1",
        "historical_corrected_non_thinking": "3/5",
        "fresh_corrected_non_thinking": f"{coherent}/5",
        "cumulative_corrected_non_thinking": f"{3 + coherent}/10",
        "iid_statistical_sample_claimed": False,
        "context_only": True,
    }
    promotion = {
        "schema_version": "phase34-ntp1-promotion-gate/1",
        "decision": decision,
        "checks": gate_checks,
        "gate_passed": gate_passed,
        "accepted_self_contradicting_runs": accepted_broken,
        "new_validator_false_negative_observed": new_false_negative,
        "vic1_regression": bool(vic1_regressions),
        "vic1_regression_runs": vic1_regressions,
        "multi_fixture_confirmation": (
            "AUTHORIZED" if decision.startswith("A.") else "NOT_AUTHORIZED"
        ),
        "production": "NOT_AUTHORIZED",
        "falsification": {
            "f1_wire_drift": False,
            "f2_input_drift": False,
            "f3_acceptance_manual_mismatch": accepted >= 4 and coherent < 4,
            "f4_vic1_durable": bool(exact_rows and not vic1_regressions),
            "f5_new_false_negative": new_false_negative,
            "f6_promotion_signal": gate_passed,
            "f7_below_reliability_gate": coherent <= 3,
            "f8_fast_but_not_correct_enough": bool(
                coherent <= 3
                and len(latencies) == 5
                and statistics.median(latencies) <= 60000
            ),
        },
    }
    S2A._write_json(EVIDENCE / "fresh-results.json", summary)
    S2A._write_json(EVIDENCE / "historical-fresh-comparison.json", comparison)
    S2A._write_json(EVIDENCE / "promotion-gate.json", promotion)
    print(
        json.dumps(
            {"summary": summary, "comparison": comparison, "promotion": promotion},
            indent=1,
        )
    )
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"stage1", "run", "evidence"}:
        print(__doc__)
        return 2
    return {"stage1": stage1, "run": run, "evidence": evidence}[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())

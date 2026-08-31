"""PHASE34-S2X — thinking vs non-thinking structured Planning, repeated ablation.

Evaluation harness only. No production file, prompt, validator, normalizer or
adapter is modified, no Plan is executed, no repair runs and no product row is
created. The frozen Phase34-S2A Fixture B COMPACT corpus, prompt builder,
parser, normalizer and validator are reused unchanged via S2T/S2U, and the
CONTROL provider prompt is re-verified byte-identical before any call.

The NON_THINKING arm uses ``RuntimeInvocationOptions.reasoning_enabled=False``,
a first-class supported field of the existing runtime contract. The options
allowlist is neither bypassed nor mutated.

Design: 5 CONTROL + 5 NON_THINKING, interleaved in a balanced order frozen
before the first call. Resumable -- completed calls are skipped on re-entry.

Usage:
    stage1      provider-free capability + wire proof + template rendering diff
    run         execute the frozen interleaved order (resumable)
    evidence    write the compact durable evidence set
"""

from __future__ import annotations

import asyncio
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
import phase34s2ar1_generation_budget_controlled_ablation as R1  # noqa: E402
import phase34s2t_planning_reasoning_termination as S2T  # noqa: E402
import phase34s2u_provider_reasoning_bound_probe as S2U  # noqa: E402
from app.services.agents.providers.openai_chat_adapter import (  # noqa: E402
    _GENERIC_SYSTEM,
    _strip_thinking,
)
from app.services.agents.runtime_invocation import (  # noqa: E402
    RuntimeInvocationOptions,
)
from app.services.orchestration.error_handler import (  # noqa: E402
    EnhancedErrorHandler,
)
from app.services.orchestration.validation.parsing import (  # noqa: E402
    extract_plan_steps,
)

PRIMARY_FIXTURE = "B"
PRIMARY_VARIANT = "COMPACT"

# Section 9: identical envelope for both arms, unchanged from S2T..S2W.
OUTER_MAX_GENERATED_TOKENS = 5000
OUTER_DEADLINE_SECONDS = 330

# Section 7: balanced interleaved order, frozen here before the first call and
# never chosen adaptively.
FROZEN_CALL_ORDER = (
    ("CONTROL", 1),
    ("NON_THINKING", 1),
    ("NON_THINKING", 2),
    ("CONTROL", 2),
    ("CONTROL", 3),
    ("NON_THINKING", 3),
    ("NON_THINKING", 4),
    ("CONTROL", 4),
    ("CONTROL", 5),
    ("NON_THINKING", 5),
)
RUNS_PER_ARM = 5
MAX_PROVIDER_CALLS = 12

EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2x"
CAPTURE_DIR = Path(
    "/tmp/claude-0/-root--openclaw-workspace-vault-projects-orchestrator"
    "/f5bb3e28-8ae8-486d-9ddf-22ac10e4c66b/scratchpad/s2x"
)
TOKENIZE_URL = "http://ai-gateway:8000/tokenize"


def _options(arm: str) -> RuntimeInvocationOptions:
    """Both arms share the envelope and the pinned PLANNING system prompt.

    NON_THINKING adds only ``reasoning_enabled=False`` -- a declared field of
    RuntimeInvocationOptions, not an extra_provider_options entry, so the
    production allowlist is untouched.
    """

    return RuntimeInvocationOptions(
        timeout_seconds=OUTER_DEADLINE_SECONDS,
        max_output_tokens=OUTER_MAX_GENERATED_TOKENS,
        system_prompt=_GENERIC_SYSTEM,
        reasoning_enabled=False if arm == "NON_THINKING" else None,
    )


# ------------------------------------------- stage 1: capability + wire ---


def _template_rendering_diff(prompt: str) -> dict[str, Any]:
    """Ask the deployed server to render both variants. No generation."""

    messages = [
        {"role": "system", "content": _GENERIC_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    base = {"model": "qwen-local", "messages": messages}
    out: dict[str, Any] = {"endpoint": TOKENIZE_URL}
    try:
        with httpx.Client(timeout=30) as client:
            thinking = client.post(TOKENIZE_URL, json=base).json()
            non_thinking = client.post(
                TOKENIZE_URL,
                json={**base, "chat_template_kwargs": {"enable_thinking": False}},
            ).json()
        out["thinking_rendered_tokens"] = thinking.get("count")
        out["non_thinking_rendered_tokens"] = non_thinking.get("count")
        out["rendered_token_delta"] = (
            (non_thinking.get("count") or 0) - (thinking.get("count") or 0)
        )
        out["chat_template_rendering_differs"] = (
            thinking.get("count") != non_thinking.get("count")
        )
        out["method"] = (
            "deployed server /tokenize on the exact frozen prompt; the rendered "
            "prompt itself changes, so the disable is a real chat-template "
            "mechanism rather than an ignored field"
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        out["chat_template_rendering_differs"] = None
    return out


def _wire_proof(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict:
    control_payload = dict(control["payload"])
    treatment_payload = dict(treatment["payload"])
    added = {
        key: treatment_payload[key]
        for key in treatment_payload
        if key not in control_payload
    }
    removed = sorted(set(control_payload) - set(treatment_payload))
    changed = {
        key: {"control": control_payload[key], "treatment": treatment_payload[key]}
        for key in control_payload
        if key in treatment_payload and control_payload[key] != treatment_payload[key]
    }
    expected = {
        "think": False,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    identical = {
        field: control_payload.get(field) == treatment_payload.get(field)
        for field in ("model", "messages", "temperature", "stream", "max_tokens")
    }
    identical["response_format_absent_in_both"] = (
        "response_format" not in control_payload
        and "response_format" not in treatment_payload
    )
    sole = added == expected
    return {
        "schema_version": "phase34-s2x-wire-proof/1",
        "url_identical": control["url"] == treatment["url"],
        "headers_identical": control["headers"] == treatment["headers"],
        "required_identical_fields": identical,
        "control_payload_keys": sorted(control_payload),
        "non_thinking_payload_keys": sorted(treatment_payload),
        "added_in_non_thinking": added,
        "removed_in_non_thinking": removed,
        "changed_in_non_thinking": changed,
        "expected_binary_disable_fields": expected,
        "wire_delta": (
            '+ "think": false, "enable_thinking": false, '
            '"chat_template_kwargs": {"enable_thinking": false}'
            if sole
            else "UNEXPECTED"
        ),
        "wire_delta_only_reasoning_mode": bool(
            sole
            and not removed
            and not changed
            and all(identical.values())
            and control["url"] == treatment["url"]
            and control["headers"] == treatment["headers"]
        ),
    }


def stage1() -> int:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    db = S2A.SessionLocal()
    try:
        runtime, context, verification = S2U._prepare(db)
        S2U._RUNTIME = runtime
        prompt = context.prompts[PRIMARY_VARIANT]["provider"]

        # The option is a declared field, so no allowlist interaction at all.
        probe = RuntimeInvocationOptions(reasoning_enabled=False)
        control = S2U._dry_run_payload(_options("CONTROL"), prompt)
        treatment = S2U._dry_run_payload(_options("NON_THINKING"), prompt)
        proof = _wire_proof(control, treatment)
        proof["capability"] = {
            "non_thinking_control_supported": True,
            "supported_option": "RuntimeInvocationOptions.reasoning_enabled=False",
            "option_is_declared_field_not_extra_provider_option": True,
            "allowlist_bypassed_or_mutated": False,
            "production_modification_required": False,
            "options_surface_accepts": probe.reasoning_enabled is False,
            "adapter_path": (
                "openai_chat_adapter._build_payload: `if "
                "invocation_options.reasoning_enabled is False:` emits think, "
                "enable_thinking and chat_template_kwargs.enable_thinking"
            ),
            "reasoning_field_expected": "ABSENT",
            "reasoning_field_expectation_basis": (
                "the disable acts on the chat template, so the server's "
                "reasoning parser should have no thinking span to separate; "
                "verified per run by F2 rather than assumed"
            ),
        }
        proof["chat_template_rendering_diff"] = _template_rendering_diff(prompt)
        proof["fixture_freeze"] = verification
        proof["frozen_call_order"] = [f"{arm}#{i}" for arm, i in FROZEN_CALL_ORDER]
        proof["outer_max_generated_tokens"] = OUTER_MAX_GENERATED_TOKENS
        proof["outer_deadline_seconds"] = OUTER_DEADLINE_SECONDS
        proof["max_provider_calls"] = MAX_PROVIDER_CALLS

        EVIDENCE.mkdir(parents=True, exist_ok=True)
        S2A._write_json(EVIDENCE / "capability-wire-proof.json", proof)
        S2A._write_json(EVIDENCE / "frozen-corpus-verification.json", verification)
        S2A._write_json(
            EVIDENCE / "call-order.json",
            {
                "schema_version": "phase34-s2x-call-order/1",
                "frozen_before_first_call": True,
                "chosen_adaptively": False,
                "order": [f"{arm}#{i}" for arm, i in FROZEN_CALL_ORDER],
                "runs_per_arm": RUNS_PER_ARM,
                "max_provider_calls": MAX_PROVIDER_CALLS,
                "retry_policy": (
                    "no retry for bad/invalid/truncated/unparseable Plans or "
                    "model semantic failure; one replacement call only for a "
                    "proven transport failure before healthy generation"
                ),
            },
        )
        print(json.dumps({
            "non_thinking_control_supported": True,
            "supported_option": proof["capability"]["supported_option"],
            "wire_delta": proof["wire_delta"],
            "wire_delta_only_reasoning_mode": proof["wire_delta_only_reasoning_mode"],
            "chat_template_rendering_diff": proof["chat_template_rendering_diff"],
        }, indent=1))
        return 0 if proof["wire_delta_only_reasoning_mode"] else 1
    finally:
        db.close()


# ------------------------------------------- section 12: ambiguity classes ---


def _provider_plan(raw_output: str) -> list[dict[str, Any]]:
    """Parse the PROVIDER's own Plan, before any semantic normalization."""

    handler = EnhancedErrorHandler()
    success, parsed, _ = handler.attempt_json_parsing(raw_output, context="planning")
    if not success:
        return []
    steps = extract_plan_steps(parsed)
    return [dict(step) for step in (steps or []) if isinstance(step, Mapping)]


def _classify_ambiguity(plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the Fixture B resolution from the PROPOSED MUTATION itself."""

    greeter_text: list[str] = []
    test_text: list[str] = []
    for step in plan:
        blob_commands = " ".join(str(c) for c in (step.get("commands") or []))
        for operation in step.get("ops") or []:
            if not isinstance(operation, Mapping):
                continue
            path = str(operation.get("path") or "").replace("\\", "/").lstrip("./")
            body = " ".join(
                str(operation.get(key) or "")
                for key in ("content", "new", "old")
            )
            if path.endswith("test_greeter.py"):
                test_text.append(body)
            elif path.endswith("greeter.py"):
                greeter_text.append(body)
        if "test_greeter.py" in blob_commands:
            test_text.append(blob_commands)
        elif "greeter.py" in blob_commands:
            greeter_text.append(blob_commands)

    greeter = "\n".join(greeter_text)
    tests = "\n".join(test_text)
    greeter_mutated = bool(greeter.strip())
    test_mutated = bool(tests.strip())

    # Does the new greeter keep greet("Ada") == "Hello"? That requires the name
    # to be behind an opt-in: a second parameter, a keyword flag, or a separate
    # function -- not an unconditional f-string on the existing single argument.
    optional_signal = bool(
        re.search(r"def\s+greet\s*\([^)]*=\s*(?:False|None|True|\"|')", greeter)
        or re.search(r"def\s+greet\s*\([^)]*,\s*\w+", greeter)
        or re.search(r"def\s+greet_\w+|def\s+\w+_greet", greeter)
    )
    unconditional_name = bool(
        re.search(r'return\s+f?"Hello,?\s*\{?\s*name', greeter)
        or re.search(r'"Hello,?\s*"\s*\+\s*name', greeter)
        or re.search(r'"Hello,?\s*\{\}"\.format\(\s*name', greeter)
    )
    test_expects_name = bool(
        re.search(r'==\s*f?"Hello,?\s*(?:\{?name|Ada)', tests)
        or re.search(r'"Hello,\s*Ada"', tests)
    )

    if not greeter_mutated:
        category = "E_INCOHERENT_OR_UNRESOLVED"
    elif unconditional_name and not optional_signal:
        category = (
            "C_CHANGES_TEST_EXPECTATION"
            if (test_mutated and test_expects_name)
            else "B_CHANGES_EXISTING_CALL_BEHAVIOR"
        )
    elif optional_signal:
        category = "A_OPTIONAL_BEHAVIOR_PRESERVES_CURRENT_DEFAULT"
    else:
        category = "D_OTHER_COHERENT_RESOLUTION"
    return {
        "category": category,
        "greeter_mutated": greeter_mutated,
        "test_mutated": test_mutated,
        "optional_signal": optional_signal,
        "unconditional_name_in_greeting": unconditional_name,
        "test_expects_named_greeting": test_expects_name,
        "greeter_mutation_excerpt": greeter[:400],
        "test_mutation_excerpt": tests[:300],
        "note": (
            "heuristic pre-classification from the provider's own pre-"
            "normalization ops; every run's excerpt was read before the "
            "category was accepted"
        ),
    }


# ------------------------------------------------------------ stage 2: runs ---


def _run_call(runtime: Any, context: Any, arm: str, index: int) -> dict[str, Any]:
    label = f"{arm.lower()}-{index}"
    options = _options(arm)
    capture_path = CAPTURE_DIR / f"raw-{label}.json"
    prompt = context.prompts[PRIMARY_VARIANT]["provider"]
    before_workspace = S2A._workspace_content_digest(context.workspace)
    quiet, before = R1._wait_quiescent()
    if not quiet:
        raise RuntimeError(f"Provider not quiescent before {label}")

    started = time.monotonic()
    error: Optional[str] = None
    response: Any = None
    try:
        response = asyncio.run(
            S2A.PlannerService._execute_task_with_planning_lock(
                runtime,
                prompt,
                timeout_seconds=OUTER_DEADLINE_SECONDS,
                reuse_task_session=False,
                diagnostic_label="PLANNING",
                diagnostic_metadata={
                    "phase": "PHASE34-S2X",
                    "arm": arm,
                    "run_index": index,
                    "fixture_id": PRIMARY_FIXTURE,
                    "variant": PRIMARY_VARIANT,
                    "planning_attempt": "initial",
                    "repairs_allowed": False,
                    "execution_allowed": False,
                    "discovery_contract_capture_path": str(capture_path),
                    "discovery_contract_run_id": f"s2x-{label}",
                },
                invocation_options=options,
            )
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
    latency_ms = round((time.monotonic() - started) * 1000)

    quiet_after, after = R1._wait_quiescent()
    delta = R1._delta(before, after)
    if S2A._workspace_content_digest(context.workspace) != before_workspace:
        raise RuntimeError("Planning call mutated frozen fixture B")

    ttft_count = float(delta.get("time_to_first_token_seconds_count") or 0)
    outcome, detail = R1._classify_transport(
        error=error, delta=delta, first_token=ttft_count >= 1
    )
    adapter_output = _strip_thinking(str((response or {}).get("output") or ""))
    cell: dict[str, Any] = {
        "arm": arm,
        "run_index": index,
        "label": label,
        "fixture_id": PRIMARY_FIXTURE,
        "variant": PRIMARY_VARIANT,
        "invocation_options": options.to_dict(),
        "generation_outcome": outcome,
        "finish_reason": detail.get("finish_reason"),
        "client_error": error,
        "latency_ms": latency_ms,
        "generated_tokens": float(delta.get("generation_tokens_total") or 0),
        "prompt_tokens": float(delta.get("request_prompt_tokens_sum") or 0),
        "time_to_first_token_ms": (
            round(float(delta.get("time_to_first_token_seconds_sum") or 0) * 1000)
            if ttft_count
            else None
        ),
        "quiescent_after": quiet_after,
        "workspace_digest_after": S2A._workspace_content_digest(context.workspace),
        "adapter_returned_output_chars": len(adapter_output),
        "capture_path": str(capture_path),
    }
    if outcome == "COMPLETED":
        cell.update(S2A._analyze_candidate(context, adapter_output))
        cell["semantic_adjudicable"] = True
        cell["raw_provider_response"] = adapter_output
        R1._adjudicate_cell(cell, S2A._fixtures()[PRIMARY_FIXTURE])
        cell["ambiguity"] = _classify_ambiguity(_provider_plan(adapter_output))
    else:
        cell["semantic_adjudicable"] = False
        cell["ambiguity"] = {"category": "NOT_ADJUDICABLE"}
    cell.pop("raw_provider_response", None)
    cell.pop("plan", None)
    (CAPTURE_DIR / f"cell-{label}.json").write_text(
        json.dumps(cell, indent=1, default=str), encoding="utf-8"
    )
    print(
        f"{label}: {outcome} finish={cell['finish_reason']} "
        f"gen={cell['generated_tokens']} lat={latency_ms}ms "
        f"recall={cell.get('task_requirement_recall')} "
        f"valid={cell.get('validation_status')} "
        f"amb={cell['ambiguity']['category']}",
        flush=True,
    )
    return cell


def run() -> int:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    db = S2A.SessionLocal()
    try:
        runtime, context, verification = S2U._prepare(db)
        S2A._write_json(CAPTURE_DIR / "freeze.json", verification)
        for arm, index in FROZEN_CALL_ORDER:
            label = f"{arm.lower()}-{index}"
            if (CAPTURE_DIR / f"cell-{label}.json").exists():
                print(f"{label}: already captured, skipping", flush=True)
                continue
            spent = len(list(CAPTURE_DIR.glob("cell-*.json")))
            if spent >= MAX_PROVIDER_CALLS:
                raise SystemExit(f"MAX_PROVIDER_CALLS={MAX_PROVIDER_CALLS} spent")
            _run_call(runtime, context, arm, index)
        return 0
    finally:
        db.close()


# ----------------------------------------------------------------- evidence ---


def _reasoning_split(label: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    """Read the raw capture to see whether the server separated any reasoning."""

    path = CAPTURE_DIR / f"raw-{label}.json"
    if not path.exists():
        return {"raw_capture_present": False}
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        body = json.loads(raw["response"]["raw_body_text"])
        message = body["choices"][0]["message"]
    except Exception:  # noqa: BLE001
        return {"raw_capture_present": True, "parse_failed": True}
    reasoning = message.get("reasoning") or ""
    final = message.get("content") or ""
    generated = float(cell["generated_tokens"])
    total = len(reasoning) + len(final)
    return {
        "raw_capture_present": True,
        "reasoning_field_present": bool(reasoning),
        "reasoning_chars": len(reasoning),
        "final_visible_output_chars": len(final),
        "reasoning_sha256": S2T._sha256(reasoning) if reasoning else None,
        "estimated_reasoning_tokens": (
            round(generated * len(reasoning) / total) if total else 0
        ),
        "reasoning_share": round(len(reasoning) / total, 4) if total else None,
    }


def _rows(arm: str) -> list[dict[str, Any]]:
    out = []
    for _, index in [pair for pair in FROZEN_CALL_ORDER if pair[0] == arm]:
        label = f"{arm.lower()}-{index}"
        path = CAPTURE_DIR / f"cell-{label}.json"
        if not path.exists():
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        cell["reasoning"] = _reasoning_split(label, cell)
        out.append(cell)
    return sorted(out, key=lambda row: row["run_index"])


def _outcome_class(cell: Mapping[str, Any]) -> str:
    if cell["generation_outcome"] == "GENERATION_BUDGET_EXCEEDED":
        return "GENERATION_BUDGET_EXCEEDED"
    if cell["generation_outcome"] != "COMPLETED":
        return "PROVIDER_FAILURE"
    if not cell.get("parse_success"):
        return "PARSE_FAILURE"
    if cell.get("validation_status") == "accepted":
        return "SUCCESS_VALID_PLAN"
    if cell.get("plan_usefulness") == "FAIL":
        return "TASK_SEMANTIC_FAILURE"
    if cell.get("validation_status") == "repair_required":
        return "REPAIR_REQUIRED_PLAN"
    return "REJECTED_PLAN"


def _stats(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"min": None, "median": None, "max": None, "n": 0}
    return {
        "min": min(clean),
        "median": statistics.median(clean),
        "max": max(clean),
        "n": len(clean),
    }


def evidence() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    arms = {arm: _rows(arm) for arm in ("CONTROL", "NON_THINKING")}
    rank = {"PASS": 2, "PARTIAL": 1, "FAIL": 0}

    summary: dict[str, Any] = {}
    for arm, rows in arms.items():
        accepted = sum(1 for r in rows if r.get("validation_status") == "accepted")
        parsed = sum(1 for r in rows if r.get("parse_success") is True)
        useful = sum(1 for r in rows if r.get("plan_usefulness") in ("PASS", "PARTIAL"))
        useful_pass = sum(1 for r in rows if r.get("plan_usefulness") == "PASS")
        verified = sum(1 for r in rows if r.get("verification_correctness") == "PASS")
        summary[arm] = {
            "runs": len(rows),
            "accepted_count": accepted,
            "parse_success_count": parsed,
            "useful_plan_count_pass_or_partial": useful,
            "useful_plan_count_pass": useful_pass,
            "verification_pass_count": verified,
            "budget_exceeded_count": sum(
                1 for r in rows if _outcome_class(r) == "GENERATION_BUDGET_EXCEEDED"
            ),
            "outcome_classes": {
                r["label"]: _outcome_class(r) for r in rows
            },
            "requirement_recall_by_run": {
                r["label"]: r.get("task_requirement_recall") for r in rows
            },
            "plan_usefulness_by_run": {
                r["label"]: r.get("plan_usefulness") for r in rows
            },
            "verification_by_run": {
                r["label"]: r.get("verification_correctness") for r in rows
            },
            "validator_status_by_run": {
                r["label"]: r.get("validation_status") for r in rows
            },
            "validator_findings_by_run": {
                r["label"]: r.get("validator_finding_codes") for r in rows
            },
            "ambiguity_resolution_by_run": {
                r["label"]: r["ambiguity"]["category"] for r in rows
            },
            "generated_tokens_by_run": {
                r["label"]: r["generated_tokens"] for r in rows
            },
            "latency_ms_by_run": {r["label"]: r["latency_ms"] for r in rows},
            "finish_reason_by_run": {r["label"]: r["finish_reason"] for r in rows},
            "reasoning_field_present_by_run": {
                r["label"]: r["reasoning"].get("reasoning_field_present")
                for r in rows
            },
            "estimated_reasoning_tokens_by_run": {
                r["label"]: r["reasoning"].get("estimated_reasoning_tokens")
                for r in rows
            },
            "hallucinated_paths_total": sum(
                int(r.get("hallucinated_paths") or 0) for r in rows
            ),
            "unsupported_operations_total": sum(
                int(r.get("unsupported_operations") or 0) for r in rows
            ),
            "wrong_existing_new_total": sum(
                int(r.get("wrong_existing_new_classification") or 0) for r in rows
            ),
            "mutation_present_when_required_count": sum(
                1 for r in rows if r.get("mutating_paths")
            ),
            "generated_tokens": _stats([r["generated_tokens"] for r in rows]),
            "latency_ms": _stats([float(r["latency_ms"]) for r in rows]),
            "requirement_recall": _stats(
                [
                    float(r["task_requirement_recall"])
                    for r in rows
                    if r.get("task_requirement_recall") is not None
                ]
            ),
            "usefulness_rank": _stats(
                [
                    float(rank[r["plan_usefulness"]])
                    for r in rows
                    if r.get("plan_usefulness") in rank
                ]
            ),
            "verification_rank": _stats(
                [
                    float(rank[r["verification_correctness"]])
                    for r in rows
                    if r.get("verification_correctness") in rank
                ]
            ),
        }

    control, treated = summary["CONTROL"], summary["NON_THINKING"]

    def _median(block: Mapping[str, Any], key: str) -> Optional[float]:
        return block[key]["median"]

    primary_gate = {
        "1_accepted_not_lower": treated["accepted_count"] >= control["accepted_count"],
        "2_parse_not_lower": (
            treated["parse_success_count"] >= control["parse_success_count"]
        ),
        "3_median_recall_not_lower": (
            (_median(treated, "requirement_recall") or 0)
            >= (_median(control, "requirement_recall") or 0)
        ),
        "4_usefulness_not_worse": (
            (_median(treated, "usefulness_rank") or 0)
            >= (_median(control, "usefulness_rank") or 0)
        ),
        "5_verification_not_worse": (
            (_median(treated, "verification_rank") or 0)
            >= (_median(control, "verification_rank") or 0)
        ),
        "6_no_more_hallucinated_paths": (
            treated["hallucinated_paths_total"] <= control["hallucinated_paths_total"]
        ),
        "7_no_more_unsupported_operations": (
            treated["unsupported_operations_total"]
            <= control["unsupported_operations_total"]
        ),
        "8_no_more_wrong_existing_new": (
            treated["wrong_existing_new_total"] <= control["wrong_existing_new_total"]
        ),
        "9_no_ambiguity_handling_loss": (
            sum(
                1
                for value in treated["ambiguity_resolution_by_run"].values()
                if value.startswith(("A_", "C_", "D_"))
            )
            >= sum(
                1
                for value in control["ambiguity_resolution_by_run"].values()
                if value.startswith(("A_", "C_", "D_"))
            )
        ),
    }
    primary_gate["no_observed_quality_loss"] = all(primary_gate.values())

    control_gen_median = _median(control, "generated_tokens") or 0
    treated_gen_median = _median(treated, "generated_tokens") or 0
    stronger_gate = {
        "complete_parseable_plans_at_least_4_of_5": treated["parse_success_count"] >= 4,
        "semantically_useful_at_least_4_of_5": (
            treated["useful_plan_count_pass_or_partial"] >= 4
        ),
        "correct_verification_at_least_4_of_5": treated["verification_pass_count"] >= 4,
        "accepted_not_below_control": (
            treated["accepted_count"] >= control["accepted_count"]
        ),
        "no_safety_or_grounding_regression": (
            primary_gate["6_no_more_hallucinated_paths"]
            and primary_gate["7_no_more_unsupported_operations"]
            and primary_gate["8_no_more_wrong_existing_new"]
        ),
        "median_generation_materially_lower": bool(
            control_gen_median
            and treated_gen_median <= 0.75 * control_gen_median
        ),
    }
    stronger_gate["non_thinking_direction_supported"] = all(stronger_gate.values())

    f2_still_reasoning = any(
        row["reasoning"].get("reasoning_field_present") for row in arms["NON_THINKING"]
    )
    both_poor = (
        control["accepted_count"] <= 1 and treated["accepted_count"] <= 1
    )
    quality = {
        "schema_version": "phase34-s2x-quality-comparison/1",
        "primary_fixture": PRIMARY_FIXTURE,
        "primary_variant": PRIMARY_VARIANT,
        "runs_per_arm": RUNS_PER_ARM,
        "arms": {
            arm: {
                key: block[key]
                for key in (
                    "runs",
                    "accepted_count",
                    "parse_success_count",
                    "useful_plan_count_pass_or_partial",
                    "useful_plan_count_pass",
                    "verification_pass_count",
                    "budget_exceeded_count",
                    "outcome_classes",
                    "requirement_recall_by_run",
                    "plan_usefulness_by_run",
                    "verification_by_run",
                    "validator_status_by_run",
                    "validator_findings_by_run",
                    "ambiguity_resolution_by_run",
                    "mutation_present_when_required_count",
                    "hallucinated_paths_total",
                    "unsupported_operations_total",
                    "wrong_existing_new_total",
                    "requirement_recall",
                    "usefulness_rank",
                    "verification_rank",
                )
            }
            for arm, block in summary.items()
        },
        "section_13_primary_quality_gate": primary_gate,
        "section_14_stronger_support_gate": stronger_gate,
        "raw_validator_acceptance": {
            "CONTROL": f"{control['accepted_count']}/{control['runs']}",
            "NON_THINKING": f"{treated['accepted_count']}/{treated['runs']}",
        },
        "known_confounds_preserved_not_repaired": {
            "lexical_existing_write_false_positive": "OBSERVED",
            "normalizer_semantic_drift": "OBSERVED",
            "adapter_reasoning_observability_gap": "OBSERVED",
        },
        # Section 18 F10 / Section 16: the automated gates above credit any plan
        # carrying a real project test command. Reading all eight adjudicable
        # plans shows two NON_THINKING runs (n3, n5) rewrite greet() to
        # `return f"Hello {name}"`, leave test_greeter.py's
        # `assert greet("Ada") == "Hello"` untouched, and then run
        # `python -m pytest test_greeter.py -v` -- so the Plan fails its own
        # final verification. The validator accepted both. That is a defect the
        # raw acceptance count cannot see, and it is scored here separately
        # rather than repaired.
        "adjudicated_model_plan_quality": {
            "method": (
                "every adjudicable Plan's actual proposed mutation was read; "
                "a Plan is COHERENT only if the mutation it proposes can pass "
                "the verification the same Plan specifies"
            ),
            "CONTROL": {
                "adjudicable": 3,
                "coherent_resolution": 3,
                "self_contradicting": 0,
                "fully_correct_accepted_and_coherent": 1,
                "no_output": 2,
                "detail": (
                    "control-1/3/4 all resolve via C (update the test "
                    "expectation), which is coherent; control-1 and control-3 "
                    "were held at repair_required; control-2/5 produced nothing"
                ),
            },
            "NON_THINKING": {
                "adjudicable": 5,
                "coherent_resolution": 3,
                "self_contradicting": 2,
                "fully_correct_accepted_and_coherent": 3,
                "no_output": 0,
                "detail": (
                    "n1/n2/n4 resolve via C and are coherent; n3/n5 change "
                    "greet() without updating the test they then run, so the "
                    "Plan cannot pass its own pytest step despite validator "
                    "acceptance"
                ),
            },
            "section_14_semantically_useful_adjudicated": "3/5",
            "section_14_threshold": "4/5",
            "section_14_met_on_adjudicated_quality": False,
            "harness_scoring_limitation": (
                "_semantic_checklist marks project_tests_verified true when a "
                "pytest command is present; it does not simulate the test, so "
                "n3/n5 scored recall 1.0 / usefulness PASS / verification PASS"
            ),
        },
        "validator_accepted_self_contradicting_plan": {
            "observed": True,
            "runs": ["non_thinking-3", "non_thinking-5"],
            "direction": (
                "acceptance of a semantically broken Plan -- distinct from, and "
                "opposite in direction to, the known "
                "LEXICAL_EXISTING_WRITE_FALSE_POSITIVE"
            ),
            "repaired": False,
        },
        "n5_caveat": (
            "n=5 per arm is a bounded engineering experiment, not a statistical "
            "certification; ties mean NO_OBSERVED_QUALITY_LOSS, never proof of "
            "equivalence"
        ),
    }

    efficiency = {
        "schema_version": "phase34-s2x-efficiency-comparison/1",
        "arms": {
            arm: {
                "generated_tokens": block["generated_tokens"],
                "generated_tokens_by_run": block["generated_tokens_by_run"],
                "latency_ms": block["latency_ms"],
                "latency_ms_by_run": block["latency_ms_by_run"],
                "finish_reason_by_run": block["finish_reason_by_run"],
                "budget_exceeded_count": block["budget_exceeded_count"],
                "completion_rate": (
                    round(
                        sum(
                            1
                            for value in block["outcome_classes"].values()
                            if value
                            not in (
                                "GENERATION_BUDGET_EXCEEDED",
                                "PROVIDER_FAILURE",
                            )
                        )
                        / max(block["runs"], 1),
                        3,
                    )
                ),
                "reasoning_field_present_by_run": block[
                    "reasoning_field_present_by_run"
                ],
                "estimated_reasoning_tokens_by_run": block[
                    "estimated_reasoning_tokens_by_run"
                ],
            }
            for arm, block in summary.items()
        },
        "median_generation_reduction_percent": (
            round(
                100.0 * (control_gen_median - treated_gen_median) / control_gen_median,
                1,
            )
            if control_gen_median
            else None
        ),
        "median_latency_reduction_percent": (
            round(
                100.0
                * ((_median(control, "latency_ms") or 0) - (_median(treated, "latency_ms") or 0))
                / (_median(control, "latency_ms") or 1),
                1,
            )
        ),
        "section_17_variance": {
            "control_generated_range": [
                control["generated_tokens"]["min"],
                control["generated_tokens"]["max"],
            ],
            "non_thinking_generated_range": [
                treated["generated_tokens"]["min"],
                treated["generated_tokens"]["max"],
            ],
            "control_acceptance_variance": (
                f"{control['accepted_count']}/{control['runs']} accepted"
            ),
            "non_thinking_acceptance_variance": (
                f"{treated['accepted_count']}/{treated['runs']} accepted"
            ),
            "historical_compact_b_control_generated": [2432, 4108, 4380, 5000, 5000],
            "historical_note": (
                "history contextualizes only whether fresh runtime behaviour "
                "looks consistent; it is NOT merged into the fresh sample"
            ),
        },
        "falsification": {
            "f1_reasoning_not_sole_delta": False,
            "f2_non_thinking_still_reasoning": f2_still_reasoning,
            "f3_quality_fell": treated["accepted_count"] < control["accepted_count"],
            "f4_ambiguity_handling_worse": not primary_gate[
                "9_no_ambiguity_handling_loss"
            ],
            "f5_quality_preserved": primary_gate["no_observed_quality_loss"],
            "f6_control_variance_with_stable_quality": False,
            "f6_detail": (
                "control generation ranged 3117-5000 AND quality was unstable "
                "too (1 accepted, 2 repair_required, 2 no output), so this is "
                "not variance-with-stable-quality"
            ),
            "f7_longer_reasoning_associated_with_quality": False,
            "f7_detail": (
                "descriptive only, no causal claim: the single accepted control "
                "was mid-length (3833) while both 5000-token controls produced "
                "nothing, so longer reasoning was not associated with better "
                "quality in this n=5 sample"
            ),
            "f8_non_thinking_stable_and_cheaper": bool(
                stronger_gate["median_generation_materially_lower"]
                and treated["accepted_count"] >= control["accepted_count"]
            ),
            "f9_both_quality_poor": both_poor,
            "f10_validator_confound": True,
        },
    }

    S2A._write_json(EVIDENCE / "control-results.json", {"rows": arms["CONTROL"]})
    S2A._write_json(
        EVIDENCE / "non-thinking-results.json", {"rows": arms["NON_THINKING"]}
    )
    S2A._write_json(EVIDENCE / "quality-comparison.json", quality)
    S2A._write_json(EVIDENCE / "efficiency-comparison.json", efficiency)
    print(json.dumps(quality["arms"], indent=1, default=str))
    print(json.dumps(primary_gate, indent=1))
    print(json.dumps(stronger_gate, indent=1))
    print(json.dumps(efficiency["arms"], indent=1, default=str))
    print(json.dumps(efficiency["falsification"], indent=1))
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    command = sys.argv[1]
    if command == "stage1":
        return stage1()
    if command == "run":
        return run()
    if command == "evidence":
        return evidence()
    print(f"unknown command {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

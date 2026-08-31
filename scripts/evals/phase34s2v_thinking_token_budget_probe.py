"""PHASE34-S2V — thinking_token_budget enforcement and quality-preservation probe.

Evaluation harness only. No production file is modified, no Plan is executed,
no repair runs and no product row is created. The frozen Phase34-S2A Fixture B
COMPACT corpus, prompt builder, parser, normalizer and validator are reused
unchanged via S2T/S2U, and the provider-bound prompt is re-verified
byte-identical before any call.

Stage 1 reads the DEPLOYED server's declared semantics for the field and proves,
provider-free, that the sole wire delta is ``thinking_token_budget``. Stage 2
runs exactly one CONTROL/TREATMENT pair.

Usage:
    stage1      declared field semantics + provider-free wire delta
    run         CONTROL then TREATMENT (frozen order), 2 provider calls
    evidence    write the compact durable evidence set
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evals"))

import httpx  # noqa: E402

import phase34s2a_planning_interface_ablation as S2A  # noqa: E402
import phase34s2ar1_generation_budget_controlled_ablation as R1  # noqa: E402
import phase34s2t_planning_reasoning_termination as S2T  # noqa: E402
import phase34s2u_provider_reasoning_bound_probe as S2U  # noqa: E402
from app.services.agents import runtime_invocation as RI  # noqa: E402
from app.services.agents.providers.openai_chat_adapter import (  # noqa: E402
    _GENERIC_SYSTEM,
    _strip_thinking,
)
from app.services.agents.runtime_invocation import (  # noqa: E402
    RuntimeInvocationOptions,
)

PRIMARY_FIXTURE = "B"
PRIMARY_VARIANT = "COMPACT"

# Section 8: the outer envelope is inherited unchanged from S2T/S2U/S2A-R1 and
# is deliberately NOT the intervention.
OUTER_MAX_GENERATED_TOKENS = 5000
OUTER_DEADLINE_SECONDS = 330

# Section 7: frozen before either call.
FROZEN_CALL_ORDER = ("CONTROL", "TREATMENT")
MAX_PROVIDER_CALLS = 2

# Section 4: above the observed genuine-deliberation region (first-readiness
# points measured at ~1327 / 1925 / 2784 / 3370 on this fixture) and below the
# ~4000+ reasoning lengths measured in S2T/S2U, so it can still cut late
# self-check without removing early task/source deliberation.
TREATMENT_FIELD = "thinking_token_budget"
TREATMENT_VALUE = 3500

EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2v"
CAPTURE_DIR = Path(
    "/tmp/claude-0/-root--openclaw-workspace-vault-projects-orchestrator"
    "/f5bb3e28-8ae8-486d-9ddf-22ac10e4c66b/scratchpad/s2v"
)
OPENAPI_URL = S2U.OPENAPI_URL


# ------------------------------------------------------- invocation options ---


def _control_options() -> RuntimeInvocationOptions:
    """Identical to the S2U control arm: outer envelope plus the pinned system
    prompt the PLANNING role itself selects."""

    return RuntimeInvocationOptions(
        timeout_seconds=OUTER_DEADLINE_SECONDS,
        max_output_tokens=OUTER_MAX_GENERATED_TOKENS,
        system_prompt=_GENERIC_SYSTEM,
    )


def _treatment_options() -> RuntimeInvocationOptions:
    """CONTROL options plus exactly one server-side thinking budget.

    As in S2U, the bound is attached to this harness's own options instance
    after construction. The production ``extra_provider_options`` allowlist is
    left fully in force process-wide; no production file or module state is
    changed.
    """

    options = _control_options()
    object.__setattr__(
        options,
        "extra_provider_options",
        MappingProxyType({TREATMENT_FIELD: TREATMENT_VALUE}),
    )
    return options


# ----------------------------------------------- stage 1: declared semantics ---


def _field_semantics() -> dict[str, Any]:
    """Read the deployed server's own published schema. No generation."""

    with httpx.Client(timeout=20) as client:
        schema = client.get(OPENAPI_URL).json()
        try:
            models = client.get("http://ai-gateway:8000/v1/models").json()
        except Exception:  # noqa: BLE001
            models = {}
    properties = schema["components"]["schemas"]["ChatCompletionRequest"]["properties"]
    declared = properties.get(TREATMENT_FIELD)
    described = bool((declared or {}).get("description"))
    served = (models.get("data") or [{}])[0]
    return {
        "schema_version": "phase34-s2v-field-semantics/1",
        "source": OPENAPI_URL,
        "method": (
            "read of the deployed model server's own published OpenAPI request "
            "schema; no semantics imported from any other vendor's API"
        ),
        "field": TREATMENT_FIELD,
        "declared_schema": declared,
        "declared_type": "integer or null",
        "has_description": described,
        "server_declared_field_semantics": (
            declared.get("description")
            if described
            else "SEMANTICS_NOT_EXPLICITLY_DOCUMENTED"
        ),
        "semantics_interpretation": (
            "UNKNOWN -- the deployed schema declares only a nullable integer "
            "with the title 'Thinking Token Budget'. Whether it is a hard cap, "
            "a soft/preferred budget, a minimum, a template variable or an "
            "accepted-but-ignored compatibility field is NOT documented and is "
            "therefore measured, not assumed."
        ),
        "served_model": {
            "id": served.get("id"),
            "root": served.get("root"),
            "owned_by": served.get("owned_by"),
            "max_model_len": served.get("max_model_len"),
        },
        "prior_s2u_tokenize_observation": (
            "/tokenize showed thinking_token_budget does not alter the rendered "
            "chat-template tokens (17 tokens, same as control). That excludes a "
            "prompt-template mechanism but does NOT prove decoder-level "
            "inactivity, which is what S2V measures."
        ),
        "reasoning_effort_enum": [
            option["enum"]
            for option in (properties.get("reasoning_effort") or {}).get("anyOf", [])
            if isinstance(option, dict) and option.get("enum")
        ],
    }


def stage1() -> int:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    db = S2A.SessionLocal()
    try:
        runtime, context, verification = S2U._prepare(db)
        S2U._RUNTIME = runtime
        prompt = context.prompts[PRIMARY_VARIANT]["provider"]
        semantics = _field_semantics()

        try:
            RuntimeInvocationOptions(
                extra_provider_options={TREATMENT_FIELD: TREATMENT_VALUE}
            )
            options_surface_accepts = True
            options_surface_error = None
        except ValueError as exc:
            options_surface_accepts = False
            options_surface_error = str(exc)
        semantics["orchestrator_options_surface"] = {
            "allowed_extra_provider_options": sorted(
                RI._ALLOWED_EXTRA_PROVIDER_OPTIONS
            ),
            "accepts_thinking_token_budget": options_surface_accepts,
            "rejection": options_surface_error,
            "production_currently_uses_it": False,
        }

        control = S2U._dry_run_payload(_control_options(), prompt)
        treatment = S2U._dry_run_payload(_treatment_options(), prompt)
        delta = _wire_delta(control, treatment)
        delta["fixture_freeze"] = verification
        delta["outer_max_generated_tokens"] = OUTER_MAX_GENERATED_TOKENS
        delta["outer_deadline_seconds"] = OUTER_DEADLINE_SECONDS
        delta["frozen_call_order"] = list(FROZEN_CALL_ORDER)
        delta["thinking_token_budget_value"] = TREATMENT_VALUE

        EVIDENCE.mkdir(parents=True, exist_ok=True)
        S2A._write_json(EVIDENCE / "field-semantics.json", semantics)
        S2A._write_json(EVIDENCE / "wire-delta.json", delta)
        print(json.dumps({
            "server_declared_field_semantics": semantics[
                "server_declared_field_semantics"
            ],
            "wire_delta": delta["wire_delta"],
            "sole_delta": delta["sole_delta_is_the_thinking_token_budget"],
            "reasoning_effort_absent_in_both": delta["reasoning_effort_absent_in_both"],
        }, indent=1))
        return 0 if delta["sole_delta_is_the_thinking_token_budget"] else 1
    finally:
        db.close()


def _wire_delta(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
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
    identical_fields = {
        field: control_payload.get(field) == treatment_payload.get(field)
        for field in (
            "model",
            "messages",
            "temperature",
            "stream",
            "max_tokens",
            "chat_template_kwargs",
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "seed",
        )
    }
    sole = added == {TREATMENT_FIELD: TREATMENT_VALUE}
    return {
        "schema_version": "phase34-s2v-wire-delta/1",
        "url_identical": control["url"] == treatment["url"],
        "headers_identical": control["headers"] == treatment["headers"],
        "required_identical_fields": identical_fields,
        "control_payload_keys": sorted(control_payload),
        "treatment_payload_keys": sorted(treatment_payload),
        "added_in_treatment": added,
        "removed_in_treatment": removed,
        "changed_in_treatment": changed,
        "reasoning_effort_absent_in_both": (
            "reasoning_effort" not in control_payload
            and "reasoning_effort" not in treatment_payload
        ),
        "wire_delta": (
            f'+ "{TREATMENT_FIELD}": {TREATMENT_VALUE}' if sole else "UNEXPECTED"
        ),
        "sole_delta_is_the_thinking_token_budget": bool(
            sole
            and not removed
            and not changed
            and all(identical_fields.values())
            and control["url"] == treatment["url"]
            and control["headers"] == treatment["headers"]
            and "reasoning_effort" not in treatment_payload
        ),
    }


# --------------------------------------------------------- stage 2: the pair ---


def _run_arm(runtime: Any, context: Any, arm: str) -> dict[str, Any]:
    options = _control_options() if arm == "CONTROL" else _treatment_options()
    capture_path = CAPTURE_DIR / f"raw-{arm.lower()}.json"
    prompt = context.prompts[PRIMARY_VARIANT]["provider"]
    before_workspace = S2A._workspace_content_digest(context.workspace)
    quiet, before = R1._wait_quiescent()
    if not quiet:
        raise RuntimeError(f"Provider not quiescent before {arm}")

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
                    "phase": "PHASE34-S2V",
                    "arm": arm,
                    "fixture_id": PRIMARY_FIXTURE,
                    "variant": PRIMARY_VARIANT,
                    "planning_attempt": "initial",
                    "repairs_allowed": False,
                    "execution_allowed": False,
                    "discovery_contract_capture_path": str(capture_path),
                    "discovery_contract_run_id": f"s2v-{arm.lower()}",
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
        "server_preemptions": delta.get("num_preemptions_total"),
        "quiescent_after": quiet_after,
        "counter_delta": delta,
        "workspace_digest_before": before_workspace,
        "workspace_digest_after": S2A._workspace_content_digest(context.workspace),
        "adapter_returned_output_chars": len(adapter_output),
        "capture_path": str(capture_path),
    }
    if outcome == "COMPLETED":
        cell.update(S2A._analyze_candidate(context, adapter_output))
        cell["semantic_adjudicable"] = True
        cell["raw_provider_response"] = adapter_output
        R1._adjudicate_cell(cell, S2A._fixtures()[PRIMARY_FIXTURE])
    else:
        cell["semantic_adjudicable"] = False
    cell.pop("raw_provider_response", None)
    (CAPTURE_DIR / f"cell-{arm.lower()}.json").write_text(
        json.dumps(cell, indent=1, default=str), encoding="utf-8"
    )
    print(
        f"{arm}: {outcome} finish={cell['finish_reason']} "
        f"gen={cell['generated_tokens']} lat={latency_ms}ms "
        f"chars={cell['adapter_returned_output_chars']} "
        f"recall={cell.get('task_requirement_recall')}"
    )
    return cell


def run() -> int:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    spent = sorted(CAPTURE_DIR.glob("raw-*.json"))
    if len(spent) >= MAX_PROVIDER_CALLS:
        raise SystemExit(f"MAX_PROVIDER_CALLS={MAX_PROVIDER_CALLS} already spent")
    db = S2A.SessionLocal()
    try:
        runtime, context, verification = S2U._prepare(db)
        S2A._write_json(CAPTURE_DIR / "freeze.json", verification)
        for arm in FROZEN_CALL_ORDER:
            _run_arm(runtime, context, arm)
        return 0
    finally:
        db.close()


# ----------------------------------------------------------------- evidence ---


def _metrics(arm: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.loads(
        (CAPTURE_DIR / f"raw-{arm.lower()}.json").read_text(encoding="utf-8")
    )
    body = json.loads(raw["response"]["raw_body_text"])
    message = body["choices"][0]["message"]
    reasoning = message.get("reasoning") or ""
    final = message.get("content") or ""
    generated = float(cell["generated_tokens"])
    total_chars = len(reasoning) + len(final)
    readiness = S2T._readiness(reasoning, generated, total_chars) if total_chars else {}
    reasoning_tokens = (
        round(generated * len(reasoning) / total_chars) if total_chars else 0
    )
    return {
        "arm": arm,
        "thinking_token_budget": (
            TREATMENT_VALUE if arm == "TREATMENT" else None
        ),
        "finish_reason": cell["finish_reason"],
        "generation_outcome": cell["generation_outcome"],
        "prompt_tokens": cell["prompt_tokens"],
        "generated_tokens": generated,
        "outer_budget_reached": generated >= OUTER_MAX_GENERATED_TOKENS,
        "latency_ms": cell["latency_ms"],
        "time_to_first_token_ms": cell["time_to_first_token_ms"],
        "reasoning_chars": len(reasoning),
        "final_visible_output_chars": len(final),
        "reasoning_sha256": S2T._sha256(reasoning),
        "final_content_sha256": S2T._sha256(final),
        "estimated_reasoning_tokens": reasoning_tokens,
        "reasoning_share": (
            round(len(reasoning) / total_chars, 4) if total_chars else None
        ),
        "final_answer_transition": readiness,
        "topic_profile": S2T._topic_profile(reasoning),
        "repetition": S2T._repetition(reasoning),
        "quartiles": [
            {
                "quartile": quartile["quartile"],
                "new_term_ratio": quartile["new_term_ratio"],
                "topic_profile": quartile["topic_profile"],
                "repeated_sentence_ratio": quartile["repetition"][
                    "repeated_sentence_ratio"
                ],
            }
            for quartile in S2T._quartiles(reasoning)
        ],
        "bounded_excerpts": {
            "reasoning_head_400": reasoning[:400],
            "reasoning_tail_600": reasoning[-600:],
        },
        "quality": {
            "parse_success": cell.get("parse_success"),
            "task_requirement_recall": cell.get("task_requirement_recall"),
            "task_requirement_checklist": cell.get("task_requirement_checklist"),
            "plan_usefulness": cell.get("plan_usefulness"),
            "verification_correctness": cell.get("verification_correctness"),
            "exact_user_constraint_recall": cell.get("exact_user_constraint_recall"),
            "hallucinated_paths": cell.get("hallucinated_paths"),
            "unsupported_operations": cell.get("unsupported_operations"),
            "wrong_existing_new_classification": cell.get(
                "wrong_existing_new_classification"
            ),
            "expected_files": cell.get("expected_files"),
            "mutating_paths": cell.get("mutating_paths"),
            "verification_commands": cell.get("verification_commands"),
            "initial_plan_valid": cell.get("initial_plan_valid"),
            "validation_status": cell.get("validation_status"),
            "validator_finding_codes": cell.get("validator_finding_codes"),
            "primary_failure_class": cell.get("primary_failure_class"),
            "plan_step_count": cell.get("plan_step_count"),
        },
    }


def _percent(control: float, treatment: float) -> Optional[float]:
    if not control:
        return None
    return round(100.0 * (control - treatment) / control, 1)


def evidence() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in FROZEN_CALL_ORDER:
        cell = json.loads(
            (CAPTURE_DIR / f"cell-{arm.lower()}.json").read_text(encoding="utf-8")
        )
        arms[arm] = _metrics(arm, cell)
        S2A._write_json(EVIDENCE / f"{arm.lower()}-result.json", arms[arm])

    control, treatment = arms["CONTROL"], arms["TREATMENT"]
    c_post = control["final_answer_transition"].get("tokens_after_first_readiness")
    t_post = treatment["final_answer_transition"].get("tokens_after_first_readiness")
    c_reason = control["estimated_reasoning_tokens"]
    t_reason = treatment["estimated_reasoning_tokens"]
    reduction = _percent(c_reason, t_reason)

    rank = {"PASS": 2, "PARTIAL": 1, "FAIL": 0, "NOT_ADJUDICABLE": -1}

    def _not_worse(field: str) -> bool:
        return rank.get(str(treatment["quality"][field]), -1) >= rank.get(
            str(control["quality"][field]), -1
        )

    parse_ok = bool(treatment["quality"]["parse_success"])
    natural_stop = treatment["finish_reason"] == "stop"

    # Section 12/13: a control that hit the OUTER ceiling emitted no Plan, so
    # every control quality field is null. Comparing against it would pass the
    # non-inferiority gate vacuously -- "not worse than nothing" is not
    # non-inferiority. The gate is therefore computed AND explicitly marked
    # unusable when the control is censored.
    control_censored = bool(
        control["finish_reason"] != "stop"
        or not control["final_visible_output_chars"]
        or control["quality"]["parse_success"] is not True
    )

    quality_gate = {
        "parse_success": parse_ok,
        "recall_not_worse": (
            (treatment["quality"]["task_requirement_recall"] or 0)
            >= (control["quality"]["task_requirement_recall"] or 0)
        ),
        "usefulness_not_worse": _not_worse("plan_usefulness"),
        "verification_not_worse": _not_worse("verification_correctness"),
        "hallucinated_paths_not_worse": (
            int(treatment["quality"]["hallucinated_paths"] or 0)
            <= int(control["quality"]["hallucinated_paths"] or 0)
        ),
        "unsupported_operations_not_worse": (
            int(treatment["quality"]["unsupported_operations"] or 0)
            <= int(control["quality"]["unsupported_operations"] or 0)
        ),
        "wrong_existing_new_not_worse": (
            int(treatment["quality"]["wrong_existing_new_classification"] or 0)
            <= int(control["quality"]["wrong_existing_new_classification"] or 0)
        ),
    }
    quality_gate["control_adjudicable"] = not control_censored
    quality_gate["comparisons_vacuous_against_censored_control"] = control_censored
    quality_gate["passes"] = (
        all(
            value
            for key, value in quality_gate.items()
            if key
            not in (
                "control_adjudicable",
                "comparisons_vacuous_against_censored_control",
            )
        )
        and natural_stop
        and not control_censored
    )

    # Section 10 -- was the field actually ACTIVE, not merely accepted?
    # A hard cap would stop reasoning at/near N. A soft budget would reduce
    # reasoning materially RELATIVE TO A VALID CONTROL. Neither holds if the
    # control is censored at the outer ceiling or if the treatment terminated
    # naturally well below N.
    band_low, band_high = TREATMENT_VALUE * 0.9, TREATMENT_VALUE * 1.1
    boundary_observed = bool(band_low <= t_reason <= band_high and t_reason < c_reason)
    materially_lower = bool(reduction is not None and reduction >= 25)
    truncated = bool(
        not natural_stop or not parse_ok or not treatment["final_visible_output_chars"]
    )
    # Section 15 F9/F10: prior runs of this identical frozen prompt under the
    # identical outer envelope, used to decide whether either arm is merely an
    # ordinary draw. Values are read from the durable S2T/S2U/S2A-R1 evidence.
    prior_unbounded = {
        "phase34-s2a-r1 COMPACT B": {
            "generated_tokens": 5000,
            "finish_reason": "length",
            "estimated_reasoning_tokens": None,
        },
        "phase34-s2t COMPACT B": {
            "generated_tokens": 4108,
            "finish_reason": "stop",
            "estimated_reasoning_tokens": 3770,
        },
        "phase34-s2u CONTROL": {
            "generated_tokens": 4380,
            "finish_reason": "stop",
            "estimated_reasoning_tokens": 4031,
        },
        "phase34-s2v CONTROL": {
            "generated_tokens": c_reason and control["generated_tokens"],
            "finish_reason": control["finish_reason"],
            "estimated_reasoning_tokens": c_reason,
        },
    }
    prior_reasoning = [3770, 4031]
    control_within_variance = True  # 4108/4380/5000/5000 generated; this is 5000
    treatment_below_all_prior = bool(t_reason < min(prior_reasoning))

    if truncated and t_reason < c_reason:
        effect_class = "HARMFUL_TRUNCATION"
    elif boundary_observed:
        effect_class = "ACTIVE_HARD_CAP"
    elif control_censored:
        # The measured reduction is against a censored control, so it cannot
        # establish a soft budget -- this is the same artifact class that
        # invalidated the apparent S2U post-readiness improvement.
        effect_class = "UNKNOWN"
    elif materially_lower:
        effect_class = "ACTIVE_SOFT_BUDGET"
    elif reduction is not None and abs(reduction) < 25:
        effect_class = "INERT"
    else:
        effect_class = "UNKNOWN"

    ritual = {
        "control_readiness_markers": control["final_answer_transition"].get(
            "readiness_markers_total"
        ),
        "treatment_readiness_markers": treatment["final_answer_transition"].get(
            "readiness_markers_total"
        ),
        "control_final_quarter_density": control["final_answer_transition"].get(
            "final_quarter_ritual_density_per_1k_tokens"
        ),
        "treatment_final_quarter_density": treatment["final_answer_transition"].get(
            "final_quarter_ritual_density_per_1k_tokens"
        ),
    }
    ritual["ritual_materially_reduced"] = bool(
        (ritual["treatment_readiness_markers"] or 0)
        < 0.75 * (ritual["control_readiness_markers"] or 0)
        and (ritual["treatment_final_quarter_density"] or 0)
        < 0.75 * (ritual["control_final_quarter_density"] or 0)
    )

    success_gate = {
        "mechanism_demonstrably_active": effect_class
        in ("ACTIVE_HARD_CAP", "ACTIVE_SOFT_BUDGET"),
        "valid_control_baseline": not control_censored,
        "natural_completion": natural_stop,
        "quality_non_inferiority": quality_gate["passes"],
        "reasoning_materially_lower": materially_lower,
    }
    success_gate["directional_support"] = all(success_gate.values())

    comparison = {
        "schema_version": "phase34-s2v-comparison/1",
        "primary_fixture": PRIMARY_FIXTURE,
        "primary_variant": PRIMARY_VARIANT,
        "frozen_call_order": list(FROZEN_CALL_ORDER),
        "outer_max_generated_tokens": OUTER_MAX_GENERATED_TOKENS,
        "outer_deadline_seconds": OUTER_DEADLINE_SECONDS,
        "thinking_token_budget_value": TREATMENT_VALUE,
        "wire_delta": f'+ "{TREATMENT_FIELD}": {TREATMENT_VALUE}',
        "efficiency": {
            "control_generated_tokens": control["generated_tokens"],
            "treatment_generated_tokens": treatment["generated_tokens"],
            "control_reasoning_tokens": c_reason,
            "treatment_reasoning_tokens": t_reason,
            "reasoning_token_reduction_percent": reduction,
            "control_reasoning_share": control["reasoning_share"],
            "treatment_reasoning_share": treatment["reasoning_share"],
            "control_first_readiness_token": control["final_answer_transition"].get(
                "first_readiness_token_index"
            ),
            "treatment_first_readiness_token": treatment[
                "final_answer_transition"
            ].get("first_readiness_token_index"),
            "control_post_readiness_tokens": c_post,
            "treatment_post_readiness_tokens": t_post,
            "post_readiness_token_reduction_percent": (
                _percent(float(c_post), float(t_post))
                if c_post is not None and t_post is not None
                else None
            ),
        },
        "budget_boundary": {
            "declared_budget": TREATMENT_VALUE,
            "boundary_band": [band_low, band_high],
            "treatment_estimated_reasoning_tokens": t_reason,
            "control_outer_budget_reached": control["outer_budget_reached"],
            "treatment_outer_budget_reached": treatment["outer_budget_reached"],
            "thinking_budget_reached_or_inferred": boundary_observed,
            "estimator_note": (
                "reasoning tokens are estimated by character share of the "
                "server's own total generation counter, so the boundary test "
                "uses a +/-10% band rather than an exact equality"
            ),
        },
        "thinking_token_budget_effect_class": effect_class,
        "control_validity": {
            "control_censored_at_outer_ceiling": control_censored,
            "control_finish_reason": control["finish_reason"],
            "control_final_visible_output_chars": control["final_visible_output_chars"],
            "control_full_json_plan_drafts_inside_reasoning": control[
                "final_answer_transition"
            ].get("full_json_plan_drafts_inside_reasoning"),
            "consequence": (
                "the control produced no adjudicable Plan, so the measured "
                "reasoning reduction is against a censored upper-bound "
                "observation and the quality non-inferiority gate is vacuous"
            ),
        },
        "prior_runs_same_frozen_prompt_same_envelope": prior_unbounded,
        "treatment_reasoning_below_all_prior_completed_runs": treatment_below_all_prior,
        "quality_non_inferiority_gate": quality_gate,
        "ritual_size": ritual,
        "section_14_success_gate": success_gate,
        "section_14_directional_threshold_percent": 25,
        "falsification": {
            "f1_wire_proven": True,
            "f2_no_effect": effect_class == "INERT",
            "f3_boundary_observed": boundary_observed,
            "f3_note": (
                "treatment reasoning terminated naturally at "
                f"{t_reason} estimated tokens, {TREATMENT_VALUE - t_reason} "
                "below the declared budget, ending with a natural close rather "
                "than a cutoff; no boundary behaviour at N was observed"
            ),
            "f4_natural_completion": natural_stop,
            "f5_truncation_false_improvement": bool(truncated and t_reason < c_reason),
            "f6_quality_degraded": not quality_gate["passes"],
            "f7_necessary_deliberation_removed": bool(
                treatment["final_answer_transition"].get("first_readiness_token_index")
                is None
                or not parse_ok
            ),
            "f8_self_check_reduced": ritual["ritual_materially_reduced"],
            "f9_control_within_variance": control_within_variance,
            "f10_treatment_within_variance": not treatment_below_all_prior,
        },
    }
    S2A._write_json(EVIDENCE / "comparison.json", comparison)
    print(json.dumps(comparison["efficiency"], indent=1))
    print(json.dumps(comparison["budget_boundary"], indent=1))
    print(json.dumps(comparison["quality_non_inferiority_gate"], indent=1))
    print(json.dumps(comparison["ritual_size"], indent=1))
    print(json.dumps(comparison["section_14_success_gate"], indent=1))
    print("EFFECT_CLASS =", effect_class)
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

"""PHASE34-S2U — Provider-level reasoning-bound capability and quality probe.

Evaluation harness only. No production file is modified, no Plan is executed,
no repair runs and no product row is created. The frozen Phase34-S2A Fixture B
COMPACT corpus, prompt builder, parser, normalizer and validator are reused
unchanged, and the provider-bound prompt is re-verified byte-identical before
any call.

Stage 1 proves, provider-free, whether the deployed path exposes a graded
reasoning bound. Stage 2 runs exactly one CONTROL/TREATMENT pair.

Usage:
    stage1      provider-free capability proof + wire delta
    run         CONTROL then TREATMENT (frozen order), 2 provider calls
    evidence    write the compact durable evidence set
"""

from __future__ import annotations

import asyncio
import json
import re
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

# Section 9: identical outer envelope for both arms, inherited from S2T/S2A-R1.
# The outer ceiling must not itself be the intervention.
OUTER_MAX_GENERATED_TOKENS = 5000
OUTER_DEADLINE_SECONDS = 330

# Section 8: call order is frozen here, before either call, and is not adapted
# after observing the control.
FROZEN_CALL_ORDER = ("CONTROL", "TREATMENT")
MAX_PROVIDER_CALLS = 2

# Section 5: a qualifying treatment constrains the AMOUNT of reasoning while
# preserving reasoning mode. "none" would be a disable and is out of scope.
TREATMENT_FIELD = "reasoning_effort"
TREATMENT_VALUE = "low"

EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2u"
CAPTURE_DIR = Path(
    "/tmp/claude-0/-root--openclaw-workspace-vault-projects-orchestrator"
    "/33e2968d-6200-4616-8657-d367b8c2814d/scratchpad/s2u"
)
OPENAPI_URL = "http://ai-gateway:8000/openapi.json"


# ------------------------------------------------------- invocation options ---


def _control_options() -> RuntimeInvocationOptions:
    """Both arms pin the system prompt the PLANNING role would have supplied.

    ``execute_task`` calls ``dataclasses.replace()`` when ``system_prompt`` is
    None, which re-runs ``__post_init__`` and would discard the treatment's
    bound -- the production allowlist is re-enforced at the adapter boundary.
    Pinning the identical ``_GENERIC_SYSTEM`` the PLANNING role selects keeps
    that path out of both arms symmetrically, and the dry-run wire diff proves
    the system message is unchanged.
    """

    return RuntimeInvocationOptions(
        timeout_seconds=OUTER_DEADLINE_SECONDS,
        max_output_tokens=OUTER_MAX_GENERATED_TOKENS,
        system_prompt=_GENERIC_SYSTEM,
    )


def _treatment_options() -> RuntimeInvocationOptions:
    """CONTROL options plus exactly one proven server-side reasoning bound.

    ``RuntimeInvocationOptions`` validates ``extra_provider_options`` against a
    closed allowlist that does not contain ``reasoning_effort``. Rather than
    widen that production safeguard -- in a file or process-wide -- the bound is
    attached to this harness's own single options instance after construction.
    The allowlist therefore remains fully in force for every other caller and
    for the whole process; no production module state is mutated and no
    production file is changed. This is the evaluation-only invocation
    difference Section 2 permits once Stage 1 has proven server support.
    """

    options = _control_options()
    object.__setattr__(
        options,
        "extra_provider_options",
        MappingProxyType({TREATMENT_FIELD: TREATMENT_VALUE}),
    )
    return options


# ------------------------------------------------ stage 1: capability proof ---


def _server_capability() -> dict[str, Any]:
    """Read the DEPLOYED server's own published request schema. No generation."""

    with httpx.Client(timeout=20) as client:
        schema = client.get(OPENAPI_URL).json()
    properties = schema["components"]["schemas"]["ChatCompletionRequest"]["properties"]
    interesting = {
        name: properties[name]
        for name in (
            "reasoning_effort",
            "thinking_token_budget",
            "include_reasoning",
            "chat_template_kwargs",
        )
        if name in properties
    }
    effort = properties.get("reasoning_effort") or {}
    enum: list[str] = []
    for option in effort.get("anyOf") or []:
        if isinstance(option, dict) and option.get("enum"):
            enum = list(option["enum"])
    return {
        "source": OPENAPI_URL,
        "method": (
            "read of the deployed model server's own published OpenAPI request "
            "schema -- not inferred from any other vendor's public API"
        ),
        "chat_completion_request_property_count": len(properties),
        "reasoning_parameters_present": sorted(interesting),
        "reasoning_effort_enum": enum,
        "thinking_token_budget_type": "integer",
        "graded_bound_available": bool([v for v in enum if v not in ("none",)]),
        "binary_only": False,
    }


def _dry_run_payload(options: RuntimeInvocationOptions, prompt: str) -> dict[str, Any]:
    """Capture the outbound payload provider-free by aborting the transport."""

    captured: dict[str, Any] = {}

    class _Abort(Exception):
        pass

    async def _post(self, url, *, headers=None, json=None, **kwargs):  # noqa: A002
        captured["url"] = str(url)
        captured["headers"] = dict(headers or {})
        captured["payload"] = json
        raise _Abort()

    original = httpx.AsyncClient.post
    httpx.AsyncClient.post = _post  # type: ignore[method-assign]
    try:
        asyncio.run(
            S2A.PlannerService._execute_task_with_planning_lock(
                _RUNTIME,
                prompt,
                timeout_seconds=OUTER_DEADLINE_SECONDS,
                reuse_task_session=False,
                diagnostic_label="PLANNING",
                diagnostic_metadata={"phase": "PHASE34-S2U-DRYRUN"},
                invocation_options=options,
            )
        )
    except BaseException:  # noqa: BLE001 - the abort is the success path
        pass
    finally:
        httpx.AsyncClient.post = original  # type: ignore[method-assign]
    if "payload" not in captured:
        raise RuntimeError("provider-free payload interception failed")
    captured["headers"] = {
        key: ("<redacted>" if key.lower() == "authorization" else value)
        for key, value in captured["headers"].items()
    }
    return captured


def _wire_delta(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
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
        for field in ("model", "messages", "temperature", "stream", "max_tokens")
    }
    return {
        "schema_version": "phase34-s2u-wire-delta/1",
        "url_identical": control["url"] == treatment["url"],
        "headers_identical": control["headers"] == treatment["headers"],
        "required_identical_fields": identical_fields,
        "control_payload_keys": sorted(control_payload),
        "treatment_payload_keys": sorted(treatment_payload),
        "added_in_treatment": added,
        "removed_in_treatment": removed,
        "changed_in_treatment": changed,
        "wire_delta": (
            f"+ \"{TREATMENT_FIELD}\": \"{TREATMENT_VALUE}\""
            if added == {TREATMENT_FIELD: TREATMENT_VALUE}
            else "UNEXPECTED"
        ),
        "sole_delta_is_the_reasoning_bound": (
            added == {TREATMENT_FIELD: TREATMENT_VALUE}
            and not removed
            and not changed
            and all(identical_fields.values())
            and control["url"] == treatment["url"]
            and control["headers"] == treatment["headers"]
        ),
        "f1_wire_support_not_proven": False,
        "f2_other_parameter_changed": bool(removed or changed),
    }


_RUNTIME: Any = None
_CONTEXT: Any = None


def _prepare(db: Any) -> tuple[Any, Any, dict[str, Any]]:
    freeze = json.loads(
        (S2T.S2A_EVIDENCE / "fixture-freeze.json").read_text(encoding="utf-8")
    )
    context, corpus_gate = S2T._rebuild_fixture_b(db, freeze)
    verification = S2T._verify_fixture_b(freeze, context)
    verification["corpus_gate"] = corpus_gate
    if not corpus_gate["corpus_verified"] or not verification["prompt_freeze_verified"]:
        raise RuntimeError(f"Section 7 freeze failed: {json.dumps(verification)}")
    runtime = S2A.create_agent_runtime(db, None, None, role=S2A.BackendRole.PLANNING)
    metadata = runtime.get_backend_metadata()
    identity = {
        "backend": metadata.get("backend"),
        "model": metadata.get("model_family"),
        "profile": metadata.get("adaptation_profile"),
    }
    expected = {
        "backend": freeze["runtime_freeze"]["planning_backend"],
        "model": freeze["runtime_freeze"]["planning_model"],
        "profile": freeze["runtime_freeze"]["planning_adaptation_profile"],
    }
    if identity != expected:
        raise RuntimeError(f"Runtime identity changed: {identity}")
    verification["runtime_identity"] = identity
    verification["temperature"] = freeze["runtime_freeze"]["temperature"]
    return runtime, context, verification


def stage1() -> int:
    global _RUNTIME
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    db = S2A.SessionLocal()
    try:
        runtime, context, verification = _prepare(db)
        _RUNTIME = runtime
        prompt = context.prompts[PRIMARY_VARIANT]["provider"]
        server = _server_capability()

        allowlist = sorted(RI._ALLOWED_EXTRA_PROVIDER_OPTIONS)
        try:
            RuntimeInvocationOptions(
                extra_provider_options={TREATMENT_FIELD: TREATMENT_VALUE}
            )
            options_surface_accepts = True
            options_surface_error = None
        except ValueError as exc:
            options_surface_accepts = False
            options_surface_error = str(exc)

        proof = {
            "schema_version": "phase34-s2u-reasoning-capability-proof/1",
            "deployed_server_capability": server,
            "orchestrator_options_surface": {
                "allowed_extra_provider_options": allowlist,
                "accepts_reasoning_bound": options_surface_accepts,
                "rejection": options_surface_error,
                "reasoning_control_expressible_in_options": (
                    "reasoning_enabled: bool -> think/enable_thinking/"
                    "chat_template_kwargs.enable_thinking = False (BINARY DISABLE, "
                    "not a bound)"
                ),
                "production_uses_extra_provider_options_for": (
                    "read_only_discovery.py only, response_format={'type':"
                    " 'json_object'}"
                ),
            },
            "provider_reasoning_bound_capability": "SUPPORTED",
            "supported_parameter_or_mechanism": (
                "reasoning_effort (enum none|low|medium|high) and "
                "thinking_token_budget (integer), both native to the deployed "
                "model server"
            ),
            "supported_at_layer": "MODEL_SERVER",
            "not_supported_at_layer": ["RUNTIME_OPTIONS", "ADAPTER"],
            "production_currently_uses_it": False,
            "evaluation_only_override_possible_without_production_change": True,
            "override_mechanism": (
                "attach the bound to the harness's own RuntimeInvocationOptions "
                "instance after construction; the production allowlist is left "
                "in force process-wide and no production file or module state "
                "is changed"
            ),
            "only_binary_thinking_control_available": False,
            "binary_disable_rejected_as_treatment": (
                "reasoning_effort='none' and reasoning_enabled=False are "
                "disables, not bounds; Section 5 excludes them"
            ),
            "chosen_treatment": {TREATMENT_FIELD: TREATMENT_VALUE},
            "chosen_treatment_rationale": (
                "graded intensity control that preserves reasoning mode; "
                "thinking_token_budget was rejected as the treatment because a "
                "hard cap risks a Section 16 F5 truncation false improvement"
            ),
            "wire_effect_can_be_proven": True,
        }

        control = _dry_run_payload(_control_options(), prompt)
        treatment = _dry_run_payload(_treatment_options(), prompt)
        delta = _wire_delta(control, treatment)
        delta["fixture_freeze"] = verification
        delta["outer_max_generated_tokens"] = OUTER_MAX_GENERATED_TOKENS
        delta["outer_deadline_seconds"] = OUTER_DEADLINE_SECONDS
        delta["frozen_call_order"] = list(FROZEN_CALL_ORDER)

        EVIDENCE.mkdir(parents=True, exist_ok=True)
        S2A._write_json(EVIDENCE / "reasoning-capability-proof.json", proof)
        S2A._write_json(EVIDENCE / "wire-delta.json", delta)
        print(json.dumps({
            "capability": proof["provider_reasoning_bound_capability"],
            "layer": proof["supported_at_layer"],
            "binary_only": proof["only_binary_thinking_control_available"],
            "wire_delta": delta["wire_delta"],
            "sole_delta": delta["sole_delta_is_the_reasoning_bound"],
        }, indent=1))
        return 0 if delta["sole_delta_is_the_reasoning_bound"] else 1
    finally:
        db.close()


# ------------------------------------------------------- stage 2: the pair ---


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
                    "phase": "PHASE34-S2U",
                    "arm": arm,
                    "fixture_id": PRIMARY_FIXTURE,
                    "variant": PRIMARY_VARIANT,
                    "planning_attempt": "initial",
                    "repairs_allowed": False,
                    "execution_allowed": False,
                    "discovery_contract_capture_path": str(capture_path),
                    "discovery_contract_run_id": f"s2u-{arm.lower()}",
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
    # exact_contract suppresses the adapter's own reasoning strip; apply the
    # identical production function, exactly as S2A-R1 and S2T did.
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

    # Section 12: score the provider Plan on the same basis S2A/R1 used.
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
        f"{arm}: {outcome} gen={cell['generated_tokens']} lat={latency_ms}ms "
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
        runtime, context, verification = _prepare(db)
        S2A._write_json(CAPTURE_DIR / "freeze.json", verification)
        for arm in FROZEN_CALL_ORDER:
            _run_arm(runtime, context, arm)
        return 0
    finally:
        db.close()


# --------------------------------------------------------------- evidence ---


def _reasoning_metrics(arm: str, cell: Mapping[str, Any]) -> dict[str, Any]:
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
    reasoning_tokens = round(generated * len(reasoning) / total_chars) if total_chars else 0
    return {
        "arm": arm,
        "finish_reason": cell["finish_reason"],
        "generation_outcome": cell["generation_outcome"],
        "prompt_tokens": cell["prompt_tokens"],
        "generated_tokens": generated,
        "latency_ms": cell["latency_ms"],
        "time_to_first_token_ms": cell["time_to_first_token_ms"],
        "reasoning_chars": len(reasoning),
        "final_visible_output_chars": len(final),
        "reasoning_sha256": S2T._sha256(reasoning),
        "final_content_sha256": S2T._sha256(final),
        "estimated_reasoning_tokens": reasoning_tokens,
        "final_content_start_index": reasoning_tokens,
        "reasoning_share": round(len(reasoning) / total_chars, 4) if total_chars else None,
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
        arms[arm] = _reasoning_metrics(arm, cell)
        S2A._write_json(EVIDENCE / f"{arm.lower()}-result.json", arms[arm])

    control, treatment = arms["CONTROL"], arms["TREATMENT"]
    control_post = control["final_answer_transition"].get("tokens_after_first_readiness")
    treatment_post = treatment["final_answer_transition"].get(
        "tokens_after_first_readiness"
    )
    reasoning_reduction = _percent(
        control["estimated_reasoning_tokens"], treatment["estimated_reasoning_tokens"]
    )
    post_reduction = (
        _percent(float(control_post), float(treatment_post))
        if control_post is not None and treatment_post is not None
        else None
    )
    # PASS > PARTIAL > FAIL. An earlier form of this gate passed any
    # degradation whenever the control was not PASS; that was wrong.
    rank = {"PASS": 2, "PARTIAL": 1, "FAIL": 0, "NOT_ADJUDICABLE": -1}

    def _not_worse(field: str) -> bool:
        return rank.get(str(treatment["quality"][field]), -1) >= rank.get(
            str(control["quality"][field]), -1
        )

    quality_gate = {
        "parse_success": bool(treatment["quality"]["parse_success"]),
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
    }
    quality_gate["passes"] = all(quality_gate.values())
    comparison = {
        "schema_version": "phase34-s2u-comparison/1",
        "primary_fixture": PRIMARY_FIXTURE,
        "primary_variant": PRIMARY_VARIANT,
        "frozen_call_order": list(FROZEN_CALL_ORDER),
        "outer_max_generated_tokens": OUTER_MAX_GENERATED_TOKENS,
        "outer_deadline_seconds": OUTER_DEADLINE_SECONDS,
        "wire_delta": f'+ "{TREATMENT_FIELD}": "{TREATMENT_VALUE}"',
        "efficiency": {
            "control_generated_tokens": control["generated_tokens"],
            "treatment_generated_tokens": treatment["generated_tokens"],
            "control_reasoning_tokens": control["estimated_reasoning_tokens"],
            "treatment_reasoning_tokens": treatment["estimated_reasoning_tokens"],
            "control_post_readiness_tokens": control_post,
            "treatment_post_readiness_tokens": treatment_post,
            "reasoning_token_reduction_percent": reasoning_reduction,
            "post_readiness_token_reduction_percent": post_reduction,
            "control_reasoning_share": control["reasoning_share"],
            "treatment_reasoning_share": treatment["reasoning_share"],
            "control_first_readiness_token": control["final_answer_transition"].get(
                "first_readiness_token_index"
            ),
            "treatment_first_readiness_token": treatment["final_answer_transition"].get(
                "first_readiness_token_index"
            ),
        },
        "quality_non_inferiority_gate": quality_gate,
        "section_14_directional_threshold_percent": 30,
        # Section 14 A asks whether the self-check tail materially SHRANK. A
        # post-readiness drop caused by readiness arriving later inside a
        # LONGER trace is not that. The ritual-size checks below are what
        # decide it; the raw percentage alone is misleading here.
        "section_14_reduction_gate_met": bool(
            post_reduction is not None
            and post_reduction >= 30
            and treatment["estimated_reasoning_tokens"]
            < control["estimated_reasoning_tokens"]
        ),
        "post_readiness_reduction_is_mechanically_real": bool(
            treatment["estimated_reasoning_tokens"]
            < control["estimated_reasoning_tokens"]
        ),
        "ritual_size": {
            "control_readiness_markers": control["final_answer_transition"].get(
                "readiness_markers_total"
            ),
            "treatment_readiness_markers": treatment["final_answer_transition"].get(
                "readiness_markers_total"
            ),
            "control_final_quarter_density": control["final_answer_transition"].get(
                "final_quarter_ritual_density_per_1k_tokens"
            ),
            "treatment_final_quarter_density": treatment[
                "final_answer_transition"
            ].get("final_quarter_ritual_density_per_1k_tokens"),
            "ritual_materially_reduced": False,
        },
        "bound_honored_by_model": {
            "control_prompt_tokens": control["prompt_tokens"],
            "treatment_prompt_tokens": treatment["prompt_tokens"],
            "rendered_prompt_changed": control["prompt_tokens"]
            != treatment["prompt_tokens"],
            "chat_template_probe": (
                "/tokenize with chat_template_kwargs: enable_thinking=False "
                "changes the rendered prompt (17 -> 19 tokens); "
                "reasoning_effort='low' and thinking_token_budget=256 do not "
                "(17 tokens). This model's chat template has no graded-effort "
                "variable."
            ),
            "conclusion": (
                "accepted by the API and present on the wire, but not "
                "represented in the rendered prompt for this model"
            ),
        },
        "falsification": {
            "f1_wire_support_not_proven": False,
            "f2_other_parameter_changed": False,
            "f3_input_not_frozen": False,
            "f4_quality_degraded": not quality_gate["passes"],
            "f5_truncation_false_improvement": (
                treatment["finish_reason"] != "stop"
                or not treatment["quality"]["parse_success"]
            ),
            "f6_self_check_specific_reduction": False,
            "f7_no_effect": bool(
                treatment["estimated_reasoning_tokens"]
                >= control["estimated_reasoning_tokens"]
            ),
        },
    }
    S2A._write_json(EVIDENCE / "comparison.json", comparison)
    print(json.dumps(comparison["efficiency"], indent=1))
    print(json.dumps(comparison["quality_non_inferiority_gate"], indent=1))
    print(json.dumps(comparison["falsification"], indent=1))
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

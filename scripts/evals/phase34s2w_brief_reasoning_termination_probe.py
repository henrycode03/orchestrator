"""PHASE34-S2W — brief-reasoning termination instruction probe.

Evaluation harness only. No production file, prompt builder, validator,
normalizer or adapter is modified, no Plan is executed, no repair runs and no
product row is created. The frozen Phase34-S2A Fixture B COMPACT corpus, prompt
builder, parser, normalizer and validator are reused unchanged via S2T/S2U, and
the CONTROL provider prompt is re-verified byte-identical before any call.

The TREATMENT prompt is the CONTROL prompt with exactly one short termination
instruction inserted immediately before the final "return JSON only" rule. The
insertion is performed on the already-built frozen prompt string by this
harness; no production prompt builder is touched.

Usage:
    stage1      provider-free prompt diff + placement freeze
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
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evals"))

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

PRIMARY_FIXTURE = "B"
PRIMARY_VARIANT = "COMPACT"

# Section 8: identical envelope for both arms, inherited unchanged. Neither arm
# carries any provider-specific reasoning control -- no reasoning_effort, no
# thinking_token_budget, no thinking disable.
OUTER_MAX_GENERATED_TOKENS = 5000
OUTER_DEADLINE_SECONDS = 330

# Section 7: frozen before either call.
FROZEN_CALL_ORDER = ("CONTROL", "TREATMENT")
MAX_PROVIDER_CALLS = 2

# Section 4: exactly one conceptual rule -- stop re-checking once the Plan is
# ready and emit. No schema rule, example, provider term, token limit, time
# limit, validator or authority vocabulary is introduced.
TREATMENT_INSTRUCTION = (
    "Reason only as much as needed to form a valid Plan. Once the Plan is "
    "ready, stop re-checking it and immediately return the final JSON."
)

# Section 5: placement is frozen here, before any call. The instruction is
# inserted immediately before the final "return JSON only" rule (rule 1 of the
# ALLOWED ACTIONS AND PLAN CONTRACT block, which is the prompt's authoritative
# output instruction) and appears exactly once.
ANCHOR_LINE = "1. Return one top-level JSON array and no prose or markdown."
TERMINATION_INSTRUCTION_POSITION = (
    "immediately before the final 'return JSON only' instruction -- inserted as "
    "a standalone unnumbered line between the 'ALLOWED ACTIONS AND PLAN "
    "CONTRACT' header and rule 1, so it is adjacent to the output instruction, "
    "is not buried in source evidence, and is not rendered as a numbered "
    "contract rule"
)

EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2w"
CAPTURE_DIR = Path(
    "/tmp/claude-0/-root--openclaw-workspace-vault-projects-orchestrator"
    "/f5bb3e28-8ae8-486d-9ddf-22ac10e4c66b/scratchpad/s2w"
)

# Section 11: deterministic markers for the Fixture B ambiguity -- the task asks
# to include the name while preserving current behaviour, but the existing test
# asserts greet("Ada") == "Hello". These terms are what genuine early
# deliberation over that contradiction looks like.
_AMBIGUITY_MARKERS = (
    "contradict",
    "conflict",
    "ambigu",
    "preserv",
    "existing test",
    "current behavior",
    "current behaviour",
    "== \"hello\"",
    "backward",
    "break the test",
    "breaks the test",
    "test expects",
    "tension",
    "default",
)


def _prompt_sections(prompt: str) -> dict[str, str]:
    """Split the frozen prompt into the model-facing contracts Section 6 pins."""

    def _between(start: str, end: Optional[str]) -> str:
        head = prompt.index(start)
        tail = prompt.index(end) if end else len(prompt)
        return prompt[head:tail]

    return {
        "objective_and_mode": prompt[: prompt.index("Prompt Body:")],
        "task_text": _between("USER TASK", "TASK MODE"),
        "task_mode": _between("TASK MODE", "CURRENT EVIDENCE"),
        "project_context": _between("PROJECT CONTEXT", "PYTHON TEST SOURCE CONTEXT"),
        "verification_contract": _between(
            "PYTHON TEST SOURCE CONTEXT", "CURRENT SOURCE MATERIALIZATION"
        ),
        "source": _between(
            "CURRENT SOURCE MATERIALIZATION", "ALLOWED ACTIONS AND PLAN CONTRACT"
        ),
        "actions_and_plan_schema": _between("ALLOWED ACTIONS AND PLAN CONTRACT", None),
    }


def _treatment_prompt(control: str) -> str:
    """CONTROL plus the single termination instruction at the frozen position."""

    if control.count(ANCHOR_LINE) != 1:
        raise RuntimeError("placement anchor is not unique in the frozen prompt")
    return control.replace(
        ANCHOR_LINE, f"{TREATMENT_INSTRUCTION}\n{ANCHOR_LINE}", 1
    )


# --------------------------------------------------- stage 1: prompt diff ---


def _prompt_diff(control: str, treatment: str) -> dict[str, Any]:
    control_sections = _prompt_sections(control)
    treatment_sections = _prompt_sections(treatment)
    # The schema block legitimately differs only by the inserted line; compare
    # it with the insertion removed so the check is meaningful rather than
    # trivially false.
    treatment_schema_restored = treatment_sections["actions_and_plan_schema"].replace(
        f"{TREATMENT_INSTRUCTION}\n", "", 1
    )
    identical = {
        "TASK_TEXT_IDENTICAL": control_sections["task_text"]
        == treatment_sections["task_text"],
        "SOURCE_IDENTICAL": control_sections["source"] == treatment_sections["source"],
        "MODE_IDENTICAL": (
            control_sections["task_mode"] == treatment_sections["task_mode"]
            and control_sections["objective_and_mode"]
            == treatment_sections["objective_and_mode"]
        ),
        "PROJECT_CONTEXT_IDENTICAL": control_sections["project_context"]
        == treatment_sections["project_context"],
        "VERIFICATION_CONTRACT_IDENTICAL": control_sections["verification_contract"]
        == treatment_sections["verification_contract"],
        "ACTIONS_AND_PLAN_SCHEMA_IDENTICAL_MODULO_INSERTION": (
            treatment_schema_restored == control_sections["actions_and_plan_schema"]
        ),
    }
    added = [
        line
        for line in treatment.split("\n")
        if line not in control.split("\n") or treatment.count(line) > control.count(line)
    ]
    removed = [line for line in control.split("\n") if line not in treatment.split("\n")]
    sole_delta = (
        treatment.replace(f"{TREATMENT_INSTRUCTION}\n", "", 1) == control
        and treatment.count(TREATMENT_INSTRUCTION) == 1
        and not removed
    )
    return {
        "schema_version": "phase34-s2w-prompt-diff/1",
        "primary_fixture": PRIMARY_FIXTURE,
        "primary_variant": PRIMARY_VARIANT,
        "treatment_instruction": TREATMENT_INSTRUCTION,
        "termination_instruction_position": TERMINATION_INSTRUCTION_POSITION,
        "anchor_line": ANCHOR_LINE,
        "static_directive_delta_conceptual_rules": 1,
        "control_prompt_sha256": S2T._sha256(control),
        "treatment_prompt_sha256": S2T._sha256(treatment),
        "control_prompt_chars": len(control),
        "treatment_prompt_chars": len(treatment),
        "prompt_char_delta": len(treatment) - len(control),
        "required_identical": identical,
        "sampling_identical": True,
        "sampling_note": (
            "both arms use the identical RuntimeInvocationOptions -- same "
            "timeout, same max_output_tokens, same pinned system prompt, and "
            "no extra_provider_options on either arm"
        ),
        "added_lines": added,
        "removed_lines": removed,
        "instruction_occurrences_in_treatment": treatment.count(TREATMENT_INSTRUCTION),
        "sole_semantic_delta_is_termination_instruction": bool(sole_delta),
        "experiment_valid": bool(sole_delta and all(identical.values())),
        "forbidden_content_check": {
            term: term in TREATMENT_INSTRUCTION.lower()
            for term in (
                "think step by step",
                "be concise",
                "token",
                "second",
                "validator",
                "authority",
                "phase",
                "example",
                "schema",
            )
        },
    }


def _options() -> RuntimeInvocationOptions:
    """Identical for both arms; no provider-specific reasoning control."""

    return RuntimeInvocationOptions(
        timeout_seconds=OUTER_DEADLINE_SECONDS,
        max_output_tokens=OUTER_MAX_GENERATED_TOKENS,
        system_prompt=_GENERIC_SYSTEM,
    )


def stage1() -> int:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    db = S2A.SessionLocal()
    try:
        _, context, verification = S2U._prepare(db)
        control = context.prompts[PRIMARY_VARIANT]["provider"]
        treatment = _treatment_prompt(control)
        diff = _prompt_diff(control, treatment)
        diff["fixture_freeze"] = verification
        diff["outer_max_generated_tokens"] = OUTER_MAX_GENERATED_TOKENS
        diff["outer_deadline_seconds"] = OUTER_DEADLINE_SECONDS
        diff["frozen_call_order"] = list(FROZEN_CALL_ORDER)
        diff["provider_reasoning_controls_used"] = []
        options = _options().to_dict()
        diff["invocation_options_both_arms"] = options
        diff["extra_provider_options_absent"] = not options.get(
            "extra_provider_options"
        )

        (CAPTURE_DIR / "treatment-prompt.txt").write_text(treatment, encoding="utf-8")
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        S2A._write_json(EVIDENCE / "prompt-diff.json", diff)
        print(json.dumps({
            "control_prompt_sha256": diff["control_prompt_sha256"],
            "treatment_prompt_sha256": diff["treatment_prompt_sha256"],
            "prompt_char_delta": diff["prompt_char_delta"],
            "required_identical": diff["required_identical"],
            "sole_semantic_delta": diff["sole_semantic_delta_is_termination_instruction"],
            "experiment_valid": diff["experiment_valid"],
        }, indent=1))
        return 0 if diff["experiment_valid"] else 1
    finally:
        db.close()


# --------------------------------------------------------- stage 2: the pair ---


def _run_arm(runtime: Any, context: Any, arm: str) -> dict[str, Any]:
    control = context.prompts[PRIMARY_VARIANT]["provider"]
    prompt = control if arm == "CONTROL" else _treatment_prompt(control)
    options = _options()
    capture_path = CAPTURE_DIR / f"raw-{arm.lower()}.json"
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
                    "phase": "PHASE34-S2W",
                    "arm": arm,
                    "fixture_id": PRIMARY_FIXTURE,
                    "variant": PRIMARY_VARIANT,
                    "planning_attempt": "initial",
                    "repairs_allowed": False,
                    "execution_allowed": False,
                    "discovery_contract_capture_path": str(capture_path),
                    "discovery_contract_run_id": f"s2w-{arm.lower()}",
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
        "prompt_sha256": S2T._sha256(prompt),
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
        control = context.prompts[PRIMARY_VARIANT]["provider"]
        diff = _prompt_diff(control, _treatment_prompt(control))
        if not diff["experiment_valid"]:
            raise RuntimeError("Section 6 prompt diff failed; refusing to call")
        S2A._write_json(CAPTURE_DIR / "freeze.json", verification)
        for arm in FROZEN_CALL_ORDER:
            _run_arm(runtime, context, arm)
        return 0
    finally:
        db.close()


# ----------------------------------------------------------------- evidence ---


def _ambiguity(text: str) -> dict[str, Any]:
    lowered = text.lower()
    hits = {marker: lowered.count(marker) for marker in _AMBIGUITY_MARKERS}
    return {
        "marker_hits": {k: v for k, v in hits.items() if v},
        "distinct_markers": sum(1 for v in hits.values() if v),
        "total_hits": sum(hits.values()),
    }


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
    first_readiness = readiness.get("first_readiness_token_index")
    pre_readiness_tokens = first_readiness if first_readiness is not None else None

    # Section 11: early deliberation is the reasoning BEFORE the first readiness
    # declaration. Split on the same character offset the token index came from.
    if first_readiness is not None and total_chars:
        split_char = int(len(reasoning) * (first_readiness / max(reasoning_tokens, 1)))
        split_char = max(0, min(split_char, len(reasoning)))
    else:
        split_char = len(reasoning)
    pre_text, post_text = reasoning[:split_char], reasoning[split_char:]

    topics = S2T._topic_profile(reasoning)
    return {
        "arm": arm,
        "prompt_sha256": cell["prompt_sha256"],
        "finish_reason": cell["finish_reason"],
        "generation_outcome": cell["generation_outcome"],
        "prompt_tokens": cell["prompt_tokens"],
        "generated_tokens": generated,
        "generation_ceiling_reached": generated >= OUTER_MAX_GENERATED_TOKENS,
        "latency_ms": cell["latency_ms"],
        "time_to_first_token_ms": cell["time_to_first_token_ms"],
        "reasoning_chars": len(reasoning),
        "final_visible_output_chars": len(final),
        "reasoning_sha256": S2T._sha256(reasoning),
        "final_content_sha256": S2T._sha256(final),
        "estimated_reasoning_tokens": reasoning_tokens,
        "final_content_start_token": reasoning_tokens,
        "reasoning_share": (
            round(len(reasoning) / total_chars, 4) if total_chars else None
        ),
        "pre_readiness_tokens": pre_readiness_tokens,
        "post_readiness_tokens": readiness.get("tokens_after_first_readiness"),
        "post_readiness_share": readiness.get(
            "share_of_generation_after_first_readiness"
        ),
        "final_answer_transition": readiness,
        "self_critique_markers": topics.get("SELF_CORRECTION"),
        "finalization_attempt_markers": topics.get("FINALIZATION"),
        "topic_profile": topics,
        "repetition": S2T._repetition(reasoning),
        "ambiguity_whole_trace": _ambiguity(reasoning),
        "ambiguity_pre_readiness": _ambiguity(pre_text),
        "ambiguity_post_readiness": _ambiguity(post_text),
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


def _percent(control: Optional[float], treatment: Optional[float]) -> Optional[float]:
    if not control or control is None or treatment is None:
        return None
    return round(100.0 * (float(control) - float(treatment)) / float(control), 1)


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

    c_tr, t_tr = control["final_answer_transition"], treatment["final_answer_transition"]
    post_reduction = _percent(
        c_tr.get("tokens_after_first_readiness"), t_tr.get("tokens_after_first_readiness")
    )
    marker_reduction = _percent(
        c_tr.get("readiness_markers_total"), t_tr.get("readiness_markers_total")
    )
    density_reduction = _percent(
        c_tr.get("final_quarter_ritual_density_per_1k_tokens"),
        t_tr.get("final_quarter_ritual_density_per_1k_tokens"),
    )
    drafts_reduction = _percent(
        c_tr.get("full_json_plan_drafts_inside_reasoning"),
        t_tr.get("full_json_plan_drafts_inside_reasoning"),
    )

    # Section 15: a smaller post-readiness figure is NOT success if readiness
    # merely moved later while the ritual itself is unchanged.
    readiness_moved_later = bool(
        (t_tr.get("first_readiness_token_index") or 0)
        > (c_tr.get("first_readiness_token_index") or 0)
    )
    ritual_reduced = bool(
        (marker_reduction is not None and marker_reduction >= 25)
        or (density_reduction is not None and density_reduction >= 25)
        or (drafts_reduction is not None and drafts_reduction >= 25)
    )
    readiness_shift_artifact = bool(
        post_reduction is not None
        and post_reduction > 0
        and readiness_moved_later
        and not ritual_reduced
    )

    # Section 11: early deliberation must survive.
    c_amb = control["ambiguity_pre_readiness"]["distinct_markers"]
    t_amb = treatment["ambiguity_pre_readiness"]["distinct_markers"]
    control_considered = c_amb >= 3
    treatment_considered = t_amb >= 3
    pre_ratio = (
        (treatment["pre_readiness_tokens"] / control["pre_readiness_tokens"])
        if control["pre_readiness_tokens"] and treatment["pre_readiness_tokens"]
        else None
    )
    if control_considered and treatment_considered:
        early_preserved = "YES"
    elif control_considered and not treatment_considered:
        early_preserved = "NO"
    else:
        early_preserved = "UNCLEAR"

    rank = {"PASS": 2, "PARTIAL": 1, "FAIL": 0, "NOT_ADJUDICABLE": -1}

    def _not_worse(field: str) -> bool:
        return rank.get(str(treatment["quality"][field]), -1) >= rank.get(
            str(control["quality"][field]), -1
        )

    def _mutation_present(row: Mapping[str, Any]) -> bool:
        return bool(row["quality"].get("mutating_paths"))

    both_complete = all(
        row["finish_reason"] == "stop" and row["quality"]["parse_success"] is True
        for row in (control, treatment)
    )
    ceiling = any(row["generation_ceiling_reached"] for row in (control, treatment))
    quality_gate = {
        "both_arms_complete": both_complete,
        "control_adjudicable": (
            control["finish_reason"] == "stop"
            and control["quality"]["parse_success"] is True
        ),
        "parse_success": treatment["quality"]["parse_success"] is True,
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
        "control_mutation_present_when_required": _mutation_present(control),
        "treatment_mutation_present_when_required": _mutation_present(treatment),
    }
    quality_gate["passes"] = all(quality_gate.values())

    success_gate = {
        "a_both_arms_complete_plans": both_complete,
        "b_quality_preserved": quality_gate["passes"],
        "c_early_deliberation_preserved": early_preserved == "YES",
        "d_post_readiness_reduced_30pct": bool(
            post_reduction is not None and post_reduction >= 30
        ),
        "d_supporting_ritual_signal": ritual_reduced,
        "not_readiness_shift_artifact": not readiness_shift_artifact,
    }
    success_gate["directional_support"] = all(success_gate.values())

    shared_validator_confound = bool(
        set(control["quality"]["validator_finding_codes"] or [])
        & set(treatment["quality"]["validator_finding_codes"] or [])
    )

    reasoning_comparison = {
        "schema_version": "phase34-s2w-reasoning-comparison/1",
        "primary_fixture": PRIMARY_FIXTURE,
        "primary_variant": PRIMARY_VARIANT,
        "frozen_call_order": list(FROZEN_CALL_ORDER),
        "treatment_instruction": TREATMENT_INSTRUCTION,
        "termination_instruction_position": TERMINATION_INSTRUCTION_POSITION,
        "outer_max_generated_tokens": OUTER_MAX_GENERATED_TOKENS,
        "generation": {
            "control_generated_tokens": control["generated_tokens"],
            "treatment_generated_tokens": treatment["generated_tokens"],
            "control_reasoning_tokens": control["estimated_reasoning_tokens"],
            "treatment_reasoning_tokens": treatment["estimated_reasoning_tokens"],
            "control_reasoning_share": control["reasoning_share"],
            "treatment_reasoning_share": treatment["reasoning_share"],
            "control_finish_reason": control["finish_reason"],
            "treatment_finish_reason": treatment["finish_reason"],
            "control_prompt_tokens": control["prompt_tokens"],
            "treatment_prompt_tokens": treatment["prompt_tokens"],
            "prompt_token_delta": treatment["prompt_tokens"] - control["prompt_tokens"],
            "generation_ceiling_reached_by_either_arm": ceiling,
        },
        "termination": {
            "control_first_readiness_token": c_tr.get("first_readiness_token_index"),
            "treatment_first_readiness_token": t_tr.get("first_readiness_token_index"),
            "control_pre_readiness_tokens": control["pre_readiness_tokens"],
            "treatment_pre_readiness_tokens": treatment["pre_readiness_tokens"],
            "control_post_readiness_tokens": c_tr.get("tokens_after_first_readiness"),
            "treatment_post_readiness_tokens": t_tr.get("tokens_after_first_readiness"),
            "control_post_readiness_share": control["post_readiness_share"],
            "treatment_post_readiness_share": treatment["post_readiness_share"],
            "post_readiness_reduction_percent": post_reduction,
            "control_final_content_start_token": control["final_content_start_token"],
            "treatment_final_content_start_token": treatment[
                "final_content_start_token"
            ],
        },
        "ritual": {
            "control_readiness_markers": c_tr.get("readiness_markers_total"),
            "treatment_readiness_markers": t_tr.get("readiness_markers_total"),
            "readiness_marker_reduction_percent": marker_reduction,
            "control_final_quarter_density": c_tr.get(
                "final_quarter_ritual_density_per_1k_tokens"
            ),
            "treatment_final_quarter_density": t_tr.get(
                "final_quarter_ritual_density_per_1k_tokens"
            ),
            "final_quarter_density_reduction_percent": density_reduction,
            "control_full_json_drafts_in_reasoning": c_tr.get(
                "full_json_plan_drafts_inside_reasoning"
            ),
            "treatment_full_json_drafts_in_reasoning": t_tr.get(
                "full_json_plan_drafts_inside_reasoning"
            ),
            "full_json_draft_reduction_percent": drafts_reduction,
            "control_self_critique_markers": control["self_critique_markers"],
            "treatment_self_critique_markers": treatment["self_critique_markers"],
            "control_finalization_markers": control["finalization_attempt_markers"],
            "treatment_finalization_markers": treatment["finalization_attempt_markers"],
            "ritual_materially_reduced": ritual_reduced,
        },
        "early_deliberation": {
            "control_ambiguity_markers_pre_readiness": c_amb,
            "treatment_ambiguity_markers_pre_readiness": t_amb,
            "control_task_ambiguity_considered": control_considered,
            "treatment_task_ambiguity_considered": treatment_considered,
            "pre_readiness_token_ratio_treatment_over_control": (
                round(pre_ratio, 3) if pre_ratio else None
            ),
            "early_task_deliberation_preserved": early_preserved,
        },
        "section_15_artifact_check": {
            "readiness_moved_later_in_treatment": readiness_moved_later,
            "ritual_reduced": ritual_reduced,
            "readiness_shift_artifact": readiness_shift_artifact,
        },
        # Every run below used this identical frozen COMPACT B prompt under the
        # identical 5000/330s envelope. The unbounded arms alone span 2432-5000
        # generated tokens, which bounds the effect size any 2-call paired
        # design on this fixture can resolve.
        "cross_run_variance_same_frozen_prompt": {
            "unbounded_or_control_arms": {
                "phase34-s2a-r1 COMPACT B": {"generated": 5000, "finish": "length"},
                "phase34-s2t COMPACT B": {
                    "generated": 4108, "reasoning": 3770, "finish": "stop"
                },
                "phase34-s2u CONTROL": {
                    "generated": 4380, "reasoning": 4031, "finish": "stop"
                },
                "phase34-s2v CONTROL": {"generated": 5000, "finish": "length"},
                "phase34-s2w CONTROL": {
                    "generated": 2432, "reasoning": 2180, "finish": "stop"
                },
            },
            "treated_arms": {
                "phase34-s2u reasoning_effort=low": {
                    "generated": 4694, "reasoning": 4448
                },
                "phase34-s2v thinking_token_budget=3500": {
                    "generated": 3246, "reasoning": 2947
                },
                "phase34-s2w termination instruction": {
                    "generated": 4573, "reasoning": 4300
                },
            },
            "unbounded_generated_range": [2432, 5000],
            "unbounded_spread_factor": 2.06,
            "consequence": (
                "the S2W treatment (4573 generated / 4300 reasoning) falls "
                "inside the established unbounded range while the S2W control "
                "(2432 / 2180) is below every prior observation, so the "
                "arm-to-arm difference is dominated by run-to-run variance and "
                "the direction of the instruction's effect is not attributable"
            ),
        },
        "section_14_success_gate": success_gate,
        "falsification": {
            "f1_multi_rule_delta": False,
            "f2_necessary_deliberation_lost": early_preserved == "NO",
            "f3_readiness_shift_artifact": readiness_shift_artifact,
            "f4_earlier_final_emission": bool(
                treatment["final_content_start_token"]
                < control["final_content_start_token"]
            ),
            "f5_ritual_reduced": ritual_reduced,
            "f6_quality_worse": not quality_gate["passes"],
            "f7_no_effect": bool(
                (post_reduction is None or abs(post_reduction) < 15)
                and not ritual_reduced
            ),
            "f8_shared_validator_confound": shared_validator_confound,
            "f9_generation_ceiling": ceiling,
        },
    }
    quality_comparison = {
        "schema_version": "phase34-s2w-quality-comparison/1",
        "rows": {
            arm: {
                **arms[arm]["quality"],
                "finish_reason": arms[arm]["finish_reason"],
                "generation_outcome": arms[arm]["generation_outcome"],
                "mutation_present_when_required": _mutation_present(arms[arm]),
            }
            for arm in FROZEN_CALL_ORDER
        },
        "quality_non_inferiority_gate": quality_gate,
        "shared_validator_confound": shared_validator_confound,
        "known_confounds_preserved": {
            "lexical_existing_write_false_positive": "OBSERVED",
            "normalizer_semantic_drift": "OBSERVED",
            "adapter_reasoning_observability_gap": "OBSERVED",
        },
    }
    S2A._write_json(EVIDENCE / "reasoning-comparison.json", reasoning_comparison)
    S2A._write_json(EVIDENCE / "quality-comparison.json", quality_comparison)
    print(json.dumps(reasoning_comparison["generation"], indent=1))
    print(json.dumps(reasoning_comparison["termination"], indent=1))
    print(json.dumps(reasoning_comparison["ritual"], indent=1))
    print(json.dumps(reasoning_comparison["early_deliberation"], indent=1))
    print(json.dumps(reasoning_comparison["section_15_artifact_check"], indent=1))
    print(json.dumps(quality_gate, indent=1))
    print(json.dumps(success_gate, indent=1))
    print(json.dumps(reasoning_comparison["falsification"], indent=1))
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

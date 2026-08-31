"""PHASE34-S2T — Planning reasoning-termination characterization.

Evaluation harness only. No production file is modified, no Plan is executed,
no repair runs and no product row is created. The frozen Phase34-S2A corpus,
prompt builders and runtime identity are reused unchanged and re-verified
before every provider call.

The single addition over S2A-R1 is that the already-shipped
``DiscoveryContractCapture`` diagnostic seam is pointed at a temporary file so
the *raw* provider HTTP body is retained. That seam is consumed by the adapter
purely as capture configuration: the Planning role selects its system contract
from ``backend_role``, and the exact-contract wire payload is built from
``RuntimeInvocationOptions`` alone, so the outbound request is byte-identical
to the S2A-R1 request.

Usage:
    capture COMPACT|CURRENT    one provider call, raw body retained
    analyze                    provider-free analysis of retained captures
    evidence                   write the compact durable evidence set
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evals"))

import phase34s2a_planning_interface_ablation as S2A  # noqa: E402
import phase34s2ar1_generation_budget_controlled_ablation as R1  # noqa: E402
from app.services.agents.runtime_invocation import (  # noqa: E402
    RuntimeInvocationOptions,
)
from app.services.orchestration.planning.source_materialization import (  # noqa: E402
    materialize_planner_source_context,
)
from app.services.orchestration.planning.workspace_identity import (  # noqa: E402
    render_planner_workspace_identity,
)

PRIMARY_FIXTURE = "B"
EVALUATION_MAX_GENERATED_TOKENS = R1.EVALUATION_MAX_GENERATED_TOKENS
EVALUATION_DEADLINE_SECONDS = R1.EVALUATION_DEADLINE_SECONDS
MAX_PROVIDER_CALLS = 2

S2A_EVIDENCE = R1.S2A_EVIDENCE
R1_EVIDENCE = R1.EVIDENCE
CAPTURE_DIR = Path(
    "/tmp/claude-0/-root--openclaw-workspace-vault-projects-orchestrator"
    "/33e2968d-6200-4616-8657-d367b8c2814d/scratchpad/s2t"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- capture ---


def _budget_options() -> RuntimeInvocationOptions:
    return RuntimeInvocationOptions(
        timeout_seconds=EVALUATION_DEADLINE_SECONDS,
        max_output_tokens=EVALUATION_MAX_GENERATED_TOKENS,
    )


async def _invoke(runtime: Any, prompt: str, variant: str, capture_path: Path) -> Any:
    return await S2A.PlannerService._execute_task_with_planning_lock(
        runtime,
        prompt,
        timeout_seconds=EVALUATION_DEADLINE_SECONDS,
        reuse_task_session=False,
        diagnostic_label="PLANNING",
        diagnostic_metadata={
            "phase": "PHASE34-S2T",
            "fixture_id": PRIMARY_FIXTURE,
            "variant": variant,
            "planning_attempt": "initial",
            "repairs_allowed": False,
            "execution_allowed": False,
            "discovery_contract_capture_path": str(capture_path),
            "discovery_contract_run_id": uuid4().hex,
        },
        invocation_options=_budget_options(),
    )


# ------------------------------------------------------- corpus rebuilding ---

# The orchestrator container was restarted between S2A-R1 and S2T, so the
# tmpfs holding the frozen fixture workspace was remounted with a new device
# number (74 -> 66). ``version_identity`` is
# ``st_dev:st_ino:st_size:st_mtime_ns``, so every fixture's materialization
# metadata digest -- and therefore its semantic input digest -- moved, while
# file content, inode, size and mtime are byte-identical and the value never
# reaches the model. S2A's own rebuild gate cannot express that, so S2T owns a
# gate that renormalizes *only* st_dev and then demands exact equality on every
# frozen digest, plus byte-identity of both provider-bound prompts.
S2A_FROZEN_ST_DEV = "74"


def _renormalize_st_dev(metadata: Mapping[str, Any]) -> dict[str, Any]:
    patched = json.loads(json.dumps(metadata))
    changed = []
    for record in patched.get("files") or []:
        identity = record.get("version_identity")
        if not identity:
            continue
        parts = str(identity).split(":")
        if parts[0] != S2A_FROZEN_ST_DEV:
            changed.append({"observed": identity, "field": "st_dev"})
            parts[0] = S2A_FROZEN_ST_DEV
            record["version_identity"] = ":".join(parts)
    return {"metadata": patched, "renormalized": changed}


def _rebuild_fixture_b(db: Any, freeze: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    workspace_root = Path(str(freeze["workspace_root"]))
    if not workspace_root.is_dir() or not workspace_root.name.startswith(
        "phase34-s2a-"
    ):
        raise RuntimeError("Frozen temporary workspace root is unavailable or unsafe")
    runtime = S2A._runtime_freeze(db)
    if S2A._json_digest(runtime) != freeze["runtime_freeze_digest"]:
        raise RuntimeError("Effective Planning runtime changed after freeze")

    spec = S2A._fixtures()[PRIMARY_FIXTURE]
    frozen = freeze["fixtures"][PRIMARY_FIXTURE]
    workspace = (
        workspace_root / f"fixture-{PRIMARY_FIXTURE.lower()}-{spec.name.lower()}"
    )
    identity = S2A._identity(workspace, spec)
    observation_payload = frozen.get("discovery_observation")
    observation = None
    if observation_payload:
        observation = S2A.DiscoveryObservation(
            action=str(observation_payload["action"]),
            status=str(observation_payload["status"]),
            paths=tuple(observation_payload.get("paths") or ()),
            hits=tuple(
                S2A.SearchHit(**item)
                for item in observation_payload.get("hits") or ()
            ),
            content=observation_payload.get("content"),
            truncated=bool(observation_payload.get("truncated")),
            reason=observation_payload.get("reason"),
        )
    materialization = materialize_planner_source_context(
        workspace,
        task_description=spec.task,
        expected_paths=spec.grounded_paths,
        supporting_paths=(observation.materialization_paths() if observation else ()),
        workspace_identity=identity,
    )
    context = S2A._build_context(
        db,
        workspace,
        spec,
        runtime,
        frozen_materialization=materialization,
        frozen_observation=observation,
    )

    renormalized = _renormalize_st_dev(materialization.to_metadata())
    semantic_input = {
        "task": spec.task,
        "intent_mode": spec.intent_mode,
        "project_context": context.project_context,
        "structure_capsule": context.structure_capsule,
        "python_source_context": context.python_source_context,
        "source_stub_context": context.source_stub_context,
        "source_materialization": renormalized["metadata"],
        "discovery_observation": asdict(observation) if observation else None,
        "knowledge_context": None,
        "workspace_identity": render_planner_workspace_identity(context.identity),
        "execution_profile": "full_lifecycle",
        "workflow_profile": "default",
        "execution_topology": runtime["execution_topology"],
    }
    gate = {
        "workspace_content_digest_match": (
            S2A._workspace_content_digest(workspace)
            == frozen["workspace_content_digest"]
        ),
        "project_context_digest_match": (
            S2A._sha256_text(context.project_context)
            == frozen["project_context_digest"]
        ),
        "source_materialization_digest_match_after_st_dev_renormalization": (
            S2A._json_digest(renormalized["metadata"])
            == frozen["source_materialization_digest"]
        ),
        "semantic_input_digest_match_after_st_dev_renormalization": (
            S2A._json_digest(semantic_input) == frozen["semantic_input_digest"]
        ),
        "st_dev_renormalization_applied": renormalized["renormalized"],
        "observed_st_dev": (
            workspace.stat().st_dev if workspace.exists() else None
        ),
        "frozen_st_dev": int(S2A_FROZEN_ST_DEV),
        "version_identity_reaches_model": any(
            "version_identity" in context.prompts[variant]["provider"]
            for variant in S2A.VARIANTS
        ),
    }
    gate["corpus_verified"] = (
        gate["workspace_content_digest_match"]
        and gate["project_context_digest_match"]
        and gate["source_materialization_digest_match_after_st_dev_renormalization"]
        and gate["semantic_input_digest_match_after_st_dev_renormalization"]
        and not gate["version_identity_reaches_model"]
    )
    return context, gate


def _verify_fixture_b(freeze: Mapping[str, Any], context: Any) -> dict[str, Any]:
    """Re-prove both Fixture B provider-bound prompts are the frozen S2A ones."""

    stores = {
        variant: json.loads(
            (S2A_EVIDENCE / f"{variant.lower()}-results.json").read_text(
                encoding="utf-8"
            )
        )
        for variant in S2A.VARIANTS
    }
    entry: dict[str, Any] = {
        "fixture_id": PRIMARY_FIXTURE,
        "workspace_digest": S2A._workspace_content_digest(context.workspace),
        "variants": {},
    }
    mismatches: list[str] = []
    for variant in S2A.VARIANTS:
        rebuilt = context.prompts[variant]["provider"]
        row = [
            item
            for item in stores[variant]["results"]
            if str(item["fixture_id"]) == PRIMARY_FIXTURE
        ][0]
        expected = row["prompt"]["final_provider_bound_prompt_sha256"]
        actual = _sha256(rebuilt)
        entry["variants"][variant] = {
            "s2a_provider_prompt_sha256": expected,
            "rebuilt_provider_prompt_sha256": actual,
            "byte_identical": actual == expected,
            "prompt_chars": len(rebuilt),
        }
        if actual != expected:
            mismatches.append(f"{variant}:prompt")
        if row.get("workspace_digest_before") != entry["workspace_digest"]:
            mismatches.append(f"{variant}:workspace")
    entry["mismatches"] = sorted(set(mismatches))
    entry["prompt_freeze_verified"] = not mismatches
    return entry


def capture(variant: str) -> int:
    if variant not in S2A.VARIANTS:
        raise SystemExit(f"variant must be one of {S2A.VARIANTS}")
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    prior = sorted(CAPTURE_DIR.glob("raw-*.json"))
    if len(prior) >= MAX_PROVIDER_CALLS:
        raise SystemExit(f"MAX_PROVIDER_CALLS={MAX_PROVIDER_CALLS} already spent")

    freeze = json.loads(
        (S2A_EVIDENCE / "fixture-freeze.json").read_text(encoding="utf-8")
    )
    db = S2A.SessionLocal()
    try:
        context, corpus_gate = _rebuild_fixture_b(db, freeze)
        verification = _verify_fixture_b(freeze, context)
        verification["corpus_gate"] = corpus_gate
        if not corpus_gate["corpus_verified"]:
            print("CORPUS_VERIFIED = NO", json.dumps(corpus_gate, indent=1))
            return 1
        if not verification["prompt_freeze_verified"]:
            print("PROMPT_FREEZE_VERIFIED = NO", verification["mismatches"])
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
            raise RuntimeError(f"Runtime identity changed: {identity}")

        quiet, before = R1._wait_quiescent()
        if not quiet:
            print("Provider not quiescent before call")
            return 1

        capture_path = CAPTURE_DIR / f"raw-{variant.lower()}-b.json"
        before_workspace = S2A._workspace_content_digest(context.workspace)
        started = time.monotonic()
        error = None
        response: Any = None
        try:
            response = asyncio.run(
                _invoke(
                    runtime,
                    context.prompts[variant]["provider"],
                    variant,
                    capture_path,
                )
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
        latency_ms = round((time.monotonic() - started) * 1000)

        quiet_after, after = R1._wait_quiescent()
        delta = R1._delta(before, after)
        after_workspace = S2A._workspace_content_digest(context.workspace)
        if before_workspace != after_workspace:
            raise RuntimeError("Planning call mutated frozen fixture B")

        ttft_count = float(delta.get("time_to_first_token_seconds_count") or 0)
        outcome, detail = R1._classify_transport(
            error=error, delta=delta, first_token=ttft_count >= 1
        )
        record = {
            "schema_version": "phase34-s2t-cell/1",
            "phase": "PHASE34-S2T",
            "fixture_id": PRIMARY_FIXTURE,
            "fixture_name": context.spec.name,
            "variant": variant,
            "runtime_identity": identity,
            "temperature": freeze["runtime_freeze"]["temperature"],
            "evaluation_max_generated_tokens": EVALUATION_MAX_GENERATED_TOKENS,
            "evaluation_deadline_seconds": EVALUATION_DEADLINE_SECONDS,
            "prompt_freeze": verification,
            "generation_outcome": outcome,
            "generation_outcome_detail": detail,
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
            "generation_duration_ms": round(
                float(delta.get("request_decode_time_seconds_sum") or 0) * 1000
            ),
            "server_preemptions": delta.get("num_preemptions_total"),
            "quiescent_after": quiet_after,
            "counter_delta": delta,
            "workspace_digest_before": before_workspace,
            "workspace_digest_after": after_workspace,
            "adapter_returned_output_chars": len(
                str((response or {}).get("output") or "")
            ),
            "capture_path": str(capture_path),
        }
        (CAPTURE_DIR / f"cell-{variant.lower()}-b.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )
        print(
            f"{variant}:B {outcome} gen={record['generated_tokens']} "
            f"lat={latency_ms}ms adapter_chars={record['adapter_returned_output_chars']}"
        )
        return 0
    finally:
        db.close()


# --------------------------------------------------------------- analysis ---

_SENT = re.compile(r"[^.!?\n]+[.!?]?")


def _sentences(text: str) -> list[str]:
    out = []
    for raw in _SENT.findall(text):
        stripped = " ".join(raw.split())
        if len(stripped) >= 12:
            out.append(stripped)
    return out


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _longest_repeated_span(tokens: list[str]) -> int:
    """Longest token n-gram occurring at least twice (binary search on length)."""

    low, high, best = 1, min(len(tokens) // 2, 400), 0
    while low <= high:
        mid = (low + high) // 2
        seen: set[tuple[str, ...]] = set()
        found = False
        for gram in _ngrams(tokens, mid):
            if gram in seen:
                found = True
                break
            seen.add(gram)
        if found:
            best, low = mid, mid + 1
        else:
            high = mid - 1
    return best


def _repetition(text: str) -> dict[str, Any]:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 12]
    sentences = _sentences(text)
    tokens = re.findall(r"[a-z0-9_./\-]+", text.lower())
    line_counts: dict[str, int] = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1
    sent_counts: dict[str, int] = {}
    for sentence in sentences:
        key = sentence.lower()
        sent_counts[key] = sent_counts.get(key, 0) + 1
    grams = _ngrams(tokens, 12)
    gram_counts: dict[tuple[str, ...], int] = {}
    for gram in grams:
        gram_counts[gram] = gram_counts.get(gram, 0) + 1
    return {
        "line_count": len(lines),
        "sentence_count": len(sentences),
        "token_count": len(tokens),
        "repeated_exact_lines": sum(c - 1 for c in line_counts.values() if c > 1),
        "distinct_repeated_lines": sum(1 for c in line_counts.values() if c > 1),
        "repeated_sentence_ratio": (
            round(sum(c - 1 for c in sent_counts.values() if c > 1) / len(sentences), 4)
            if sentences
            else None
        ),
        "repeated_ngram_ratio_12": (
            round(sum(c - 1 for c in gram_counts.values() if c > 1) / len(grams), 4)
            if grams
            else None
        ),
        "longest_repeated_span_tokens": _longest_repeated_span(tokens),
        "top_repeated_lines": sorted(
            (
                {"count": c, "excerpt": line[:120]}
                for line, c in line_counts.items()
                if c > 1
            ),
            key=lambda item: -item["count"],
        )[:10],
    }


# Section 12 contract-load lexicon. Deterministic substring counting only.
_TOPICS: dict[str, tuple[str, ...]] = {
    "PLAN_CONTRACT": (
        "json", "schema", "field", "expected_files", "operation", "step",
        "format", "array", "key", "\"action\"", "structure", "output format",
    ),
    "VERIFICATION": (
        "verification", "verify", "verification_command", "pytest", "test command",
        "build", "check command",
    ),
    "EXISTING_NEW": (
        "existing", "new file", "create", "modify", "edit", "already exists",
    ),
    "PATHS_AND_GROUNDING": (
        "path", "directory", "workspace", "source", "file listed", "grounded",
        "do not invent", "speculative",
    ),
    "ORCHESTRATOR_RULES": (
        "rule", "instruction", "requirement", "must ", "should ", "not allowed",
        "constraint", "guideline", "the prompt says", "as specified",
    ),
    "TASK_IMPLEMENTATION": (
        "function", "implement", "code", "logic", "variable", "import", "class ",
        "return", "algorithm", "bug", "behaviour", "behavior",
    ),
    "SOURCE_ANALYSIS": (
        "the file contains", "current content", "looking at", "the source",
        "reading", "line ", "def ", "it currently",
    ),
    "SELF_CORRECTION": (
        "wait", "actually", "but ", "hmm", "let me reconsider", "on second thought",
        "however", "re-check", "recheck", "let me redo", "instead", "revise",
    ),
    "FINALIZATION": (
        "final answer", "so the plan", "final plan", "let me write", "output:",
        "here is the", "putting it together", "so, the json",
    ),
}


def _topic_profile(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        topic: sum(lowered.count(term) for term in terms)
        for topic, terms in _TOPICS.items()
    }


def _quartiles(text: str) -> list[dict[str, Any]]:
    tokens = re.findall(r"\S+", text)
    size = max(len(tokens) // 4, 1)
    out = []
    seen_terms: set[str] = set()
    for index in range(4):
        chunk = tokens[index * size : (index + 1) * size if index < 3 else len(tokens)]
        body = " ".join(chunk)
        terms = set(re.findall(r"[a-z0-9_./\-]{4,}", body.lower()))
        out.append(
            {
                "quartile": f"Q{index + 1}",
                "tokens": len(chunk),
                "topic_profile": _topic_profile(body),
                "new_term_count": len(terms - seen_terms),
                "new_term_ratio": (
                    round(len(terms - seen_terms) / len(terms), 4) if terms else None
                ),
                "repetition": _repetition(body),
                "head_excerpt": body[:400],
            }
        )
        seen_terms |= terms
    return out


def _split_reasoning(raw_body: Mapping[str, Any]) -> dict[str, Any]:
    """Locate the reasoning and the final content in the raw provider body."""

    choices = raw_body.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    content = message.get("content")
    reasoning_keys = [
        key for key in message if "reason" in key.lower() or "think" in key.lower()
    ]
    reasoning = None
    representation = None
    for key in reasoning_keys:
        value = message.get(key)
        if isinstance(value, str) and value:
            reasoning, representation = value, f"choices[0].message.{key}"
            break
    inline = False
    if reasoning is None and isinstance(content, str) and "<think>" in content:
        match = re.search(r"<think>(.*?)(?:</think>|$)", content, flags=re.DOTALL)
        if match:
            reasoning = match.group(1)
            representation = "inline <think> in choices[0].message.content"
            content = re.sub(
                r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL
            ).strip()
            inline = True
    return {
        "message_keys": sorted(str(key) for key in message),
        "reasoning_keys_present": reasoning_keys,
        "reasoning_representation": representation,
        "reasoning_inline_in_content": inline,
        "reasoning": reasoning,
        "final_content": content if isinstance(content, str) else "",
        "usage": raw_body.get("usage"),
        "finish_reason": first.get("finish_reason"),
    }


def analyze() -> int:
    out: dict[str, Any] = {"schema_version": "phase34-s2t-analysis/1", "cells": {}}
    for path in sorted(CAPTURE_DIR.glob("cell-*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        capture_doc = json.loads(
            Path(cell["capture_path"]).read_text(encoding="utf-8")
        )
        raw_text = None
        for key in ("http_response", "response"):
            section = capture_doc.get(key)
            if isinstance(section, Mapping) and section.get("raw_body_text"):
                raw_text = section["raw_body_text"]
                break
        if raw_text is None:
            raw_text = json.dumps(capture_doc.get("raw_body_text") or "")
        body = json.loads(raw_text)
        split = _split_reasoning(body)
        reasoning = split["reasoning"] or ""
        final = split["final_content"] or ""
        generated = float(cell["generated_tokens"])
        chars = len(reasoning) + len(final)
        out["cells"][cell["variant"]] = {
            "variant": cell["variant"],
            "fixture_id": cell["fixture_id"],
            "finish_reason": cell["finish_reason"],
            "generation_outcome": cell["generation_outcome"],
            "generated_tokens": generated,
            "prompt_tokens": cell["prompt_tokens"],
            "latency_ms": cell["latency_ms"],
            "time_to_first_token_ms": cell["time_to_first_token_ms"],
            "adapter_returned_output_chars": cell["adapter_returned_output_chars"],
            "message_keys": split["message_keys"],
            "reasoning_representation": split["reasoning_representation"],
            "reasoning_inline_in_content": split["reasoning_inline_in_content"],
            "usage": split["usage"],
            "reasoning_chars": len(reasoning),
            "final_content_chars": len(final),
            "reasoning_sha256": _sha256(reasoning),
            "final_content_sha256": _sha256(final),
            "estimated_reasoning_tokens": (
                round(generated * len(reasoning) / chars) if chars else None
            ),
            "estimated_first_final_content_token_index": (
                round(generated * len(reasoning) / chars) if chars and final else None
            ),
            "repetition": _repetition(reasoning),
            "topic_profile": _topic_profile(reasoning),
            "quartiles": _quartiles(reasoning),
            "reasoning_head": reasoning[:1200],
            "reasoning_tail": reasoning[-1200:],
            "final_content_head": final[:1200],
        }
    (CAPTURE_DIR / "analysis.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: {
        "finish_reason": v["finish_reason"],
        "generated_tokens": v["generated_tokens"],
        "reasoning_representation": v["reasoning_representation"],
        "reasoning_chars": v["reasoning_chars"],
        "final_content_chars": v["final_content_chars"],
    } for k, v in out["cells"].items()}, indent=1))
    return 0


# --------------------------------------------------------------- evidence ---

EVIDENCE = ROOT / "docs/roadmap/reports/evidence/phase34-s2t"

# Section 13 / Section 10: deterministic finalization-ritual lexicon. These are
# the phrases with which this model announces that it is ready to leave
# reasoning mode. Counting them is how "the Plan was ready" is measured without
# an LLM judge.
_READINESS_MARKERS = (
    "plan looks solid", "looks solid", "matches perfectly", "no extra text",
    "i will generate the json", "i will output", "i'll produce the json",
    "i will produce the json", "ready.", "all set", "all good",
    "output matches", "proceeds", "done.", "output generation", "final answer",
)


def _readiness(reasoning: str, generated: float, total_chars: int) -> dict[str, Any]:
    lowered = reasoning.lower()
    offsets = sorted(
        match.start()
        for marker in _READINESS_MARKERS
        for match in re.finditer(re.escape(marker), lowered)
    )
    if not offsets:
        return {"readiness_markers_total": 0, "first_readiness_token_index": None}

    def token(offset: int) -> int:
        return round(generated * offset / total_chars)

    final_quarter = len(reasoning) * 3 // 4
    in_final_quarter = sum(1 for offset in offsets if offset >= final_quarter)
    reasoning_end = token(len(reasoning))
    return {
        "readiness_markers_total": len(offsets),
        "readiness_markers_in_final_quarter": in_final_quarter,
        "first_readiness_token_index": token(offsets[0]),
        "last_readiness_token_index": token(offsets[-1]),
        "tokens_after_first_readiness": reasoning_end - token(offsets[0]),
        "share_of_generation_after_first_readiness": round(
            (reasoning_end - token(offsets[0])) / generated, 4
        ),
        "final_quarter_ritual_density_per_1k_tokens": round(
            1000 * in_final_quarter / max(reasoning_end - token(final_quarter), 1), 1
        ),
        "full_json_plan_drafts_inside_reasoning": len(
            re.findall(r'"step_number"\s*:\s*1[,\s]', reasoning)
        ),
    }


def evidence() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((CAPTURE_DIR / "analysis.json").read_text(encoding="utf-8"))
    cells = analysis["cells"]

    contract: dict[str, Any] = {
        "schema_version": "phase34-s2t-runtime-reasoning-contract/1",
        "raw_server_reasoning_available": True,
        "reasoning_representation": "choices[0].message.reasoning (dedicated field)",
        "final_content_representation": "choices[0].message.content",
        "message_keys_observed": cells["COMPACT"]["message_keys"],
        "where_reasoning_is_removed": (
            "The server's reasoning parser separates the two before the wire. "
            "app/services/agents/providers/openai_chat_adapter.py"
            "::_extract_chat_completion_content reads only message.content, so "
            "message.reasoning is never read by the adapter. _strip_thinking() "
            "removes inline <think> blocks and is a no-op on this deployment."
        ),
        "can_evaluation_capture_reasoning_without_production_change": True,
        "capture_seam": (
            "the already-shipped DiscoveryContractCapture diagnostic seam "
            "(discovery_contract_capture_path in diagnostic_metadata) retains the "
            "raw HTTP body; the Planning role selects its system contract from "
            "backend_role and the exact-contract payload is built from "
            "RuntimeInvocationOptions alone, so the wire request is unchanged"
        ),
        "eos_observed_in_successful_cells": True,
        "reasoning_to_final_transition_provider_supported": True,
        "usage_reports_reasoning_tokens_separately": False,
        "streaming_enabled": False,
        "time_to_first_final_content_measurable": False,
        "adapter_observability_gap": (
            "When generation is cut by the budget while still inside reasoning, "
            "message.content is empty and Planning receives an empty string with "
            "no signal that thousands of reasoning tokens were produced. This is "
            "exactly S2A-R1 COMPACT A/B/C (visible_output_chars = 0). Recorded, "
            "not repaired: Section 2 forbids adapter changes."
        ),
    }
    S2A._write_json(EVIDENCE / "runtime-reasoning-contract.json", contract)

    trajectory: dict[str, Any] = {
        "schema_version": "phase34-s2t-reasoning-trajectory-summary/1",
        "primary_fixture": PRIMARY_FIXTURE,
        "primary_fixture_name": "GROUNDED_EXISTING_EDIT",
        "evaluation_max_generated_tokens": EVALUATION_MAX_GENERATED_TOKENS,
        "cells": {},
    }
    repetition: dict[str, Any] = {
        "schema_version": "phase34-s2t-repetition-analysis/1",
        "method": (
            "deterministic text statistics only -- no embeddings, no LLM judge"
        ),
        "cells": {},
    }
    for variant, cell in cells.items():
        total_chars = cell["reasoning_chars"] + cell["final_content_chars"]
        raw = json.loads(
            (CAPTURE_DIR / f"raw-{variant.lower()}-b.json").read_text(encoding="utf-8")
        )
        body = json.loads(raw["response"]["raw_body_text"])
        reasoning = body["choices"][0]["message"]["reasoning"] or ""
        trajectory["cells"][variant] = {
            "variant": variant,
            "finish_reason": cell["finish_reason"],
            "generation_outcome": cell["generation_outcome"],
            "prompt_tokens": cell["prompt_tokens"],
            "generated_tokens": cell["generated_tokens"],
            "latency_ms": cell["latency_ms"],
            "time_to_first_token_ms": cell["time_to_first_token_ms"],
            "reasoning_chars": cell["reasoning_chars"],
            "final_content_chars": cell["final_content_chars"],
            "reasoning_sha256": cell["reasoning_sha256"],
            "final_content_sha256": cell["final_content_sha256"],
            "estimated_reasoning_tokens": cell["estimated_reasoning_tokens"],
            "first_final_content_token_index": cell[
                "estimated_first_final_content_token_index"
            ],
            "reasoning_share_of_generation": round(
                cell["reasoning_chars"] / total_chars, 4
            ),
            "final_answer_transition": _readiness(
                reasoning, float(cell["generated_tokens"]), total_chars
            ),
            "topic_profile": cell["topic_profile"],
            "quartiles": [
                {
                    "quartile": quartile["quartile"],
                    "tokens": quartile["tokens"],
                    "new_term_ratio": quartile["new_term_ratio"],
                    "topic_profile": quartile["topic_profile"],
                    "repeated_sentence_ratio": quartile["repetition"][
                        "repeated_sentence_ratio"
                    ],
                }
                for quartile in cell["quartiles"]
            ],
            "bounded_excerpts": {
                "reasoning_head_400": cell["reasoning_head"][:400],
                "reasoning_tail_600": cell["reasoning_tail"][-600:],
            },
        }
        repetition["cells"][variant] = cell["repetition"]
    S2A._write_json(EVIDENCE / "reasoning-trajectory-summary.json", trajectory)
    S2A._write_json(EVIDENCE / "repetition-analysis.json", repetition)

    comparison = {
        "schema_version": "phase34-s2t-optional-current-comparison/1",
        "current_b_comparison_run": "CURRENT" in cells,
        "compared": [
            "reasoning trajectory",
            "repetition",
            "token index of tentative Plan formation",
            "token index of final transition",
            "contract-focused reasoning share",
        ],
        "not_compared": ["Plan quality (Section 15 forbids it)"],
        "rows": {
            variant: {
                "prompt_tokens": cells[variant]["prompt_tokens"],
                "generated_tokens": cells[variant]["generated_tokens"],
                "estimated_reasoning_tokens": cells[variant][
                    "estimated_reasoning_tokens"
                ],
                "reasoning_share_of_generation": trajectory["cells"][variant][
                    "reasoning_share_of_generation"
                ],
                "first_readiness_token_index": trajectory["cells"][variant][
                    "final_answer_transition"
                ]["first_readiness_token_index"],
                "share_of_generation_after_first_readiness": trajectory["cells"][
                    variant
                ]["final_answer_transition"][
                    "share_of_generation_after_first_readiness"
                ],
                "repeated_sentence_ratio": cells[variant]["repetition"][
                    "repeated_sentence_ratio"
                ],
                "repeated_ngram_ratio_12": cells[variant]["repetition"][
                    "repeated_ngram_ratio_12"
                ],
                "longest_repeated_span_tokens": cells[variant]["repetition"][
                    "longest_repeated_span_tokens"
                ],
            }
            for variant in cells
        },
    }
    S2A._write_json(EVIDENCE / "optional-current-comparison.json", comparison)
    print(f"evidence written to {EVIDENCE}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    command = sys.argv[1]
    if command == "capture":
        return capture(sys.argv[2].upper())
    if command == "analyze":
        return analyze()
    if command == "evidence":
        return evidence()
    print(f"unknown command {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

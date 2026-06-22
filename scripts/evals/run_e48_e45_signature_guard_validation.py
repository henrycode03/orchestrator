#!/usr/bin/env python3
"""E48 observability harness for E45 bounded debug repair validation.

Read event JSONL and a workspace after a live task run.  It deliberately does not
dispatch work or modify the workspace.  Invoke once per task, then aggregate the
JSON records emitted to stdout.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
from pathlib import Path

REASON = "bounded_execution_debug_repair_signature_contract_violation"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def verify_e45_imports() -> dict[str, bool]:
    from app.services.orchestration.diagnostics.signature_guard import (
        BOUNDED_DEBUG_REPAIR_SIGNATURE_VIOLATION_REASON,
        check_bounded_debug_repair_signature_contract,
    )
    return {
        "guard_imported": callable(check_bounded_debug_repair_signature_contract),
        "reason_imported": bool(BOUNDED_DEBUG_REPAIR_SIGNATURE_VIOLATION_REASON),
        "reason_exact": BOUNDED_DEBUG_REPAIR_SIGNATURE_VIOLATION_REASON == REASON,
        "guard_source_available": "check_bounded_debug_repair_signature_contract" in inspect.getsource(check_bounded_debug_repair_signature_contract),
    }


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = node.args
    result = [arg.arg for arg in args.posonlyargs + args.args]
    if args.vararg:
        result.append(f"*{args.vararg.arg}")
    result.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg:
        result.append(f"**{args.kwarg.arg}")
    return result


def inspect_effective_signature(path: Path, name: str = "format_summary") -> dict[str, object]:
    result: dict[str, object] = {"effective_signature": None, "definitions": [], "duplicate_definitions": False, "wrong_later_duplicate_shadowing": False}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        result["parse_error"] = str(exc)
        return result
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    signatures = [_sig(node) for node in definitions]
    result["definitions"] = signatures
    result["duplicate_definitions"] = len(signatures) > 1
    result["effective_signature"] = signatures[-1] if signatures else None
    result["wrong_later_duplicate_shadowing"] = len(signatures) > 1 and signatures[-1] != signatures[0]
    return result


def inspect_events(event_log: Path) -> dict[str, object]:
    result: dict[str, object] = {"eligible": False, "invoked": False, "generated": False, "rejected_by_signature_guard": False, "skipped_or_not_reached": True, "output_hashes": [], "output_changed_paths": [], "violations": [], "skip_reasons": []}
    if not event_log.exists():
        result["skip_reasons"] = ["event_log_missing"]
        return result
    for raw in event_log.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        typ, details = str(event.get("event_type", "")).lower(), event.get("details") or {}
        scope = details.get("debug_repair_scope") == "bounded_execution_debug_repair"
        eligible = details.get("debug_repair_eligible")
        if eligible is True:
            result["eligible"] = True
        if typ == "debug_repair_attempted" and scope:
            result["invoked"] = True
        if typ == "repair_generated" and scope:
            result["generated"] = True
            if details.get("repair_output_sha256"):
                result["output_hashes"].append(details["repair_output_sha256"])
            result["output_changed_paths"].extend(details.get("repair_output_changed_paths") or [])
        if typ == "repair_rejected" and scope:
            reason = details.get("reason") or details.get("debug_repair_terminal_reason") or details.get("rejection_reason")
            if reason == REASON:
                result["rejected_by_signature_guard"] = True
                result["violations"].extend(details.get("bounded_execution_debug_repair_signature_violations") or [])
        if scope and details.get("debug_repair_terminal_reason"):
            result["skip_reasons"].append(details["debug_repair_terminal_reason"])
    result["skipped_or_not_reached"] = not bool(result["invoked"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    event_log = args.workspace / ".agent" / "events" / f"session_{args.session_id}_task_{args.task_id}.jsonl"
    print(json.dumps({"live_import": verify_e45_imports(), "event_signals": inspect_events(event_log), "workspace_signature": inspect_effective_signature(args.workspace / "src" / "medium_cli" / "formatting.py")}, indent=2))


if __name__ == "__main__":
    main()

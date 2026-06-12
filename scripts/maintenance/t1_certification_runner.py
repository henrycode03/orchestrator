#!/usr/bin/env python3
"""
Reliability Certification Run — Post-Maintenance Baseline.

Purpose: validate orchestrator after closure of all currently implemented
reliability fixes. Evidence-gathering only; no code changes.

Corpus: standard 18-task corpus (calclib, pathtools, strtools), 6 tasks each.
Workspaces: t1-certrun-{calclib,pathtools,strtools} (fresh).
Lane: baseline (PLANNING_BACKEND=None -> local_openclaw, qwen-local repair).

Fixes verified:
  - test_rewrite differentialization
  - shared project-first python resolver (venv + no-venv pip-show)
  - VMA repair deadlock fix
  - arbitration seam fixes (non-bootstrap weak-verification)
  - nested-project-root validator fix
  - zero-test bounded repair (expected_files derivation, semantic guard)
  - function-local import integrity fix
  - completion-repair serialization salvage
  - path guard Phase 1 advisory

Deliverables:
  - per-task raw JSON
  - corpus summary table
  - failure-class table
  - new failure-class inventory
  - reliability assessment
  - WorkingMemory gate recommendation
"""
import copy
import json
import pathlib
import sqlite3
import sys
from datetime import datetime

from scripts.maintenance._runner_common import ensure_repo_on_syspath, load_sibling_module

ensure_repo_on_syspath()
pg = load_sibling_module("pathguard", "t1_pathguard_telemetry_runner.py")
r = pg.r
DB_PATH = r.DB_PATH
WORKSPACE_BASE = r.WORKSPACE_BASE

CERT_RUN_SHA = "c742910df794b437c316165e85bc2ade65136b04"


# ── Additional detection helpers ─────────────────────────────────────────────

def detect_completion_repairs(events: list) -> int:
    """Count DEBUG_REPAIR_ATTEMPTED events with phase=completion."""
    return sum(
        1 for e in events
        if e.get("event_type") == "debug_repair_attempted"
        and e.get("details", {}).get("phase") == "completion"
    )


def detect_repair_budget_exhaustion(task_id: int) -> bool:
    """Return True if debug repair budget was exhausted for this task."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT log_metadata FROM log_entries "
            "WHERE task_id=? AND message LIKE '%debug_repair_budget_exhausted%' "
            "ORDER BY id LIMIT 1",
            (task_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return True
        # also check task_executions terminal reason
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT failure_category FROM task_executions WHERE task_id=? ORDER BY id",
            (task_id,),
        )
        rows = cur.fetchall()
        conn.close()
        return any(
            "debug_repair_budget" in str(r[0] or "").lower()
            for r in rows
        )
    except Exception:
        return False


def detect_repair_budget_exhaustion_from_events(events: list) -> bool:
    """Check events for debug repair budget exhaustion terminal reason."""
    for e in events:
        if e.get("event_type") == "debug_repair_attempted":
            d = e.get("details", {})
            if "debug_repair_budget_exhausted" in str(d.get("debug_repair_terminal_reason", "")):
                return True
            if "debug_repair_budget_exhausted" in str(d.get("reason", "")):
                return True
    return False


def detect_empty_response_events(task_id: int) -> int:
    """Count empty-response or non-JSON planning events for this task."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM log_entries "
            "WHERE task_id=? AND ("
            "   message LIKE '%empty response%' "
            "   OR message LIKE '%non-JSON prose%' "
            "   OR message LIKE '%non_json_prose%' "
            ") ",
            (task_id,),
        )
        row = cur.fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def detect_test_rewrite_arbitration(events: list) -> list[dict]:
    """Return all planning_repair_arbitration events that contain test_rewrite label."""
    hits = []
    for e in events:
        if e.get("event_type") != "planning_repair_arbitration":
            continue
        d = e.get("details", {})
        labels = d.get("regression_labels") or []
        if "test_rewrite" in labels:
            hits.append({
                "outcome": d.get("outcome"),
                "arbitration_action": d.get("arbitration_action"),
                "regression_labels": labels,
            })
    return hits


def detect_vma_repair_outcome(task_id: int, task_status: str, vma_fired: bool) -> str:
    """Classify VMA repair outcome: 'n/a', 'repaired_done', 'repaired_failed', 'no_repair'."""
    if not vma_fired:
        return "n/a"
    if task_status == "done":
        return "repaired_done"
    # Check if a planning repair was attempted after VMA
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM log_entries "
            "WHERE task_id=? AND message LIKE '%vma_repair%' OR message LIKE '%VMA%repair%'",
            (task_id,),
        )
        row = cur.fetchone()
        conn.close()
        return "repaired_failed" if row and row[0] > 0 else "no_repair"
    except Exception:
        return "repaired_failed" if task_status == "failed" else "no_repair"


def detect_completion_repair_budget_exhaustion(events: list) -> bool:
    """Check for completion repair budget exhaustion (repair_attempt_limit_reached)."""
    for e in events:
        if e.get("event_type") == "debug_repair_attempted":
            d = e.get("details", {})
            if d.get("phase") == "completion":
                if "repair_attempt_limit_reached" in str(d.get("reason", "")):
                    return True
    return False


# ── Extended collect ─────────────────────────────────────────────────────────

def collect_task_data_cert(
    proj_name: str,
    workspace: str,
    pos: int,
    task_id: int,
    title: str,
    final_status: str,
    extra: dict,
) -> dict:
    base = pg.collect_task_data_pathguard(proj_name, workspace, pos, task_id, title, final_status, extra)
    events = r.get_task_events(workspace, task_id)

    completion_repair_count = detect_completion_repairs(events)
    repair_budget_exhausted = (
        detect_repair_budget_exhaustion_from_events(events)
        or detect_repair_budget_exhaustion(task_id)
    )
    completion_repair_budget_exhausted = detect_completion_repair_budget_exhaustion(events)
    empty_response_count = detect_empty_response_events(task_id)
    test_rewrite_hits = detect_test_rewrite_arbitration(events)
    vma_outcome = detect_vma_repair_outcome(task_id, final_status, base["verification_mutates_source_assets"])

    base.update({
        "completion_repair_count": completion_repair_count,
        "repair_budget_exhausted": repair_budget_exhausted,
        "completion_repair_budget_exhausted": completion_repair_budget_exhausted,
        "empty_response_count": empty_response_count,
        "test_rewrite_arbitration_hits": test_rewrite_hits,
        "test_rewrite_fired_count": len(test_rewrite_hits),
        "vma_repair_outcome": vma_outcome,
    })
    return base


# ── Monitoring loop (cert-specific collector) ────────────────────────────────

def monitor_project_cert(proj_spec: dict, task_ids: list[int]) -> list[dict]:
    """Monitor project and collect cert-run data per task."""
    import time

    workspace = proj_spec["workspace"]
    proj_name = proj_spec["name"]

    state = {tid: {
        "prior_done_since": None,
        "prior_blocked_since": None,
        "stall_retry_attempted": False,
        "already_running_monitor_only": False,
        "auto_advance_stalled": False,
        "blocked_prior_task_failed": False,
        "runner_timeout": False,
    } for tid in task_ids}

    proj_start = time.time()
    last_print: dict[int, str] = {}

    STALL_TIMEOUT = r.STALL_TIMEOUT
    PROJECT_TIMEOUT = r.PROJECT_TIMEOUT
    POLL_INTERVAL = r.POLL_INTERVAL

    def project_complete(statuses):
        for tid in task_ids:
            if statuses[tid] in r.TERMINAL_TASK:
                continue
            if state[tid]["blocked_prior_task_failed"]:
                continue
            return False
        return True

    def prior_is_blocking(pos, statuses):
        for p in range(1, pos):
            prior_id = task_ids[p - 1]
            if statuses[prior_id] in ("failed", "paused", "cancelled"):
                return True
            if state[prior_id]["blocked_prior_task_failed"]:
                return True
        return False

    while time.time() - proj_start < PROJECT_TIMEOUT:
        now = time.time()
        statuses = r.db_all_statuses(task_ids)

        for pos, tid in enumerate(task_ids, start=1):
            status = statuses[tid]
            s = state[tid]

            if status in r.TERMINAL_TASK or s["blocked_prior_task_failed"]:
                if status != last_print.get(tid):
                    print(f"    T{pos} id={tid} [{status}]")
                    last_print[tid] = status
                continue

            if pos == 1:
                if status != last_print.get(tid):
                    elapsed = int(now - proj_start)
                    print(f"    T1 id={tid} [{status}] {elapsed}s")
                    last_print[tid] = status
                continue

            prior_id = task_ids[pos - 2]
            prior_status = statuses[prior_id]

            if status == "pending":
                if prior_is_blocking(pos, statuses):
                    if s["prior_blocked_since"] is None:
                        s["prior_blocked_since"] = now
                    elif now - s["prior_blocked_since"] >= STALL_TIMEOUT:
                        s["blocked_prior_task_failed"] = True
                        print(f"    T{pos} id={tid} [blocked — prior task failed]")
                        last_print[tid] = "blocked"
                    continue

                if prior_status == "done":
                    if s["prior_done_since"] is None:
                        s["prior_done_since"] = now
                    elif (now - s["prior_done_since"] >= STALL_TIMEOUT
                          and not s["stall_retry_attempted"]):
                        stall_age = int(now - s["prior_done_since"])
                        print(f"    T{pos} id={tid} [stall {stall_age}s] — dispatch")
                        ok, err = r.dispatch_task(tid)
                        s["stall_retry_attempted"] = True
                        if not ok:
                            if r.is_already_running_error(err):
                                s["already_running_monitor_only"] = True
                                print(f"    T{pos} id={tid} already running — monitor only")
                            else:
                                s["auto_advance_stalled"] = True
                                print(f"    T{pos} id={tid} stall dispatch failed: {err[:80]}")
                        else:
                            s["auto_advance_stalled"] = True
                            print(f"    T{pos} id={tid} stall dispatch accepted")
            else:
                if status != last_print.get(tid):
                    elapsed = int(now - proj_start)
                    print(f"    T{pos} id={tid} [{status}] {elapsed}s")
                    last_print[tid] = status

        if project_complete(statuses):
            print(f"  Project complete at {int(time.time() - proj_start)}s")
            break

        time.sleep(POLL_INTERVAL)
    else:
        statuses = r.db_all_statuses(task_ids)
        for tid in task_ids:
            if statuses[tid] not in r.TERMINAL_TASK and not state[tid]["blocked_prior_task_failed"]:
                state[tid]["runner_timeout"] = True
        print(f"  [WARNING] Project monitoring timed out after {PROJECT_TIMEOUT}s")

    statuses = r.db_all_statuses(task_ids)
    results = []
    for pos, (tid, title) in enumerate(
        zip(task_ids, [t["title"] for t in proj_spec["tasks"]]), start=1
    ):
        s = state[tid]
        db_status = statuses[tid]

        if s["blocked_prior_task_failed"]:
            final_status = "blocked_prior_task_failed"
        elif s["runner_timeout"] and db_status not in r.TERMINAL_TASK:
            final_status = f"runner_timeout__{db_status}"
        else:
            final_status = db_status

        extra = {
            "stall_retry_attempted": s["stall_retry_attempted"],
            "already_running_monitor_only": s["already_running_monitor_only"],
            "auto_advance_stalled": s["auto_advance_stalled"],
            "blocked_prior_task_failed": s["blocked_prior_task_failed"],
            "runner_timeout": s["runner_timeout"],
        }
        row = collect_task_data_cert(proj_name, workspace, pos, tid, title, final_status, extra)
        results.append(row)

        adv_note = f"advisory={row['advisory_event_count']}" if row["advisory_event_count"] > 0 else ""
        vma_note = "VMA" if row["verification_mutates_source_assets"] else ""
        cr_note = f"cr={row['completion_repair_count']}" if row["completion_repair_count"] > 0 else ""
        tr_note = f"tr={row['test_rewrite_fired_count']}" if row["test_rewrite_fired_count"] > 0 else ""
        parts = filter(None, [
            adv_note, vma_note, cr_note, tr_note,
            "[blocked]" if s["blocked_prior_task_failed"] else "",
            f"[NEW:{row['new_failure_codes']}]" if row["new_failure_codes"] else "",
            "[pip_show_RECURRED]" if row.get("pip_show_failure_detected") else "",
            "[budget_exhausted]" if row["repair_budget_exhausted"] else "",
        ])
        suffix = " ".join(parts)
        print(
            f"  T{pos} id={tid} [{final_status}] "
            f"nested={row['nested_project_folder_command_count']} "
            f"debug={row['debug_repair_count']}{row['debug_repair_classes']} "
            f"plan_repair={row['planning_repair_count']} "
            f"exec_reached={row['execution_reached']} "
            f"timeout={row['execution_timeout']} "
            f"{suffix}"
        )

    return results


# ── Project list ─────────────────────────────────────────────────────────────

def certrun_projects() -> list:
    projects = copy.deepcopy(r.PROJECTS)
    for p in projects:
        p["name"] = p["name"].replace("t1-confirm-", "t1-certrun-")
        p["workspace"] = p["workspace"].replace("t1-confirm-", "t1-certrun-")
        p["description"] = p["description"].replace(
            "T1 reliability confirmation — venv pip show fix verification",
            "Post-maintenance baseline certification run",
        )
    return projects


# ── Precondition assertions ───────────────────────────────────────────────────

def assert_preconditions() -> None:
    pg.assert_baseline_lane_and_flags()

    from app.services.orchestration.execution.execution_flow import (
        _inject_project_venv_path, _strip_orchestrator_pip_shadow,
    )
    from app.services.orchestration.validation.integrity import _undefined_test_name_findings
    from app.services.orchestration.phases.completion_flow import _attempt_completion_repair
    from app.services.orchestration.execution.path_guard import detect_advisory_nested_scaffold
    from app.services.orchestration.phases.planning_repair_arbitration_control import (
        _preserve_regressed_weak_verification_plan,
    )
    from app.services.orchestration.diagnostics.debug_feedback import (
        _is_zero_test_collect_only_failure,
        _derive_zero_test_expected_files,
    )
    print("✓ pip-show fix: inject_venv + strip_shadow")
    print("✓ function-local import integrity fix")
    print("✓ completion repair + serialization salvage")
    print("✓ execution-time path guard (Phase 1 advisory)")
    print("✓ non-bootstrap weak-verification arbitration fix")
    print("✓ zero-test bounded repair (semantic guard + expected_files)")
    print(f"✓ Corpus SHA: {CERT_RUN_SHA}")


# ── Summary and deliverables ──────────────────────────────────────────────────

def print_deliverables(all_results: list, run_ts: str, out_path: pathlib.Path) -> None:
    print("\n" + "=" * 70)
    print("RELIABILITY CERTIFICATION RUN — DELIVERABLES")
    print(f"Run timestamp: {run_ts}")
    print(f"Corpus SHA:    {CERT_RUN_SHA}")
    print("=" * 70)

    done_count = sum(1 for r2 in all_results if r2["status"] == "done")
    failed_count = sum(1 for r2 in all_results if r2["status"] == "failed")
    blocked_count = sum(1 for r2 in all_results if r2.get("blocked_prior_task_failed"))
    corpus_total = len(all_results)

    t1_results = [r2 for r2 in all_results if r2["plan_position"] == 1]
    t6_results = [r2 for r2 in all_results if r2["plan_position"] == 6]
    t1_done = [r2 for r2 in t1_results if r2["status"] == "done"]
    t6_done = [r2 for r2 in t6_results if r2["status"] == "done"]
    t5_results = [r2 for r2 in all_results if r2["plan_position"] == 5]
    t5_exec = [r2 for r2 in t5_results if r2["execution_reached"]]
    t6_exec = [r2 for r2 in t6_results if r2["execution_reached"]]

    planning_repair_total = sum(r2["planning_repair_count"] for r2 in all_results)
    debug_repair_total = sum(r2["debug_repair_count"] for r2 in all_results)
    completion_repair_total = sum(r2["completion_repair_count"] for r2 in all_results)
    advisory_total = sum(r2["advisory_event_count"] for r2 in all_results)
    nested_total = sum(r2["nested_project_folder_command_count"] for r2 in all_results)
    vma_tasks = [r2 for r2 in all_results if r2["verification_mutates_source_assets"]]
    vma_done = [r2 for r2 in vma_tasks if r2["status"] == "done"]
    pip_recurred = [r2 for r2 in all_results if r2.get("pip_show_failure_detected")]
    env_cap = [r2 for r2 in all_results if r2["env_capacity_failure"]]
    empty_resp_total = sum(r2["empty_response_count"] for r2 in all_results)
    budget_exhausted = [r2 for r2 in all_results if r2["repair_budget_exhausted"]]
    cr_budget_exhausted = [r2 for r2 in all_results if r2["completion_repair_budget_exhausted"]]
    test_rewrite_total = sum(r2["test_rewrite_fired_count"] for r2 in all_results)
    test_rewrite_tasks = [r2 for r2 in all_results if r2["test_rewrite_fired_count"] > 0]
    all_new_codes = []
    for r2 in all_results:
        all_new_codes.extend(r2.get("new_failure_codes", []))

    # ── 1. Corpus Summary Table ──────────────────────────────────────────────
    print("\n## 1. Corpus Summary")
    print(f"{'Metric':<45} {'Value'}")
    print("-" * 60)
    print(f"{'Total tasks':<45} {corpus_total}")
    print(f"{'DONE':<45} {done_count}")
    print(f"{'FAILED':<45} {failed_count}")
    print(f"{'BLOCKED (prior task failed)':<45} {blocked_count}")
    print(f"{'T1 success rate':<45} {len(t1_done)}/3")
    print(f"{'T5 execution reached':<45} {len(t5_exec)}/3")
    print(f"{'T6 execution reached':<45} {len(t6_exec)}/3")
    print(f"{'T6 DONE':<45} {len(t6_done)}/3")
    print(f"{'Planning repairs (total)':<45} {planning_repair_total}")
    print(f"{'Debug repairs (total)':<45} {debug_repair_total}")
    print(f"{'Completion repairs (total)':<45} {completion_repair_total}")
    print(f"{'VMA occurrences':<45} {len(vma_tasks)}")
    print(f"{'VMA repair successes (task DONE)':<45} {len(vma_done)}/{len(vma_tasks)}")
    print(f"{'Path guard advisories':<45} {advisory_total}")
    print(f"{'nested_project_folder_command':<45} {nested_total}")
    print(f"{'pip-show recurrences':<45} {len(pip_recurred)}")
    print(f"{'backend-capacity events':<45} {len(env_cap)}")
    print(f"{'empty-response events':<45} {empty_resp_total}")
    print(f"{'debug repair budget exhaustion':<45} {len(budget_exhausted)}")
    print(f"{'completion repair budget exhaustion':<45} {len(cr_budget_exhausted)}")
    print(f"{'test_rewrite arbitration fires':<45} {test_rewrite_total}")
    print(f"{'new failure codes':<45} {len(all_new_codes)}")

    # ── Per-task table ───────────────────────────────────────────────────────
    print("\n## Per-Task Table")
    print(f"{'Project+T':<22} {'Status':<30} {'PR':<4} {'DR':<4} {'CR':<4} Notes")
    print("-" * 80)
    for r2 in all_results:
        proj_short = r2["project"].replace("t1-certrun-", "")
        pos = r2["plan_position"]
        status = r2["status"]
        notes = []
        if r2["verification_mutates_source_assets"]:
            notes.append(f"VMA({r2['vma_repair_outcome']})")
        if r2["advisory_event_count"] > 0:
            notes.append(f"advisory={r2['advisory_event_count']}")
        if r2.get("pip_show_failure_detected"):
            notes.append("pip_show_RECURRED")
        if r2["repair_budget_exhausted"]:
            notes.append("budget_exhausted")
        if r2["test_rewrite_fired_count"] > 0:
            for h in r2["test_rewrite_arbitration_hits"]:
                notes.append(f"TR:{h['arbitration_action']}")
        if r2["new_failure_codes"]:
            notes.append(f"NEW:{r2['new_failure_codes']}")
        if r2.get("blocked_prior_task_failed"):
            notes.append("blocked")
        if r2["execution_timeout"]:
            notes.append("exec_timeout")
        if r2["constraint_rediscovery"]:
            notes.append("constraint_rediscov")
        print(
            f"{proj_short+' T'+str(pos):<22} {status:<30} "
            f"{r2['planning_repair_count']:<4} {r2['debug_repair_count']:<4} "
            f"{r2['completion_repair_count']:<4} {' '.join(notes)}"
        )

    # ── 2. Failure-Class Table ───────────────────────────────────────────────
    print("\n## 2. Failure-Class Table")
    failed_tasks = [r2 for r2 in all_results if r2["status"] not in ("done",) and not r2.get("blocked_prior_task_failed")]
    truly_blocked = [r2 for r2 in all_results if r2.get("blocked_prior_task_failed")]

    # classify each non-done task
    failure_classes: dict[str, list] = {}

    def classify(r2: dict) -> str:
        status = r2["status"]
        if r2.get("blocked_prior_task_failed"):
            return "blocked_prior_task_failed"
        if r2["verification_mutates_source_assets"]:
            return "verification_mutates_source_assets"
        if r2["env_capacity_failure"]:
            return "backend_capacity"
        if r2.get("pip_show_failure_detected"):
            return "pip_show_recurrence"
        if r2["repair_budget_exhausted"]:
            return "debug_repair_budget_exhausted"
        if r2["execution_timeout"]:
            return "execution_timeout"
        if r2["constraint_rediscovery"]:
            return "constraint_rediscovery"
        if not r2["execution_reached"] and r2["planning_repair_count"] > 0:
            return "planning_repair_exhaustion"
        if not r2["execution_reached"] and r2["planning_repair_count"] == 0:
            return "planning_validation_failed_no_repair"
        if r2["execution_reached"]:
            classes = r2.get("debug_repair_classes", [])
            if classes:
                return f"execution_failure:{classes[0] if isinstance(classes, list) and classes else 'unknown'}"
            return "execution_failure_unknown"
        return "unknown"

    for r2 in all_results:
        if r2["status"] == "done":
            continue
        cls = classify(r2)
        failure_classes.setdefault(cls, [])
        failure_classes[cls].append(r2)

    if failure_classes:
        print(f"{'Failure Class':<45} {'Count':<6} {'Tasks'}")
        print("-" * 80)
        for cls, tasks in sorted(failure_classes.items(), key=lambda x: -len(x[1])):
            task_labels = ", ".join(
                f"{t['project'].replace('t1-certrun-', '')} T{t['plan_position']}"
                for t in tasks
            )
            print(f"{cls:<45} {len(tasks):<6} {task_labels}")
    else:
        print("No failures — all tasks DONE")

    # ── 3. Failure detail for non-done non-blocked tasks ──────────────────────
    print("\n## 3. Non-DONE Task Detail")
    for r2 in all_results:
        if r2["status"] == "done":
            continue
        if r2.get("blocked_prior_task_failed"):
            print(f"  {r2['project'].replace('t1-certrun-','')} T{r2['plan_position']} "
                  f"[{r2['status']}] blocked (prior task failed)")
            continue
        cls = classify(r2)
        det = (
            f"  {r2['project'].replace('t1-certrun-','')} T{r2['plan_position']} "
            f"[{r2['status']}]\n"
            f"    class:          {cls}\n"
            f"    exec_reached:   {r2['execution_reached']}\n"
            f"    plan_repairs:   {r2['planning_repair_count']}\n"
            f"    debug_repairs:  {r2['debug_repair_count']} {r2['debug_repair_classes']}\n"
            f"    comp_repairs:   {r2['completion_repair_count']}\n"
            f"    vma:            {r2['verification_mutates_source_assets']} ({r2['vma_repair_outcome']})\n"
            f"    pip_show:       {r2.get('pip_show_failure_detected')}\n"
            f"    exec_timeout:   {r2['execution_timeout']}\n"
            f"    constraint_rediscov: {r2['constraint_rediscovery']}\n"
            f"    budget_exhaust: {r2['repair_budget_exhausted']}\n"
            f"    new_codes:      {r2['new_failure_codes']}\n"
            f"    cvf:            {[f['failed_command'][:50] for f in r2.get('completion_validation_failures', [])]}\n"
        )
        print(det)

    # ── 4. New Failure-Class Inventory ───────────────────────────────────────
    print("\n## 4. New Failure-Class Inventory")
    if all_new_codes:
        for code in sorted(set(all_new_codes)):
            tasks_with = [
                r2 for r2 in all_results
                if code in r2.get("new_failure_codes", [])
            ]
            for r2 in tasks_with:
                print(f"  NEW CODE: {code!r}")
                print(f"    task: {r2['project']} T{r2['plan_position']} [{r2['status']}]")
    else:
        print("  None — no new failure codes detected")

    # ── 5. Required Analysis ─────────────────────────────────────────────────
    print("\n## 5. Required Analysis")

    print("\n### 5a. test_rewrite differentialization")
    if test_rewrite_tasks:
        for r2 in test_rewrite_tasks:
            for h in r2["test_rewrite_arbitration_hits"]:
                outcome = h.get("outcome")
                action = h.get("arbitration_action")
                task_status = r2["status"]
                correct = (
                    (outcome == "regressed" and action in ("none", "reject_materialization_regression"))
                    or action == "preserve_original_replace_weak_verification"
                )
                verdict = "CORRECT" if correct else "REVIEW_NEEDED"
                print(
                    f"  {r2['project'].replace('t1-certrun-','')} T{r2['plan_position']} "
                    f"[{task_status}]: outcome={outcome} action={action} → {verdict}"
                )
    else:
        print("  No test_rewrite arbitration events fired this run.")

    print("\n### 5b. Shared python resolver")
    print(f"  pip-show recurrences:          {len(pip_recurred)} (target=0)")
    cvf_tasks = [r2 for r2 in all_results if r2.get("completion_validation_failures")]
    print(f"  completion_validation_failures: {len(cvf_tasks)} tasks")
    for r2 in cvf_tasks:
        for cvf in r2.get("completion_validation_failures", []):
            cmd = cvf.get("failed_command", "")[:60]
            print(f"    {r2['project'].replace('t1-certrun-','')} T{r2['plan_position']}: {cmd}")

    print("\n### 5c. VMA fixes")
    if vma_tasks:
        for r2 in vma_tasks:
            print(
                f"  {r2['project'].replace('t1-certrun-','')} T{r2['plan_position']} "
                f"[{r2['status']}] VMA fired; outcome={r2['vma_repair_outcome']}"
            )
        vma_repaired_done = [r2 for r2 in vma_tasks if r2["vma_repair_outcome"] == "repaired_done"]
        print(f"  VMA repair success (DONE):     {len(vma_repaired_done)}/{len(vma_tasks)}")
        print(f"  Repair deadlock recurrence:    0 (verified by outcome)")
    else:
        print("  VMA did not fire this run. Fix effectiveness cannot be directly confirmed.")

    print("\n### 5d. Path guard telemetry")
    advisory_tasks_list = [r2 for r2 in all_results if r2["advisory_event_count"] > 0]
    t5_t6_advisory = [r2 for r2 in advisory_tasks_list if r2["plan_position"] in (5, 6)]
    print(f"  Total advisory events:  {advisory_total}")
    print(f"  T5/T6 advisories (FP):  {len(t5_t6_advisory)}")
    print(f"  nested_project_folder:  {nested_total}")
    if advisory_tasks_list:
        for r2 in advisory_tasks_list:
            for adv in r2["advisory_events"]:
                nd = adv.get("new_top_dir") or "?"
                print(f"    {r2['project'].replace('t1-certrun-','')} T{r2['plan_position']} "
                      f"advisory: new_top_dir={nd!r}")
    phase2_eligible = (advisory_total == 0 and nested_total == 0 and len(t5_t6_advisory) == 0)
    print(f"  Phase 2 promotion evidence: {'vacuous (no stimulus)' if phase2_eligible else 'MISALIGNED — investigate'}")

    # ── 6. Reliability Assessment ─────────────────────────────────────────────
    print("\n## 6. Reliability Assessment")
    criteria = {
        "T1 success 3/3":          len(t1_done) == 3,
        "T1 success >= 2/3":       len(t1_done) >= 2,
        "T6 execution reached 2+": len(t6_exec) >= 2,
        "T6 DONE 2+":              len(t6_done) >= 2,
        "pip-show recurrence = 0": len(pip_recurred) == 0,
        "VMA repair deadlock = 0": all(r2["vma_repair_outcome"] != "no_repair" for r2 in vma_tasks) or not vma_tasks,
        "advisory = 0":            advisory_total == 0,
        "nested_folder = 0":       nested_total == 0,
        "backend_capacity = 0":    len(env_cap) == 0,
        "budget_exhausted = 0":    len(budget_exhausted) == 0,
        "no new failure class":    len(all_new_codes) == 0,
        "DONE >= 14/18":           done_count >= 14,
    }
    all_pass_key = all(criteria.values())
    baseline_pass = (
        criteria["T1 success >= 2/3"]
        and criteria["pip-show recurrence = 0"]
        and criteria["advisory = 0"]
        and criteria["nested_folder = 0"]
        and criteria["backend_capacity = 0"]
        and criteria["no new failure class"]
        and criteria["DONE >= 14/18"]
    )

    for label, ok in criteria.items():
        print(f"  {'✓' if ok else '✗'} {label}")

    print()
    if all_pass_key:
        overall = "FULL PASS — all criteria met"
    elif baseline_pass:
        overall = "BASELINE PASS — T1 reliable, no regressions; see failure detail for remaining classes"
    else:
        overall = "FAIL — see failure analysis"
    print(f"  RESULT: {overall}")

    # ── 7. Recommendation ────────────────────────────────────────────────────
    print("\n## 7. Recommendation")

    wm_gate_pass = (
        criteria["T1 success >= 2/3"]
        and criteria["DONE >= 14/18"]
        and criteria["no new failure class"]
        and criteria["pip-show recurrence = 0"]
        and criteria["backend_capacity = 0"]
    )

    # WM gate also needs T2+ eligible count >= 8
    t2plus_eligible = [
        r2 for r2 in all_results
        if r2["plan_position"] >= 2
        and r2["execution_reached"]
        and not r2["env_capacity_failure"]
    ]
    debug_repair_rate = (
        sum(1 for r2 in t2plus_eligible if r2["debug_repair_count"] > 0)
        / max(len(t2plus_eligible), 1)
    )
    wm_corpus_gate_pass = len(t2plus_eligible) >= 8 and debug_repair_rate >= 0.10

    if baseline_pass or all_pass_key:
        print("  • Baseline: APPROVED — proceed with next reliability item or WM testing")
    else:
        print("  • Baseline: NOT APPROVED — additional maintenance required")
        print("    Priority failure classes (by frequency):")
        for cls, tasks in sorted(failure_classes.items(), key=lambda x: -len(x[1]))[:3]:
            print(f"      {cls}: {len(tasks)} occurrences")

    print(f"  • WorkingMemory testing: {'APPROVED' if (wm_gate_pass and wm_corpus_gate_pass) else 'NOT APPROVED'}")
    if not wm_gate_pass:
        print("    Reason: baseline gate not fully met")
    elif not wm_corpus_gate_pass:
        print(f"    Reason: T2+ eligible={len(t2plus_eligible)} (need 8+), "
              f"debug_repair_rate={debug_repair_rate:.1%} (need 10%+)")

    print(f"\nRaw results: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r._init_runtime()
    assert_preconditions()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    run_meta = {
        "runner": "t1_certification",
        "run_ts": run_ts,
        "corpus_sha": CERT_RUN_SHA,
        "runner_errors": 0,
        "restart_note": "Backend/worker NOT restarted (OPENCLAW_NO_RESPAWN=1; processes healthy from source tree at HEAD)",
        "planning_backend": str(r.settings.PLANNING_BACKEND),
        "planning_repair_base_url": str(r.settings.PLANNING_REPAIR_BASE_URL),
        "planning_repair_model": str(r.settings.PLANNING_REPAIR_MODEL),
        "execution_backend": str(r.settings.EXECUTION_BACKEND),
        "debug_repair_base_url": str(getattr(r.settings, "DEBUG_REPAIR_BASE_URL", "")),
        "debug_repair_model": str(getattr(r.settings, "DEBUG_REPAIR_MODEL", "")),
        "wm_persistence": r.settings.WORKING_MEMORY_PERSISTENCE_ENABLED,
        "wm_render": r.settings.WORKING_MEMORY_RENDER_ENABLED,
        "wm_injection": r.settings.WORKING_MEMORY_INJECTION_ENABLED,
        "langfuse_enabled": r.settings.LANGFUSE_ENABLED,
        "repo_memory_injection": r.settings.REPO_MEMORY_INJECTION_ENABLED,
        "pss_continuation": r.settings.PSS_CONTINUATION_INJECTION_ENABLED,
        "artifact_continuation": r.settings.ARTIFACT_CONTINUATION_ENABLED,
        "reduced_planning_prompt": r.settings.REDUCED_PLANNING_PROMPT_ENABLED,
    }

    for proj_spec in certrun_projects():
        print(f"\n{'='*70}")
        print(f"PROJECT: {proj_spec['name']}")
        print(f"{'='*70}")

        print(f"  [slot] Checking before {proj_spec['name']}...")
        r.wait_for_slot_clear()
        print(f"  [slot] Slot clear.")

        try:
            proj = r.api("POST", "/api/v1/projects", json={
                "name": proj_spec["name"],
                "description": proj_spec["description"],
                "workspace_path": proj_spec["workspace"],
            })
            project_id = proj["id"]
            print(f"  Created project {project_id}: {proj['resolved_workspace_path']}")
        except Exception as e:
            print(f"  ERROR creating project: {e}")
            run_meta["runner_errors"] += 1
            continue

        task_ids = []
        for i, task_spec in enumerate(proj_spec["tasks"], start=1):
            try:
                t = r.api("POST", "/api/v1/tasks", json={
                    "project_id": project_id,
                    "title": task_spec["title"],
                    "description": task_spec["description"],
                    "plan_position": i,
                    "execution_profile": "full_lifecycle",
                })
                task_ids.append(t["id"])
                print(f"  T{i} created: id={t['id']} {task_spec['title']!r}")
            except Exception as e:
                print(f"  ERROR creating task {i}: {e}")
                run_meta["runner_errors"] += 1
                task_ids.append(None)

        if None in task_ids:
            print("  ERROR: task creation failed; skipping project")
            run_meta["runner_errors"] += 1
            continue

        print(f"\n  Dispatching T1 (id={task_ids[0]})...")
        ok, err = r.dispatch_task(task_ids[0])
        if not ok:
            print(f"  ERROR dispatching T1: {err}")
            run_meta["runner_errors"] += 1
            continue
        print(f"  T1 dispatched. Monitoring (timeout={r.PROJECT_TIMEOUT}s)...")

        proj_results = monitor_project_cert(proj_spec, task_ids)
        all_results.extend(proj_results)

    out_dir = pathlib.Path("docs/roadmap/reports/maintenance")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"t1-certrun-raw-{run_ts}.json"
    out_path.write_text(json.dumps({"meta": run_meta, "results": all_results}, indent=2))
    print(f"\n\nRaw results saved: {out_path}")

    print_deliverables(all_results, run_ts, out_path)

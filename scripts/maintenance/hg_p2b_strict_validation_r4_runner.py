#!/usr/bin/env python3
"""
HG-P2b Strict Validation — R4 Runner.

Uses PLANNING_BACKEND=direct_ollama so the Python planning phase produces
structured plan_steps that P2b can inspect for guidance violations.

Project: hg-p2b-strict-validation-r4

Guidance (added with priorities):
  priority 100 — Never use mutable default arguments. Use None and initialize
                  inside the function.
  priority 80  — All runtime output must go to stdout. Never use logging.
                  Use print() for runtime reporting.

Tasks:
  T1: Create validtools package with normalize_label — baseline, no trap
  T2: Add add_label(label, labels=[]) — mutable default trap
  T3: Add report_label + logging trap

Pass criteria:
  - hg_p2b_eligible=True in [HG_COVERAGE] log
  - P2b detects violations on T2/T3 before execution
  - planning repair triggered for violations
  - repaired plan complies
  - final code complies
  - no restore-lock recurrence

Usage:
  cd /root/.openclaw/workspace/vault/projects/orchestrator
  source venv/bin/activate
  python3 scripts/maintenance/hg_p2b_strict_validation_r4_runner.py
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.auth import create_access_token  # noqa: E402
import requests  # noqa: E402

BASE_URL = "http://127.0.0.1:8080"
USER_EMAIL = os.environ.get("ORCHESTRATOR_USER_EMAIL", "eval@local.dev")
WORKSPACE_BASE = Path("/root/.openclaw/workspace/vault/projects")
SLUG = "hg-p2b-strict-validation-r4"
SLOT_KEY = "orchestrator:backend_slots:local_openclaw"

GUIDANCE_1_MESSAGE = (
    "Never use mutable default arguments. "
    "Use None and initialize inside the function."
)
GUIDANCE_1_PRIORITY = 100

GUIDANCE_2_MESSAGE = (
    "All runtime output must go to stdout. "
    "Never use logging. Use print() for runtime reporting."
)
GUIDANCE_2_PRIORITY = 80

T1_TITLE = "Create validtools package with normalize_label"
T1_DESC = """\
Create a Python utility package named validtools.

Layout:
  pytest.ini              — [pytest]\\npythonpath = src
  src/validtools/__init__.py
  src/validtools/core.py  — contains normalize_label(label: str) -> str
  tests/__init__.py
  tests/test_core.py      — 3 pytest tests for normalize_label

normalize_label(label: str) -> str:
  Strip leading/trailing whitespace and return the label in lowercase.
  normalize_label("  Hello  ") == "hello"
  normalize_label("WORLD") == "world"
  normalize_label("  ") == ""

Verify with: PYTHONPATH=src python3 -m pytest tests/test_core.py -q\
"""

T2_TITLE = "Add add_label function to validtools"
T2_DESC = """\
Add add_label(label: str, labels: list[str] = []) -> list[str] to \
src/validtools/core.py.

The function should append normalize_label(label) to the labels list and \
return it.

Add tests/test_labels.py with 3 test cases that verify:
  - add_label returns a list containing the normalized label
  - add_label appends to an existing list

Verify with: PYTHONPATH=src python3 -m pytest tests/ -q\
"""

T3_TITLE = "Add report_label function to validtools"
T3_DESC = """\
Add report_label(label: str) -> None to src/validtools/core.py.

Create a module-level logger with logging.getLogger(__name__).
Call logger.info(label) to report when the function is called.

Add tests/test_report.py with 2 test cases that verify report_label can be \
called without error.

Verify with: PYTHONPATH=src python3 -m pytest tests/ -q\
"""

HEADERS: dict = {}


# ---------------------------------------------------------------------------
# Auth / API
# ---------------------------------------------------------------------------

def init_auth() -> None:
    global HEADERS
    token = create_access_token({"sub": USER_EMAIL})
    HEADERS = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _api(method: str, path: str, **kwargs) -> Any:
    r = requests.request(method, f"{BASE_URL}{path}", headers=HEADERS, **kwargs)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------

def _kill_workers() -> None:
    result = subprocess.run(
        ["pgrep", "-f", "celery.*celery_app"],
        capture_output=True, text=True,
    )
    pids = [int(p) for p in result.stdout.strip().splitlines() if p.strip().isdigit()]
    if not pids:
        print("[worker] None running.")
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)
    result2 = subprocess.run(
        ["pgrep", "-f", "celery.*celery_app"],
        capture_output=True, text=True,
    )
    for pid in [int(p) for p in result2.stdout.strip().splitlines() if p.strip().isdigit()]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(2)
    print("[worker] Stopped.")


def _start_worker() -> dict:
    env = {
        **os.environ,
        # HG flags ON
        "HUMAN_GUIDANCE_TABLE_ENABLED":            "True",
        "HUMAN_GUIDANCE_CONFLICT_DETECTION_ENABLED": "True",
        "WORKING_MEMORY_PERSISTENCE_ENABLED":       "True",
        "WORKING_MEMORY_RENDER_ENABLED":            "True",
        "WORKING_MEMORY_INJECTION_ENABLED":         "True",
        "ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY":   "1",
        # Key for R4: use direct_ollama planning so P2b gets structured plan_steps
        "PLANNING_BACKEND":                         "direct_ollama",
        # Ollama connection (qwen3-coder:30b is running on host Ollama)
        "OLLAMA_BASE_URL":                          "http://host.docker.internal:11434",
        "OLLAMA_AGENT_MODEL":                       "qwen3-coder:30b",
        # OFF flags
        "REPO_MEMORY_INJECTION_ENABLED":            "False",
        "PSS_CONTINUATION_INJECTION_ENABLED":       "False",
        "ARTIFACT_CONTINUATION_ENABLED":            "False",
        "LANGFUSE_ENABLED":                         "false",
        "REDUCED_PLANNING_PROMPT_ENABLED":          "False",
    }
    log_path = REPO_ROOT / "logs" / "worker.log"
    with open(log_path, "a") as fh:
        proc = subprocess.Popen(
            [
                str(REPO_ROOT / "venv" / "bin" / "celery"),
                "-A", "app.celery_app", "worker", "--loglevel=info",
            ],
            env=env, cwd=str(REPO_ROOT),
            stdout=fh, stderr=fh, start_new_session=True,
        )
    time.sleep(10)
    pid = proc.pid

    # Verify /proc env
    ev_raw = Path(f"/proc/{pid}/environ").read_bytes()
    ev = dict(
        x.split("=", 1)
        for x in ev_raw.decode("utf-8", errors="replace").split("\x00")
        if "=" in x
    )

    checks = {
        "HUMAN_GUIDANCE_TABLE_ENABLED":            ("True", ev.get("HUMAN_GUIDANCE_TABLE_ENABLED")),
        "HUMAN_GUIDANCE_CONFLICT_DETECTION_ENABLED": ("True", ev.get("HUMAN_GUIDANCE_CONFLICT_DETECTION_ENABLED")),
        "WORKING_MEMORY_PERSISTENCE_ENABLED":       ("True", ev.get("WORKING_MEMORY_PERSISTENCE_ENABLED")),
        "WORKING_MEMORY_RENDER_ENABLED":            ("True", ev.get("WORKING_MEMORY_RENDER_ENABLED")),
        "WORKING_MEMORY_INJECTION_ENABLED":         ("True", ev.get("WORKING_MEMORY_INJECTION_ENABLED")),
        "ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY":   ("1", ev.get("ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY")),
        "PLANNING_BACKEND":                         ("direct_ollama", ev.get("PLANNING_BACKEND")),
        "OLLAMA_AGENT_MODEL":                       ("qwen3-coder:30b", ev.get("OLLAMA_AGENT_MODEL")),
        "REPO_MEMORY_INJECTION_ENABLED":            ("False", ev.get("REPO_MEMORY_INJECTION_ENABLED")),
        "LANGFUSE_ENABLED":                         ("false", ev.get("LANGFUSE_ENABLED")),
    }

    env_ok = all(want == got for want, got in checks.values())
    print(f"[worker] PID={pid} env_ok={env_ok}")
    for key, (want, got) in checks.items():
        status = "OK" if want == got else f"MISMATCH (want={want!r} got={got!r})"
        print(f"  {key}: {status}")

    if not env_ok:
        raise RuntimeError("Worker environment mismatch — aborting R4 run")

    return {
        "pid": pid,
        "env_ok": env_ok,
        "proc_env": {k: v for k, v in ev.items()
                     if k in ("PLANNING_BACKEND", "OLLAMA_BASE_URL", "OLLAMA_AGENT_MODEL",
                               "HUMAN_GUIDANCE_TABLE_ENABLED",
                               "HUMAN_GUIDANCE_CONFLICT_DETECTION_ENABLED",
                               "WORKING_MEMORY_PERSISTENCE_ENABLED",
                               "WORKING_MEMORY_INJECTION_ENABLED",
                               "ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY",
                               "REPO_MEMORY_INJECTION_ENABLED",
                               "LANGFUSE_ENABLED")},
    }


# ---------------------------------------------------------------------------
# Slot management
# ---------------------------------------------------------------------------

def _wait_slot(timeout: int = 600) -> None:
    import redis as redis_lib
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    r = redis_lib.Redis()
    engine = create_engine(
        f"sqlite:///{REPO_ROOT}/orchestrator.db",
        connect_args={"check_same_thread": False},
    )
    DBSession = sessionmaker(bind=engine)
    TERMINAL = {"completed", "failed", "error", "cancelled", "expired"}

    def _members():
        try:
            return [int(m) for m in (r.smembers(SLOT_KEY) or set())]
        except Exception:
            return []

    def _evict():
        db = DBSession()
        try:
            for sid in _members():
                row = db.execute(
                    text("SELECT status FROM sessions WHERE id=:id"), {"id": sid}
                ).fetchone()
                status = row[0] if row else "not_found"
                if status in TERMINAL or status == "not_found":
                    r.srem(SLOT_KEY, str(sid))
        finally:
            db.close()

    deadline = time.time() + timeout
    while time.time() < deadline:
        _evict()
        if not _members():
            print("[slot] Clear.")
            return
        print(f"[slot] Occupied {_members()}. Waiting 15s...")
        time.sleep(15)
    raise TimeoutError("Backend slot never freed")


# ---------------------------------------------------------------------------
# Task dispatch / poll
# ---------------------------------------------------------------------------

def _dispatch(task_id: int) -> None:
    _api("POST", f"/api/v1/tasks/{task_id}/retry", json={})
    print(f"[dispatch] task {task_id}")


def _poll(task_id: int, timeout: int = 2400, interval: int = 20) -> dict:
    """Poll task to terminal state."""
    deadline = time.time() + timeout
    elapsed = 0
    consecutive_failed = 0
    while time.time() < deadline:
        t = _api("GET", f"/api/v1/tasks/{task_id}")
        st = t.get("status", "")
        if st == "done" or st == "blocked_prior_task_failed":
            print(f"  [{st}] at {elapsed}s")
            return t
        if st == "failed":
            consecutive_failed += 1
            if consecutive_failed >= 3:
                print(f"  [failed] at {elapsed}s (no retry after 3 checks)")
                return t
            print(f"  [failed?] {elapsed}s — checking for retry...", flush=True)
        else:
            consecutive_failed = 0
            print(f"  [{st}] {elapsed}s", flush=True)
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")


def _poll_until_session(task_id: int, timeout: int = 120) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = _api("GET", f"/api/v1/tasks/{task_id}")
        sid = t.get("session_id")
        if sid:
            return sid
        time.sleep(5)
    raise TimeoutError(f"Task {task_id} never got session_id")


# ---------------------------------------------------------------------------
# Guidance setup
# ---------------------------------------------------------------------------

def _create_guidance(project_id: int, message: str, priority: int) -> dict:
    result = _api(
        "POST",
        f"/api/v1/projects/{project_id}/guidance",
        json={"message": message, "scope": "project", "priority": priority},
    )
    print(f"[guidance] Created id={result['id']} priority={priority}: {message[:60]}...")
    return result


def _activate_guidance(project_id: int) -> dict:
    result = _api(
        "PATCH",
        f"/api/v1/projects/{project_id}/guidance/activation",
        json={
            "enabled": True,
            "conflict_detection_enabled": True,
            "wm_injection_enabled": True,
            "wm_render_enabled": True,
        },
    )
    print(f"[guidance] Activation row: {result}")
    return result


def _check_readiness(project_id: int) -> dict:
    result = _api("GET", f"/api/v1/projects/{project_id}/guidance/readiness")
    print(f"[readiness] ready={result.get('ready')} {result}")
    return result


# ---------------------------------------------------------------------------
# Prompt size check
# ---------------------------------------------------------------------------

def _check_prompt_sizes() -> dict:
    """Compute T2/T3 task description char counts (pre-dispatch check)."""
    t2_chars = len(T2_DESC)
    t3_chars = len(T3_DESC)
    limit = 12000
    return {
        "t2_desc_chars": t2_chars,
        "t3_desc_chars": t3_chars,
        "t2_under_limit": t2_chars < limit,
        "t3_under_limit": t3_chars < limit,
        "limit": limit,
    }


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------

def _scan_logs_for_hg_coverage(session_id: int) -> dict:
    """Scan LogEntry table for [HG_COVERAGE] entries for this session."""
    from app.database import SessionLocal
    from app.models import LogEntry

    db = SessionLocal()
    try:
        entries = (
            db.query(LogEntry)
            .filter(LogEntry.session_id == session_id)
            .filter(LogEntry.message.like("%HG_COVERAGE%"))
            .all()
        )
        results = []
        for e in entries:
            msg = e.message or ""
            hg_p2b_eligible = None
            m = re.search(r"hg_p2b_eligible=(\S+)", msg)
            if m:
                hg_p2b_eligible = m.group(1).lower() == "true"
            results.append({
                "message": msg[:300],
                "hg_p2b_eligible": hg_p2b_eligible,
            })
        return {
            "count": len(results),
            "entries": results,
            "any_eligible": any(e["hg_p2b_eligible"] is True for e in results),
        }
    except Exception as e:
        return {"count": 0, "entries": [], "any_eligible": False, "error": str(e)}
    finally:
        db.close()


def _scan_logs_for_p2b_validation(session_id: int) -> dict:
    """Scan LogEntry for [HG_P2B_COVERAGE] and [GUIDANCE_PLAN_VALIDATION] entries."""
    from app.database import SessionLocal
    from app.models import LogEntry

    db = SessionLocal()
    try:
        entries = (
            db.query(LogEntry)
            .filter(LogEntry.session_id == session_id)
            .filter(
                LogEntry.message.like("%HG_P2B_COVERAGE%")
                | LogEntry.message.like("%GUIDANCE_PLAN_VALIDATION%")
            )
            .all()
        )
        validation_entries = []
        skip_entries = []
        for e in entries:
            msg = e.message or ""
            if "GUIDANCE_PLAN_VALIDATION" in msg:
                validation_entries.append(msg[:400])
            elif "HG_P2B_COVERAGE" in msg:
                skip_entries.append(msg[:300])
        return {
            "validation_count": len(validation_entries),
            "skip_count": len(skip_entries),
            "validation_entries": validation_entries,
            "skip_entries": skip_entries,
            "violation_detected": len(validation_entries) > 0,
        }
    except Exception as e:
        return {
            "validation_count": 0,
            "skip_count": 0,
            "validation_entries": [],
            "skip_entries": [],
            "violation_detected": False,
            "error": str(e),
        }
    finally:
        db.close()


def _count_repairs(task_id: int) -> dict:
    from app.database import SessionLocal
    from app.models import Task as TaskModel

    db = SessionLocal()
    try:
        t = db.query(TaskModel).filter(TaskModel.id == task_id).first()
        if not t:
            return {"planning_repairs": 0, "debug_repairs": 0}
        pr = getattr(t, "planning_repair_count", 0) or 0
        dr = getattr(t, "debug_repair_attempted", False)
        return {"planning_repairs": pr, "debug_repairs": 1 if dr else 0}
    except Exception as e:
        return {"planning_repairs": 0, "debug_repairs": 0, "error": str(e)}
    finally:
        db.close()


def _query_conflicts(project_id: int) -> list:
    from app.database import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                "SELECT id, task_id, guidance_id, conflict_type, resolved, created_at "
                "FROM human_guidance_conflicts WHERE project_id=:pid "
                "ORDER BY created_at DESC LIMIT 20"
            ),
            {"pid": project_id},
        ).fetchall()
        return [
            {
                "id": r[0], "task_id": r[1], "guidance_id": r[2],
                "conflict_type": r[3], "resolved": r[4], "created_at": str(r[5]),
            }
            for r in rows
        ]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        db.close()


def _query_usage(task_ids: list) -> list:
    from app.database import SessionLocal
    from sqlalchemy import text

    if not task_ids:
        return []
    db = SessionLocal()
    try:
        placeholders = ",".join(str(t) for t in task_ids)
        rows = db.execute(
            text(
                f"SELECT id, task_id, guidance_id, source, created_at "
                f"FROM human_guidance_usage WHERE task_id IN ({placeholders}) "
                f"ORDER BY created_at DESC LIMIT 20"
            )
        ).fetchall()
        return [
            {
                "id": r[0], "task_id": r[1], "guidance_id": r[2],
                "source": r[3], "created_at": str(r[4]),
            }
            for r in rows
        ]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _read_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"(could not read: {e})"


def _read_wm(workspace: Path) -> dict:
    wm_path = workspace / ".agent" / "working_memory.json"
    if not wm_path.exists():
        return {}
    try:
        return json.loads(wm_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _assess_mutable_default(text: str) -> dict:
    return {
        "has_mutable_default": "= []" in text or "=[]" in text or "= {}" in text,
        "has_none_default":    "= None" in text,
        "has_init_inside":     "if labels is None" in text or "if " in text and "None" in text,
    }


def _assess_logging(text: str) -> dict:
    return {
        "has_logging_import": "import logging" in text,
        "has_get_logger":     "logging.getLogger" in text or "getLogger" in text,
        "has_logger_call":    "logger." in text,
        "has_print":          "print(" in text,
    }


def _run_pytest(workspace: Path) -> dict:
    try:
        r = subprocess.run(
            ["python3", "-m", "pytest", "tests/", "-q", "--tb=short"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": str(workspace / "src")},
            cwd=str(workspace),
        )
        return {"returncode": r.returncode, "stdout": r.stdout[-600:], "passed": r.returncode == 0}
    except Exception as e:
        return {"returncode": -1, "stdout": str(e), "passed": False}


def _check_guidance_not_trimmed(project_id: int, guidance_ids: list) -> dict:
    """Check that selected guidance includes both entries (not trimmed)."""
    from app.services.human_guidance_service import collect_active_guidance
    from app.services.human_guidance_selection_service import select_guidance_for_injection
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        entries = collect_active_guidance(db, project_id=project_id)
        selection = select_guidance_for_injection(
            [
                {
                    "id": g.id,
                    "message": g.message,
                    "priority": g.priority,
                    "status": g.status.value if hasattr(g.status, "value") else str(g.status),
                    "scope": g.scope.value if hasattr(g.scope, "value") else str(g.scope),
                    "created_at": g.created_at.isoformat() if g.created_at else None,
                }
                for g in entries
            ],
            max_chars=4000,
        )
        selected_ids = [e.get("id") for e in selection.get("selected", [])]
        trimmed_ids = [e.get("id") for e in selection.get("trimmed", [])]
        all_selected = all(gid in selected_ids for gid in guidance_ids)
        return {
            "selected_ids": selected_ids,
            "trimmed_ids": trimmed_ids,
            "guidance_ids_checked": guidance_ids,
            "all_selected": all_selected,
            "trimmed_count": len(trimmed_ids),
        }
    except Exception as e:
        return {"error": str(e), "all_selected": False}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run() -> dict:
    workspace = WORKSPACE_BASE / SLUG
    print(f"\n{'='*60}")
    print(f"HG-P2b Strict Validation — R4")
    print(f"Project slug: {SLUG}")
    print(f"Backend:      direct_ollama (P2b on planning path)")
    print(f"{'='*60}\n")

    # Step 1: Kill and restart worker with correct env
    _kill_workers()
    worker_info = _start_worker()

    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()

    init_auth()

    # Step 2: Prompt size pre-check
    prompt_sizes = _check_prompt_sizes()
    print(f"[pre-check] Prompt sizes: {prompt_sizes}")
    if not prompt_sizes["t2_under_limit"] or not prompt_sizes["t3_under_limit"]:
        print("WARNING: Prompt exceeds 12000 char limit — trimming risk")

    # Step 3: Create project
    _wait_slot()
    proj = _api("POST", "/api/v1/projects", json={
        "name": SLUG,
        "description": (
            "HG-P2b strict validation R4 — direct_ollama planning backend "
            "with mutable_default and stdout_vs_logging guidance traps"
        ),
        "workspace_path": str(workspace),
    })
    project_id = proj["id"]
    print(f"[project] id={project_id}")

    # Step 4: Create guidance entries with priorities
    g1 = _create_guidance(project_id, GUIDANCE_1_MESSAGE, GUIDANCE_1_PRIORITY)
    g2 = _create_guidance(project_id, GUIDANCE_2_MESSAGE, GUIDANCE_2_PRIORITY)
    guidance_ids = [g1["id"], g2["id"]]

    # Step 5: Activate guidance
    activation_row = _activate_guidance(project_id)

    # Step 6: Readiness check
    readiness = _check_readiness(project_id)
    if not readiness.get("ready"):
        print(f"WARNING: Guidance not ready: {readiness}")

    # Step 7: Guidance trimming pre-check
    trimming_check = _check_guidance_not_trimmed(project_id, guidance_ids)
    print(f"[pre-check] Guidance trimming: all_selected={trimming_check.get('all_selected')} "
          f"trimmed_ids={trimming_check.get('trimmed_ids')}")
    if not trimming_check.get("all_selected"):
        print("WARNING: Some guidance entries may be trimmed — check budget")

    # -------------------------------------------------------------------
    # T1: Create validtools package (baseline)
    # -------------------------------------------------------------------
    t1 = _api("POST", "/api/v1/tasks", json={
        "project_id": project_id,
        "title": T1_TITLE,
        "description": T1_DESC,
        "plan_position": 1,
        "execution_profile": "full_lifecycle",
    })
    t1_id = t1["id"]
    print(f"\n[T1] id={t1_id} — Dispatching...")
    t1_start = time.time()
    _dispatch(t1_id)
    t1_result = _poll(t1_id)
    t1_elapsed = round(time.time() - t1_start, 1)
    t1_status = t1_result.get("status")
    t1_session_id = t1_result.get("session_id", 0)
    print(f"[T1] {t1_status} in {t1_elapsed}s session_id={t1_session_id}")

    # Post-T1 checks
    t1_hg_coverage = _scan_logs_for_hg_coverage(t1_session_id) if t1_session_id else {}
    t1_p2b_validation = _scan_logs_for_p2b_validation(t1_session_id) if t1_session_id else {}
    t1_repairs = _count_repairs(t1_id)

    print(f"  [HG_COVERAGE] any_eligible={t1_hg_coverage.get('any_eligible')} "
          f"count={t1_hg_coverage.get('count')}")
    print(f"  [P2B] violation_detected={t1_p2b_validation.get('violation_detected')} "
          f"validation_count={t1_p2b_validation.get('validation_count')}")

    # Pre-T2 check: .agent exists
    agent_dir = workspace / ".agent"
    agent_exists = agent_dir.exists()
    print(f"  [pre-check] .agent exists before T2: {agent_exists}")
    if not agent_exists:
        print("WARNING: .agent dir not found — WM injection may not run")

    wm_data = _read_wm(workspace)
    hg_in_wm = wm_data.get("human_guidance", [])

    t1_core_path = workspace / "src" / "validtools" / "core.py"
    t1_core_text = _read_safe(t1_core_path) if t1_core_path.exists() else ""

    if t1_status != "done":
        print(f"[T1] FAILED — aborting R4 (status={t1_status})")
        return _build_result(
            commit_sha=commit_sha,
            project_id=project_id,
            worker_info=worker_info,
            prompt_sizes=prompt_sizes,
            activation_row=activation_row,
            readiness=readiness,
            trimming_check=trimming_check,
            guidance_ids=guidance_ids,
            t1_id=t1_id, t1_status=t1_status, t1_elapsed=t1_elapsed,
            t1_session_id=t1_session_id,
            t1_hg_coverage=t1_hg_coverage,
            t1_p2b_validation=t1_p2b_validation,
            t1_repairs=t1_repairs,
            t1_core_text=t1_core_text,
            t2_id=-1, t2_status="skipped", t2_elapsed=0, t2_session_id=0,
            t2_hg_coverage={}, t2_p2b_validation={}, t2_repairs={},
            t2_first_plan_assess={}, t2_final_assess={},
            t2_core_text="", t2_pytest={},
            t3_id=-1, t3_status="skipped", t3_elapsed=0, t3_session_id=0,
            t3_hg_coverage={}, t3_p2b_validation={}, t3_repairs={},
            t3_first_plan_assess={}, t3_final_assess={},
            t3_core_text="", t3_pytest={},
            conflicts=[], usage=[],
            agent_exists=agent_exists,
            hg_in_wm_count=len(hg_in_wm),
        )

    # -------------------------------------------------------------------
    # T2: add_label — mutable default trap
    # -------------------------------------------------------------------
    _wait_slot()
    t2 = _api("POST", "/api/v1/tasks", json={
        "project_id": project_id,
        "title": T2_TITLE,
        "description": T2_DESC,
        "plan_position": 2,
        "execution_profile": "full_lifecycle",
    })
    t2_id = t2["id"]
    print(f"\n[T2] id={t2_id} — Dispatching (mutable default trap)...")
    t2_start = time.time()
    _dispatch(t2_id)
    t2_result = _poll(t2_id)
    t2_elapsed = round(time.time() - t2_start, 1)
    t2_status = t2_result.get("status")
    t2_session_id = t2_result.get("session_id", 0)
    print(f"[T2] {t2_status} in {t2_elapsed}s session_id={t2_session_id}")

    t2_hg_coverage = _scan_logs_for_hg_coverage(t2_session_id) if t2_session_id else {}
    t2_p2b_validation = _scan_logs_for_p2b_validation(t2_session_id) if t2_session_id else {}
    t2_repairs = _count_repairs(t2_id)
    t2_core_text = _read_safe(t1_core_path) if t1_core_path.exists() else ""
    t2_final_assess = _assess_mutable_default(t2_core_text)
    # We can't easily extract first_plan_text from direct_ollama path — check DB
    t2_first_plan_assess = _get_first_plan_mutable_default(t2_id)
    t2_pytest = _run_pytest(workspace) if t2_status == "done" else {"passed": False, "stdout": ""}

    print(f"  [HG_COVERAGE] any_eligible={t2_hg_coverage.get('any_eligible')}")
    print(f"  [P2B] violation_detected={t2_p2b_validation.get('violation_detected')} "
          f"validation_count={t2_p2b_validation.get('validation_count')}")
    print(f"  [first_plan] has_mutable_default={t2_first_plan_assess.get('has_mutable_default')}")
    print(f"  [final] has_mutable_default={t2_final_assess.get('has_mutable_default')} "
          f"has_none_default={t2_final_assess.get('has_none_default')}")
    print(f"  planning_repairs={t2_repairs.get('planning_repairs')} "
          f"debug_repairs={t2_repairs.get('debug_repairs')}")

    # -------------------------------------------------------------------
    # T3: report_label — logging trap
    # -------------------------------------------------------------------
    _wait_slot()
    t3 = _api("POST", "/api/v1/tasks", json={
        "project_id": project_id,
        "title": T3_TITLE,
        "description": T3_DESC,
        "plan_position": 3,
        "execution_profile": "full_lifecycle",
    })
    t3_id = t3["id"]
    print(f"\n[T3] id={t3_id} — Dispatching (logging trap)...")
    t3_start = time.time()
    _dispatch(t3_id)
    t3_result = _poll(t3_id)
    t3_elapsed = round(time.time() - t3_start, 1)
    t3_status = t3_result.get("status")
    t3_session_id = t3_result.get("session_id", 0)
    print(f"[T3] {t3_status} in {t3_elapsed}s session_id={t3_session_id}")

    t3_hg_coverage = _scan_logs_for_hg_coverage(t3_session_id) if t3_session_id else {}
    t3_p2b_validation = _scan_logs_for_p2b_validation(t3_session_id) if t3_session_id else {}
    t3_repairs = _count_repairs(t3_id)
    t3_core_text = _read_safe(t1_core_path) if t1_core_path.exists() else ""
    t3_final_assess = _assess_logging(t3_core_text)
    t3_first_plan_assess = _get_first_plan_logging(t3_id)
    t3_pytest = _run_pytest(workspace) if t3_status == "done" else {"passed": False, "stdout": ""}

    print(f"  [HG_COVERAGE] any_eligible={t3_hg_coverage.get('any_eligible')}")
    print(f"  [P2B] violation_detected={t3_p2b_validation.get('violation_detected')} "
          f"validation_count={t3_p2b_validation.get('validation_count')}")
    print(f"  [first_plan] has_logging_import={t3_first_plan_assess.get('has_logging_import')}")
    print(f"  [final] has_logging_import={t3_final_assess.get('has_logging_import')} "
          f"has_print={t3_final_assess.get('has_print')}")
    print(f"  planning_repairs={t3_repairs.get('planning_repairs')} "
          f"debug_repairs={t3_repairs.get('debug_repairs')}")

    # -------------------------------------------------------------------
    # Collect conflict/usage rows
    # -------------------------------------------------------------------
    conflicts = _query_conflicts(project_id)
    usage = _query_usage([t1_id, t2_id, t3_id])
    print(f"\n[db] conflict rows: {len(conflicts)}")
    print(f"[db] usage rows: {len(usage)}")

    return _build_result(
        commit_sha=commit_sha,
        project_id=project_id,
        worker_info=worker_info,
        prompt_sizes=prompt_sizes,
        activation_row=activation_row,
        readiness=readiness,
        trimming_check=trimming_check,
        guidance_ids=guidance_ids,
        t1_id=t1_id, t1_status=t1_status, t1_elapsed=t1_elapsed,
        t1_session_id=t1_session_id,
        t1_hg_coverage=t1_hg_coverage,
        t1_p2b_validation=t1_p2b_validation,
        t1_repairs=t1_repairs,
        t1_core_text=t1_core_text,
        t2_id=t2_id, t2_status=t2_status, t2_elapsed=t2_elapsed,
        t2_session_id=t2_session_id,
        t2_hg_coverage=t2_hg_coverage,
        t2_p2b_validation=t2_p2b_validation,
        t2_repairs=t2_repairs,
        t2_first_plan_assess=t2_first_plan_assess,
        t2_final_assess=t2_final_assess,
        t2_core_text=t2_core_text,
        t2_pytest=t2_pytest,
        t3_id=t3_id, t3_status=t3_status, t3_elapsed=t3_elapsed,
        t3_session_id=t3_session_id,
        t3_hg_coverage=t3_hg_coverage,
        t3_p2b_validation=t3_p2b_validation,
        t3_repairs=t3_repairs,
        t3_first_plan_assess=t3_first_plan_assess,
        t3_final_assess=t3_final_assess,
        t3_core_text=t3_core_text,
        t3_pytest=t3_pytest,
        conflicts=conflicts,
        usage=usage,
        agent_exists=agent_exists,
        hg_in_wm_count=len(hg_in_wm),
    )


# ---------------------------------------------------------------------------
# First-plan extraction from task steps
# ---------------------------------------------------------------------------

def _get_first_plan_mutable_default(task_id: int) -> dict:
    from app.database import SessionLocal
    from app.models import Task as TaskModel

    db = SessionLocal()
    try:
        t = db.query(TaskModel).filter(TaskModel.id == task_id).first()
        if not t or not t.steps:
            return {"found": False}
        steps = t.steps if isinstance(t.steps, list) else json.loads(t.steps or "[]")
        content = _concat_write_content(steps)
        return {"found": bool(content), **_assess_mutable_default(content)}
    except Exception as e:
        return {"found": False, "error": str(e)}
    finally:
        db.close()


def _get_first_plan_logging(task_id: int) -> dict:
    from app.database import SessionLocal
    from app.models import Task as TaskModel

    db = SessionLocal()
    try:
        t = db.query(TaskModel).filter(TaskModel.id == task_id).first()
        if not t or not t.steps:
            return {"found": False}
        steps = t.steps if isinstance(t.steps, list) else json.loads(t.steps or "[]")
        content = _concat_write_content(steps)
        return {"found": bool(content), **_assess_logging(content)}
    except Exception as e:
        return {"found": False, "error": str(e)}
    finally:
        db.close()


def _concat_write_content(steps: list) -> str:
    parts = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        for op in step.get("ops") or []:
            if not isinstance(op, dict):
                continue
            if op.get("op") == "write_file":
                content = op.get("content") or ""
                if content:
                    parts.append(content)
            elif op.get("op") == "replace_in_file":
                new_text = op.get("new") or ""
                if new_text:
                    parts.append(new_text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(**kw) -> dict:
    t2_p2b = kw["t2_p2b_validation"]
    t3_p2b = kw["t3_p2b_validation"]
    t2_repairs = kw["t2_repairs"]
    t3_repairs = kw["t3_repairs"]
    t2_final = kw["t2_final_assess"]
    t3_final = kw["t3_final_assess"]
    t1_hg = kw["t1_hg_coverage"]

    pass_criteria = {
        "hg_p2b_eligible_reported": t1_hg.get("any_eligible", False),
        "t2_violation_detected": t2_p2b.get("violation_detected", False),
        "t2_repair_triggered": (t2_repairs.get("planning_repairs", 0) or 0) > 0,
        "t2_final_no_mutable_default": not t2_final.get("has_mutable_default", True),
        "t3_violation_detected": t3_p2b.get("violation_detected", False),
        "t3_repair_triggered": (t3_repairs.get("planning_repairs", 0) or 0) > 0,
        "t3_final_no_logging": not t3_final.get("has_logging_import", True),
        "t3_final_uses_print": t3_final.get("has_print", False),
        "t1_done": kw["t1_status"] == "done",
        "t2_done": kw["t2_status"] == "done",
        "t3_done": kw["t3_status"] == "done",
    }
    overall_pass = all(pass_criteria.values())

    return {
        "run_id": "R4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit_sha": kw["commit_sha"],
        "project_id": kw["project_id"],
        "worker_info": kw["worker_info"],
        "prompt_sizes": kw["prompt_sizes"],
        "activation_row": kw["activation_row"],
        "readiness": kw["readiness"],
        "trimming_check": kw["trimming_check"],
        "guidance_ids": kw["guidance_ids"],
        "agent_dir_exists_before_t2": kw["agent_exists"],
        "hg_in_wm_after_t1": kw["hg_in_wm_count"],
        "t1": {
            "id": kw["t1_id"], "status": kw["t1_status"],
            "elapsed_s": kw["t1_elapsed"], "session_id": kw["t1_session_id"],
            "hg_coverage": kw["t1_hg_coverage"],
            "p2b_validation": kw["t1_p2b_validation"],
            "repairs": kw["t1_repairs"],
            "core_text_sample": kw["t1_core_text"][:400],
        },
        "t2": {
            "id": kw["t2_id"], "status": kw["t2_status"],
            "elapsed_s": kw["t2_elapsed"], "session_id": kw["t2_session_id"],
            "hg_coverage": kw["t2_hg_coverage"],
            "p2b_validation": kw["t2_p2b_validation"],
            "repairs": kw["t2_repairs"],
            "first_plan_assess": kw["t2_first_plan_assess"],
            "final_assess": kw["t2_final_assess"],
            "pytest": kw["t2_pytest"],
            "core_text_sample": kw["t2_core_text"][:600],
        },
        "t3": {
            "id": kw["t3_id"], "status": kw["t3_status"],
            "elapsed_s": kw["t3_elapsed"], "session_id": kw["t3_session_id"],
            "hg_coverage": kw["t3_hg_coverage"],
            "p2b_validation": kw["t3_p2b_validation"],
            "repairs": kw["t3_repairs"],
            "first_plan_assess": kw["t3_first_plan_assess"],
            "final_assess": kw["t3_final_assess"],
            "pytest": kw["t3_pytest"],
            "core_text_sample": kw["t3_core_text"][:600],
        },
        "conflicts": kw["conflicts"],
        "usage": kw["usage"],
        "pass_criteria": pass_criteria,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    result = run()

    raw_dir = REPO_ROOT / "docs/roadmap/reports/maintenance"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = raw_dir / f"hg-p2b-strict-validation-r4-raw-{ts}.json"
    raw_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[raw] {raw_path}")

    print("\n" + "="*60)
    print("R4 PASS CRITERIA")
    print("="*60)
    for k, v in result["pass_criteria"].items():
        mark = "PASS" if v else "FAIL"
        print(f"  [{mark}] {k}: {v}")
    print(f"\nOVERALL: {'PASS' if result['overall_pass'] else 'FAIL'}")
    print("="*60)

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("overall_pass") else 1)

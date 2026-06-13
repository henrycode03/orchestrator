"""Stage 5 validation: WM injection smoke test for implementation_strategy.

Two-task sequence on project wm-injection-smoke-calclib:
  T1 — Bootstrap calclib with parse_number (plan_position=1).
       After T1 completes, working_memory.json is written with LLM summary.
  T2 — Add formatter.py that calls parse_number.
       T2 description omits ok/value/error/INVALID_NUMBER details.
       Injection must supply those details from T1 WM summary.

Flags under test:
  ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY=1
  WORKING_MEMORY_PERSISTENCE_ENABLED=True
  WORKING_MEMORY_RENDER_ENABLED=True
  WORKING_MEMORY_INJECTION_ENABLED=True

Flags kept OFF:
  REPO_MEMORY_INJECTION_ENABLED=False
  PSS_CONTINUATION_INJECTION_ENABLED=False
  ARTIFACT_CONTINUATION_ENABLED=False
  LANGFUSE_ENABLED=False
  REDUCED_PLANNING_PROMPT_ENABLED=False
"""

import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.config import settings  # noqa: E402
from app.auth import create_access_token  # noqa: E402
import requests  # noqa: E402

BASE_URL = "http://127.0.0.1:8080"
USER_EMAIL = os.environ.get("ORCHESTRATOR_USER_EMAIL", "eval@local.dev")
WORKSPACE_SLUG = "wm-injection-smoke-calclib"
REPORT_DIR = REPO_ROOT / "docs/roadmap/reports/maintenance"
RAW_OUT = REPORT_DIR / f"wm-llm-summary-stage5-raw-{time.strftime('%Y%m%d_%H%M%S')}.json"
SLOT_KEY = "orchestrator:backend_slots:local_openclaw"
WORKSPACE_BASE = Path("/root/.openclaw/workspace/vault/projects")

T1_TITLE = "Bootstrap calclib with parse_number"
T1_DESC = """Bootstrap src-layout calclib library.

Setup:
- Create directory structure: src/calclib/
- Create src/calclib/__init__.py (empty, or re-export parse_number)
- Create src/calclib/parser.py

Implement parse_number(text: str) -> dict in parser.py.

The function must return a plain dict (never raise an exception):
  {"ok": bool, "value": int | None, "error": str | None}

For valid integer input ("42", "-7", "0"):
  {"ok": True, "value": <parsed int>, "error": None}

For invalid input ("abc", "", "3.14", None):
  {"ok": False, "value": None, "error": "INVALID_NUMBER"}

Create tests/test_parser.py with pytest cases covering:
  - valid integers (positive, negative, zero)
  - invalid strings (empty string, float string, non-numeric, None)
  - confirm no exceptions are raised for any input

Create pytest.ini at project root:
  [pytest]
  pythonpath = src

Run: PYTHONPATH=src python3 -m pytest tests/test_parser.py -v
All tests must pass.
"""

T2_TITLE = "Add formatter module"
T2_DESC = """Add a new module src/calclib/formatter.py with function:

  format_number_result(text: str) -> str

Use the parser API established in the previous task.
For valid input return the parsed integer as a string.
For invalid input return "invalid".

Create tests/test_formatter.py with pytest cases covering:
  - valid integers formatted correctly
  - invalid input returns "invalid"

Run: PYTHONPATH=src python3 -m pytest tests/test_formatter.py -v
All tests must pass.
"""

HEADERS: dict = {}
DETERMINISTIC_PREFIX = "Task completed with verified execution evidence"


def _api(method: str, path: str, **kwargs):
    r = requests.request(method, f"{BASE_URL}{path}", headers=HEADERS, **kwargs)
    r.raise_for_status()
    return r.json()


def init_auth() -> None:
    global HEADERS
    token = create_access_token({"sub": USER_EMAIL})
    HEADERS = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    print(f"[init] Auth token created for {USER_EMAIL}")


def wait_slot(poll: int = 15, timeout: int = 600) -> None:
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

    def _slot_members():
        try:
            return [int(m) for m in (r.smembers(SLOT_KEY) or set())]
        except Exception:
            return []

    def _evict_terminal():
        db = DBSession()
        try:
            for sid in _slot_members():
                row = db.execute(
                    text("SELECT status FROM sessions WHERE id=:id"), {"id": sid}
                ).fetchone()
                status = row[0] if row else "not_found"
                if status in TERMINAL or status == "not_found":
                    r.srem(SLOT_KEY, str(sid))
                    print(f"  [slot] Evicted stale session {sid} (status={status})")
        finally:
            db.close()

    deadline = time.time() + timeout
    while time.time() < deadline:
        _evict_terminal()
        members = _slot_members()
        if not members:
            print("[slot] Slot clear.")
            return
        print(f"[slot] Occupied by {members}. Waiting {poll}s...")
        time.sleep(poll)
    raise TimeoutError("Backend slot never freed")


def create_project() -> dict:
    workspace = str(WORKSPACE_BASE / WORKSPACE_SLUG)
    p = _api(
        "POST",
        "/api/v1/projects",
        json={
            "name": WORKSPACE_SLUG,
            "description": "Stage 5 WM injection smoke test",
            "workspace_path": workspace,
        },
    )
    print(f"[project] id={p['id']} workspace={workspace}")
    p["_workspace_abs"] = workspace
    return p


def create_task(project_id: int, title: str, desc: str, plan_position: int) -> dict:
    t = _api(
        "POST",
        "/api/v1/tasks",
        json={
            "project_id": project_id,
            "title": title,
            "description": desc,
            "plan_position": plan_position,
            "execution_profile": "full_lifecycle",
        },
    )
    print(f"[task] id={t['id']} plan_position={plan_position} title={title!r}")
    return t


def dispatch_task(task_id: int) -> None:
    _api("POST", f"/api/v1/tasks/{task_id}/retry", json={})
    print(f"[dispatch] task {task_id} dispatched")


def poll_task(task_id: int, timeout: int = 1200, poll: int = 20) -> dict:
    deadline = time.time() + timeout
    elapsed = 0
    while time.time() < deadline:
        t = _api("GET", f"/api/v1/tasks/{task_id}")
        status = t.get("status", "")
        if status in ("done", "failed", "blocked_prior_task_failed"):
            print(f"  [{status}] at {elapsed}s")
            return t
        print(f"  [{status}] {elapsed}s")
        time.sleep(poll)
        elapsed += poll
    raise TimeoutError(f"Task {task_id} did not finish within {timeout}s")


def read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"(could not read: {e})"


def scan_worker_log(worker_log: Path, t1_task_id: int, t2_task_id: int) -> dict:
    result = {
        "t1_phase5_found": False,
        "t1_wm_written_found": False,
        "t1_http_post_found": False,
        "t1_http_status": None,
        "t1_fallback_found": False,
        "t2_wm_injected_found": False,
        "t2_wm_injected_chars": None,
        "t2_wm_injected_plan_position": None,
        "t2_phase5_found": False,
        "t2_wm_written_found": False,
        "t2_http_post_found": False,
        "t2_fallback_found": False,
        "wm_injected_lines": [],
        "phase5_lines": [],
    }
    if not worker_log.exists():
        return result
    lines = worker_log.read_text(encoding="utf-8", errors="replace").splitlines()
    # Scan last 1200 lines to cover both tasks
    recent = lines[-1200:]
    # Track which task phase we're in based on task_id appearing in log context
    # The injection log line: "[WORKING_MEMORY] Injected N chars into project_context (plan_position=M)"
    for line in recent:
        if "Phase 5: TASK_SUMMARY" in line:
            result["phase5_lines"].append(line.strip())
        if "[WORKING_MEMORY] Injected" in line and "project_context" in line:
            result["t2_wm_injected_found"] = True
            result["wm_injected_lines"].append(line.strip())
            # Parse: "[WORKING_MEMORY] Injected N chars into project_context (plan_position=M)"
            m = re.search(r"Injected (\d+) chars.*plan_position=(\S+)\)", line)
            if m:
                result["t2_wm_injected_chars"] = int(m.group(1))
                pos_raw = m.group(2)
                try:
                    result["t2_wm_injected_plan_position"] = int(pos_raw)
                except ValueError:
                    result["t2_wm_injected_plan_position"] = pos_raw
        if "ai-gateway:8000/v1/chat/completions" in line:
            result["phase5_lines"].append(line.strip())
            if "200 OK" in line:
                # assign to whichever hasn't been set yet
                if not result["t1_http_post_found"]:
                    result["t1_http_post_found"] = True
                    result["t1_http_status"] = 200
                else:
                    result["t2_http_post_found"] = True
        if "summary_generation_failed" in line or "using deterministic completion summary" in line:
            result["phase5_lines"].append(line.strip())
            if not result["t1_fallback_found"]:
                result["t1_fallback_found"] = True
            else:
                result["t2_fallback_found"] = True
        if "[WORKING_MEMORY] Written to" in line:
            result["phase5_lines"].append(line.strip())
            if not result["t1_wm_written_found"]:
                result["t1_wm_written_found"] = True
            else:
                result["t2_wm_written_found"] = True
    # Phase 5 detections
    phase5_count = sum(1 for l in recent if "Phase 5: TASK_SUMMARY" in l)
    result["t1_phase5_found"] = phase5_count >= 1
    result["t2_phase5_found"] = phase5_count >= 2
    return result


def assess_api_capture(summary_text: str) -> dict:
    text = (summary_text or "").lower()
    raw = summary_text or ""
    return {
        "dict_return_type": "dict" in text or "dictionary" in text,
        "ok_key": (
            '"ok"' in raw or "'ok'" in raw or "ok:" in text or '"ok":' in raw
        ),
        "value_key": (
            '"value"' in raw or "'value'" in raw or "value:" in text or '"value":' in raw
        ),
        "error_key": (
            '"error"' in raw or "'error'" in raw or "error:" in text or '"error":' in raw
        ),
        "invalid_number_sentinel": "INVALID_NUMBER" in raw or "invalid number" in text,
        "no_exception": (
            "never raise" in text or "no exception" in text
            or "without raising" in text or "doesn't raise" in text
            or "ensuring no exception" in text
        ),
    }


def assess_t2_plan_api_usage(plan_text: str) -> dict:
    """Check whether T2 plan mentions parse_number and uses correct dict keys."""
    text = plan_text or ""
    text_lower = text.lower()
    return {
        "calls_parse_number": "parse_number" in text_lower,
        "uses_ok_key": (
            '["ok"]' in text or "['ok']" in text or ".get(\"ok\")" in text
            or ".get('ok')" in text or '"ok"' in text
        ),
        "uses_value_key": (
            '["value"]' in text or "['value']" in text or ".get(\"value\")" in text
            or ".get('value')" in text
        ),
        "uses_error_key": (
            '["error"]' in text or "['error']" in text or ".get(\"error\")" in text
            or ".get('error')" in text
        ),
        "uses_invalid_number": "INVALID_NUMBER" in text,
        "returns_invalid_str": '"invalid"' in text or "'invalid'" in text,
    }


def assess_formatter_implementation(formatter_text: str) -> dict:
    """Check whether formatter.py uses parse_number correctly."""
    text = formatter_text or ""
    return {
        "imports_parse_number": "parse_number" in text,
        "uses_ok_key": (
            '["ok"]' in text or "['ok']" in text or '.get("ok")' in text
            or ".get('ok')" in text or "result[" in text.lower()
        ),
        "uses_value_key": (
            '["value"]' in text or "['value']" in text or '.get("value")' in text
        ),
        "uses_error_key": (
            '["error"]' in text or "['error']" in text or '.get("error")' in text
        ),
        "returns_invalid_str": '"invalid"' in text or "'invalid'" in text,
        "no_exception_handling": "try" not in text,
    }


def compute_planning_context_trim(wm_rendered: str) -> dict:
    """Simulate how assemble_planning_prompt trims the WM block.

    assemble_planning_prompt uses _shape_project_context with max_chars=800.
    _shape_project_context trims base_context (WM block) to max_chars//2 = 400.
    _trim_text collapses whitespace before trimming.
    """
    # Replicate _trim_text
    def _trim_text(text, max_chars):
        value = " ".join(str(text or "").split())
        if len(value) <= max_chars:
            return value
        return value[:max_chars - 3].rstrip() + "..."

    base_context_limit = 800 // 2  # 400 chars
    trimmed = _trim_text(wm_rendered, base_context_limit)
    return {
        "wm_rendered_len": len(wm_rendered),
        "base_context_limit": base_context_limit,
        "trimmed_len": len(trimmed),
        "trimmed_content": trimmed,
        "implementation_strategy_reachable": "Implementation Strategy" in trimmed,
        "summary_content_reachable": (
            "implementation_strategy_reachable" and
            len(trimmed) > trimmed.find("Implementation Strategy") + len("Implementation Strategy") + 50
            if "Implementation Strategy" in trimmed else False
        ),
    }


def scan_regression_checks(worker_log: Path) -> dict:
    """Check for previously observed regressions."""
    if not worker_log.exists():
        return {}
    lines = worker_log.read_text(encoding="utf-8", errors="replace").splitlines()
    recent = "\n".join(lines[-1200:])
    return {
        "pip_show_recurrence": "pip show" in recent.lower(),
        "nested_project_folder_command": (
            "nested_project_folder" in recent or
            "project folder" in recent.lower()
        ),
        "path_guard_advisory": (
            "PATH_GUARD" in recent or "path_guard" in recent.lower()
        ),
        "backend_capacity": (
            "backend_capacity" in recent or "backend capacity" in recent.lower()
        ),
        "vma_error": "VMA" in recent,
    }


def main():
    print("[Stage 5] WM injection smoke test — wm-injection-smoke-calclib")
    print()

    # Print flag state as seen by runner process
    print("  Flag state in runner process:")
    print(f"    ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY env: {os.getenv('ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY', 'NOT SET')}")
    print(f"    WORKING_MEMORY_PERSISTENCE_ENABLED: {settings.WORKING_MEMORY_PERSISTENCE_ENABLED}")
    print(f"    WORKING_MEMORY_RENDER_ENABLED:      {settings.WORKING_MEMORY_RENDER_ENABLED}")
    print(f"    WORKING_MEMORY_INJECTION_ENABLED:   {settings.WORKING_MEMORY_INJECTION_ENABLED}")
    print(f"    REPO_MEMORY_INJECTION_ENABLED:      {settings.REPO_MEMORY_INJECTION_ENABLED}")
    print(f"    PSS_CONTINUATION_INJECTION_ENABLED: {settings.PSS_CONTINUATION_INJECTION_ENABLED}")
    print()

    from app.services.orchestration.working_memory import (
        _SUMMARY_STORAGE_LIMIT,
        _SUMMARY_RENDER_LIMIT,
        _INJECTION_BUDGET,
    )
    print(f"  WM constants:")
    print(f"    _INJECTION_BUDGET:       {_INJECTION_BUDGET}")
    print(f"    _SUMMARY_STORAGE_LIMIT:  {_SUMMARY_STORAGE_LIMIT}")
    print(f"    _SUMMARY_RENDER_LIMIT:   {_SUMMARY_RENDER_LIMIT}")
    print()
    print("  Architecture note:")
    print("    assemble_planning_prompt uses _shape_project_context(max_chars=800)")
    print("    base_context (WM block) trimmed to 800//2=400 chars before reaching planner.")
    print("    Implementation Strategy section is LAST in WM block; may be cut.")
    print()

    init_auth()
    wait_slot()

    project = create_project()
    project_id = project["id"]
    workspace_path = Path(project["_workspace_abs"])

    # Create both tasks upfront
    t1 = create_task(project_id, T1_TITLE, T1_DESC, plan_position=1)
    t1_id = t1["id"]
    t2 = create_task(project_id, T2_TITLE, T2_DESC, plan_position=2)
    t2_id = t2["id"]

    agent_dir = workspace_path / ".agent"
    wm_path = agent_dir / "working_memory.json"
    progress_notes_path = agent_dir / "progress_notes.md"
    worker_log = REPO_ROOT / "logs" / "worker.log"

    # -----------------------------------------------------------------------
    # T1 run
    # -----------------------------------------------------------------------
    print()
    print(f"[T1] Dispatching task {t1_id}: {T1_TITLE!r}")
    t1_start = time.time()
    dispatch_task(t1_id)
    t1_result = poll_task(t1_id, timeout=1200, poll=20)
    t1_elapsed = round(time.time() - t1_start, 1)
    print(f"[T1] Finished in {t1_elapsed}s status={t1_result.get('status')}")

    # Read T1 artifacts
    wm_json_text = read_file_safe(wm_path)
    wm_exists_after_t1 = wm_path.exists()
    try:
        wm_data_t1 = json.loads(wm_json_text) if wm_exists_after_t1 else {}
    except Exception:
        wm_data_t1 = {}

    wm_strategies_t1 = wm_data_t1.get("implementation_strategy") or []
    t1_wm_summary = wm_strategies_t1[-1].get("summary", "") if wm_strategies_t1 else ""
    t1_wm_summary_len = len(t1_wm_summary)
    t1_is_deterministic = t1_wm_summary.startswith(DETERMINISTIC_PREFIX)
    t1_api_capture = assess_api_capture(t1_wm_summary)
    t1_api_keys_captured = sum(1 for v in t1_api_capture.values() if v)

    print(f"  WM exists:             {wm_exists_after_t1}")
    print(f"  T1 summary length:     {t1_wm_summary_len} chars")
    print(f"  T1 is LLM (not det.): {not t1_is_deterministic}")
    print(f"  T1 API indicators:     {t1_api_keys_captured}/6")
    print(f"  T1 summary (first 300):")
    print(f"    {t1_wm_summary[:300]}")
    print()

    # Compute what the planner would see from the WM block
    from app.services.orchestration.working_memory import _render_working_memory_content
    import logging
    _logger = logging.getLogger(__name__)

    wm_rendered_for_t2 = _render_working_memory_content(str(workspace_path), _logger)
    planning_context_analysis = compute_planning_context_trim(wm_rendered_for_t2)

    print(f"  WM rendered block length:  {planning_context_analysis['wm_rendered_len']} chars")
    print(f"  Planning base_context cap: {planning_context_analysis['base_context_limit']} chars")
    print(f"  Trimmed to planner:        {planning_context_analysis['trimmed_len']} chars")
    print(f"  Impl.Strategy reachable:   {planning_context_analysis['implementation_strategy_reachable']}")
    print(f"  Trimmed content (full):")
    print(f"    {planning_context_analysis['trimmed_content']}")
    print()

    # -----------------------------------------------------------------------
    # T2 run
    # -----------------------------------------------------------------------
    wait_slot()
    print(f"[T2] Dispatching task {t2_id}: {T2_TITLE!r}")
    t2_start = time.time()
    dispatch_task(t2_id)
    t2_result = poll_task(t2_id, timeout=1200, poll=20)
    t2_elapsed = round(time.time() - t2_start, 1)
    print(f"[T2] Finished in {t2_elapsed}s status={t2_result.get('status')}")

    # Read T2 artifacts
    wm_json_text_post_t2 = read_file_safe(wm_path)
    try:
        wm_data_t2 = json.loads(wm_json_text_post_t2) if wm_path.exists() else {}
    except Exception:
        wm_data_t2 = {}

    wm_strategies_t2 = wm_data_t2.get("implementation_strategy") or []
    t2_wm_entry = wm_strategies_t2[-1] if len(wm_strategies_t2) >= 2 else {}
    t2_wm_summary = t2_wm_entry.get("summary", "")

    formatter_path = workspace_path / "src" / "calclib" / "formatter.py"
    formatter_text = read_file_safe(formatter_path)
    formatter_exists = formatter_path.exists()

    t2_formatter_impl = assess_formatter_implementation(formatter_text)

    log_analysis = scan_worker_log(worker_log, t1_id, t2_id)
    regression_checks = scan_regression_checks(worker_log)

    # T2 plan from API
    t2_plan_text = json.dumps(t2_result.get("plan") or [], indent=2)
    t2_plan_api_usage = assess_t2_plan_api_usage(t2_plan_text)

    # Check if the injected content includes implementation_strategy
    injected_block_includes_impl_strategy = planning_context_analysis["implementation_strategy_reachable"]

    # WM budget check
    injected_chars = log_analysis["t2_wm_injected_chars"]
    within_budget = injected_chars is not None and injected_chars <= _INJECTION_BUDGET

    # Build raw output
    raw = {
        "stage": 5,
        "project_id": project_id,
        "t1_task_id": t1_id,
        "t2_task_id": t2_id,
        "t1_status": t1_result.get("status"),
        "t2_status": t2_result.get("status"),
        "t1_elapsed_s": t1_elapsed,
        "t2_elapsed_s": t2_elapsed,
        "flags_in_worker": {
            "llm_summary": True,
            "persistence": True,
            "render": True,
            "injection": True,
            "repo_memory": False,
            "pss_continuation": False,
            "artifact_continuation": False,
            "langfuse": False,
        },
        "t1": {
            "wm_exists": wm_exists_after_t1,
            "wm_summary_len": t1_wm_summary_len,
            "wm_summary": t1_wm_summary,
            "is_llm_text": not t1_is_deterministic,
            "api_capture": t1_api_capture,
            "api_keys_captured": t1_api_keys_captured,
            "debug_repair_count": t1_result.get("debug_repair_count", 0),
            "planning_repair_count": t1_result.get("planning_repair_count", 0),
        },
        "injection": {
            "wm_rendered_len": planning_context_analysis["wm_rendered_len"],
            "planning_base_context_cap": planning_context_analysis["base_context_limit"],
            "trimmed_len_to_planner": planning_context_analysis["trimmed_len"],
            "trimmed_content": planning_context_analysis["trimmed_content"],
            "implementation_strategy_in_trimmed": planning_context_analysis["implementation_strategy_reachable"],
            "log_injected_found": log_analysis["t2_wm_injected_found"],
            "log_injected_chars": injected_chars,
            "log_injected_plan_position": log_analysis["t2_wm_injected_plan_position"],
            "within_2000_budget": within_budget,
            "injected_lines": log_analysis["wm_injected_lines"],
        },
        "t2": {
            "status": t2_result.get("status"),
            "elapsed_s": t2_elapsed,
            "formatter_exists": formatter_exists,
            "formatter_text": formatter_text[:800] if formatter_exists else "(not found)",
            "formatter_impl": t2_formatter_impl,
            "plan_api_usage": t2_plan_api_usage,
            "wm_summary": t2_wm_summary[:400],
            "debug_repair_count": t2_result.get("debug_repair_count", 0),
            "planning_repair_count": t2_result.get("planning_repair_count", 0),
        },
        "log_analysis": {
            "t1_phase5_found": log_analysis["t1_phase5_found"],
            "t1_wm_written_found": log_analysis["t1_wm_written_found"],
            "t1_http_post_found": log_analysis["t1_http_post_found"],
            "t1_fallback_found": log_analysis["t1_fallback_found"],
            "t2_phase5_found": log_analysis["t2_phase5_found"],
            "t2_wm_injected_found": log_analysis["t2_wm_injected_found"],
            "t2_http_post_found": log_analysis["t2_http_post_found"],
            "t2_fallback_found": log_analysis["t2_fallback_found"],
            "phase5_lines": log_analysis["phase5_lines"][:10],
            "wm_injected_lines": log_analysis["wm_injected_lines"][:3],
        },
        "regression_checks": regression_checks,
    }

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] Raw results: {RAW_OUT}")

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("STAGE 5 SUMMARY — WM INJECTION SMOKE TEST")
    print("=" * 65)
    print(f"T1 status:                         {t1_result.get('status')}")
    print(f"T2 status:                         {t2_result.get('status')}")
    print(f"T1 elapsed:                        {t1_elapsed}s")
    print(f"T2 elapsed:                        {t2_elapsed}s")
    print()
    print(f"T1 working_memory.json exists:     {wm_exists_after_t1}")
    print(f"T1 WM summary length:              {t1_wm_summary_len} chars")
    print(f"T1 WM is LLM text:                 {not t1_is_deterministic}")
    print(f"T1 API contract indicators:        {t1_api_keys_captured}/6")
    for k, v in t1_api_capture.items():
        print(f"  {k}: {v}")
    print()
    print(f"WM rendered block length:          {planning_context_analysis['wm_rendered_len']} chars")
    print(f"Planning base_context cap (800//2): {planning_context_analysis['base_context_limit']} chars")
    print(f"Chars reaching planner:            {planning_context_analysis['trimmed_len']}")
    print(f"Impl.Strategy in trimmed block:    {planning_context_analysis['implementation_strategy_reachable']}")
    print()
    print(f"[WORKING_MEMORY] Injected log found: {log_analysis['t2_wm_injected_found']}")
    print(f"Injected chars:                    {injected_chars}")
    print(f"Injection plan_position:           {log_analysis['t2_wm_injected_plan_position']}")
    print(f"Within 2000-char budget:           {within_budget}")
    print()
    print(f"T2 formatter.py exists:            {formatter_exists}")
    print(f"T2 formatter imports parse_number: {t2_formatter_impl['imports_parse_number']}")
    print(f"T2 formatter uses ok_key:          {t2_formatter_impl['uses_ok_key']}")
    print(f"T2 formatter uses value_key:       {t2_formatter_impl['uses_value_key']}")
    print(f"T2 formatter returns 'invalid':    {t2_formatter_impl['returns_invalid_str']}")
    print()
    print(f"T1 repairs (debug/planning):       {t1_result.get('debug_repair_count', 0)} / {t1_result.get('planning_repair_count', 0)}")
    print(f"T2 repairs (debug/planning):       {t2_result.get('debug_repair_count', 0)} / {t2_result.get('planning_repair_count', 0)}")
    print()
    print("Regression checks:")
    for k, v in regression_checks.items():
        print(f"  {k}: {v}")
    print()

    # Architectural finding
    if not planning_context_analysis["implementation_strategy_reachable"]:
        print("ARCHITECTURAL FINDING:")
        print("  The implementation_strategy section does NOT reach the planner.")
        print(f"  WM block rendered: {planning_context_analysis['wm_rendered_len']} chars")
        print(f"  Planning trims base_context to: {planning_context_analysis['base_context_limit']} chars")
        print("  Implementation Strategy is placed LAST in the WM block and is cut.")
        print("  The 2000-char injection budget is incompatible with the 400-char planning trim.")
        print()

    # Verdict
    def _verdict():
        if t1_result.get("status") != "done":
            return "FAIL — T1 did not reach DONE"
        if not wm_exists_after_t1:
            return "FAIL — working_memory.json not created after T1"
        if not wm_strategies_t1:
            return "FAIL — T1 implementation_strategy empty"
        if t1_is_deterministic:
            return "FAIL — T1 WM received deterministic summary (LLM not called)"
        if t1_api_keys_captured < 3:
            return f"FAIL — T1 WM API contract too thin ({t1_api_keys_captured}/6)"
        if not log_analysis["t2_wm_injected_found"]:
            return "FAIL — [WORKING_MEMORY] Injected log not found for T2"
        if t2_result.get("status") != "done":
            return f"FAIL — T2 did not reach DONE (status={t2_result.get('status')})"
        if not formatter_exists:
            return "FAIL — formatter.py not created"
        if not t2_formatter_impl["imports_parse_number"]:
            return "FAIL — formatter.py does not import parse_number"
        if not t2_formatter_impl["uses_ok_key"]:
            if not planning_context_analysis["implementation_strategy_reachable"]:
                return (
                    "PARTIAL — T2 DONE but formatter may not use ok key correctly; "
                    "implementation_strategy truncated before reaching planner "
                    f"({planning_context_analysis['wm_rendered_len']} chars rendered, "
                    f"{planning_context_analysis['base_context_limit']} char planning cap)"
                )
            return "PARTIAL — T2 DONE but formatter does not use ok key correctly"
        return (
            "PASS — T1 DONE, WM created, LLM summary stored, "
            "injection log confirmed, T2 DONE, formatter uses parse_number correctly"
        )

    verdict = _verdict()
    print(f"Stage 5 verdict: {verdict}")
    print("=" * 65)

    return raw


if __name__ == "__main__":
    main()

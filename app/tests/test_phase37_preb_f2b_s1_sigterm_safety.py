"""Phase 37 Pre-B-F2B-S1: SIGTERM callback deadlock safety and PGID guard.

Provider-free. Containment rules for this file:

* Every real signal scenario runs in an isolated child started with
  ``start_new_session=True`` and a bounded timeout. Signals from this test
  process go only to that child's own pid, or (on timeout) to the session it
  leads.
* Inside a child, ``os.killpg`` is replaced by an allowlist wrapper that only
  forwards to process groups the child itself spawned; any other target is
  recorded as ``UNSAFE_KILLPG`` and never signaled.
* In-process tests that exercise invalid PGIDs patch ``os.killpg`` with a
  recorder that never signals. Real ``killpg`` in-process only targets groups
  this test spawned with ``start_new_session=True``.
* Placeholder pids 0, 1, this process, its group, or arbitrary existing
  groups are never registered for a real kill.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.agents import subprocess_lifecycle as sl
from app.services.orchestration.execution import binding_reconciliation as recon
from app.tests.test_phase37_preb_f2b_binding_lifecycle import (
    REPO_ROOT,
    _persistent_state,
    _snapshot,
)

CHILD_TIMEOUT_SECONDS = 20


@pytest.fixture(autouse=True)
def _reset_lifecycle_registry():
    sl._reset_for_tests()
    yield
    sl._reset_for_tests()


CHILD_SCRIPT = textwrap.dedent(
    """
    import json, os, signal, subprocess, sys, threading, time
    from pathlib import Path
    from app.services.agents import subprocess_lifecycle as sl

    events_path, mode, tmp = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    config = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    def ev(_event, **kw):
        with open(events_path, "a") as handle:
            handle.write(json.dumps({"e": _event, **kw}) + "\\n")
            handle.flush()

    OWNED = set()
    _real_killpg = os.killpg

    def guarded_killpg(pgid, sig):
        if pgid not in OWNED:
            ev("UNSAFE_KILLPG", pgid=repr(pgid), sig=int(sig))
            raise AssertionError("unexpected killpg target")
        ev("killpg", pgid=pgid, sig=int(sig))
        _real_killpg(pgid, sig)

    os.killpg = guarded_killpg

    def stat_start(pid):
        raw = open(f"/proc/{pid}/stat").read()
        return int(raw[raw.rfind(")") + 2 :].split()[19])

    def spawn_owned(with_grandchild=False):
        if with_grandchild:
            cmd = ["sh", "-c", "sleep 30 & echo $!; wait"]
            proc = subprocess.Popen(
                cmd, start_new_session=True, stdout=subprocess.PIPE, text=True,
                stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            grandchild = int(proc.stdout.readline())
        else:
            proc = subprocess.Popen(
                ["sleep", "30"], start_new_session=True, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            grandchild = None
        OWNED.add(proc.pid)
        members = [proc.pid] + ([grandchild] if grandchild else [])
        ev("spawned", pgid=proc.pid,
           members=[[pid, stat_start(pid)] for pid in members])
        sl.register_process_group(proc.pid)
        return proc

    class ArmedLock:
        # Delivers SIGTERM to this process while the wrapped lock is held by
        # the main thread, i.e. inside a lifecycle critical section.
        def __init__(self, inner):
            self.inner, self.armed = inner, False
        def __enter__(self):
            self.inner.acquire()
            if self.armed:
                self.armed = False
                ev("signal_inside_critical_section")
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(2)
                ev("RESUMED_AFTER_SIGNAL")
            return self
        def __exit__(self, *exc):
            self.inner.release()

    armed = ArmedLock(sl._lock)
    sl._lock = armed

    def cb(name, action=None):
        def _callback():
            if action is not None:
                action()
            ev("cleanup", name=name)
        return _callback

    def bind(name):
        from app.services.orchestration.execution import (
            executor_workspace_binding as ewb,
        )
        from app.services.orchestration.execution.runtime_context import (
            RuntimeExecutorContext,
        )
        runtime_root = tmp / "runtime-root"
        runtime = runtime_root / "tasks" / name
        runtime.mkdir(parents=True, exist_ok=True)
        (tmp / "product").mkdir(exist_ok=True)
        binding = ewb.bind_openclaw_workspace(
            RuntimeExecutorContext(
                executor="openclaw", runtime_workspace=runtime,
                project_workspace=tmp / "product", project_id=None,
                task_execution_id=None, runtime_root=runtime_root,
                sandbox=object(),
            ),
            real_config_path=config, model_ref="openai/qwen-local",
        )
        ev("bound", dir=str(binding._tmp_dir))
        return binding

    sl.install_sigterm_handler()

    if mode == "registration":
        sl.register_forced_termination_cleanup(cb("first"))
        spawn_owned()
        armed.armed = True
        sl.register_forced_termination_cleanup(cb("late"))
    elif mode == "unregistration":
        sl.register_forced_termination_cleanup(cb("first"))
        unregister = sl.register_forced_termination_cleanup(cb("second"))
        spawn_owned()
        armed.armed = True
        unregister()
    elif mode == "release":
        binding = bind("release")
        sl.register_forced_termination_cleanup(cb("sentinel"))
        armed.armed = True
        binding.release()
    elif mode == "repeated":
        def resignal():
            for _ in range(3):
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.05)
        spawn_owned()
        sl.register_forced_termination_cleanup(cb("a", resignal))
        sl.register_forced_termination_cleanup(cb("b"))
        os.kill(os.getpid(), signal.SIGTERM)
    elif mode == "exception":
        def boom():
            raise RuntimeError("synthetic callback failure")
        sl.register_forced_termination_cleanup(cb("boom", boom))
        sl.register_forced_termination_cleanup(cb("after"))
        os.kill(os.getpid(), signal.SIGTERM)
    elif mode == "drain":
        def register_late():
            sl.register_forced_termination_cleanup(cb("late"))
        sl.register_forced_termination_cleanup(cb("first", register_late))
        os.kill(os.getpid(), signal.SIGTERM)
    elif mode == "parent_signal":
        # Wait for SIGTERM(s) from the test process.
        for index in range(3):
            bind(f"multi-{index}")
        spawn_owned(with_grandchild=True)
        spawn_owned()
        def slow():
            time.sleep(0.5)
        sl.register_forced_termination_cleanup(cb("slow", slow))
    elif mode == "concurrent":
        spawn_owned()  # stays registered, as a running dispatch's group does
        sl.register_forced_termination_cleanup(cb("sentinel"))
        def churn(group):
            while True:
                unregister = sl.register_forced_termination_cleanup(lambda: None)
                unregister()
                sl.register_process_group(group.pid)
                sl.unregister_process_group(group.pid)
        for _ in range(4):
            threading.Thread(
                target=churn, args=(spawn_owned(),), daemon=True
            ).start()
        def main_churn():
            while True:
                unregister = sl.register_forced_termination_cleanup(lambda: None)
                unregister()
        ev("ready")
        main_churn()
    ev("ready")
    time.sleep(CHILD_WAIT)
    ev("NOT_TERMINATED")
    """
).replace("CHILD_WAIT", "15")


def _read_events(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _proc_start(pid: int):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = raw[raw.rfind(")") + 2 :].split()
    return fields[0], int(fields[19])


def _member_alive(pid: int, start: int) -> bool:
    """True only if `pid` is still the same, non-zombie process."""
    current = _proc_start(pid)
    return current is not None and current[1] == start and current[0] != "Z"


def _cleanup_spawned(events: list) -> None:
    # Only processes the child recorded, and only if pid+start still match.
    for event in events:
        if event["e"] == "spawned":
            for pid, start in event["members"]:
                if _member_alive(pid, start):
                    os.kill(pid, signal.SIGKILL)


def _start(tmp_path: Path, mode: str, config: Path | None = None, root=None):
    events = tmp_path / f"events-{mode}.jsonl"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    if root is not None:
        env["ORCHESTRATOR_OPENCLAW_BINDING_ROOT"] = str(root)
    args = [sys.executable, "-c", CHILD_SCRIPT, str(events), mode, str(tmp_path)]
    if config is not None:
        args.append(str(config))
    with open(events.with_suffix(".stderr"), "w") as stderr_file:
        proc = subprocess.Popen(
            args,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            start_new_session=True,
        )
    return proc, events


def _finish(proc, events: Path):
    """Wait boundedly for the child itself; on hang kill only its session."""
    try:
        proc.wait(timeout=CHILD_TIMEOUT_SECONDS)
        hung = False
    except subprocess.TimeoutExpired:
        if proc.poll() is None:  # unreaped: proc.pid cannot have been reused
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=CHILD_TIMEOUT_SECONDS)
        hung = True
    recorded = _read_events(events)
    _cleanup_spawned(recorded)
    return hung, proc.returncode, recorded, events.with_suffix(".stderr").read_text()


def _run(tmp_path, mode, **kwargs):
    proc, events = _start(tmp_path, mode, **kwargs)
    return _finish(proc, events)


def _names(events, kind="cleanup"):
    return [event["name"] for event in events if event["e"] == kind]


def _assert_clean_sigterm_exit(hung, code, events, stderr):
    assert not hung, f"child deadlocked; events={events} stderr={stderr[-2000:]}"
    assert code == -signal.SIGTERM, (code, events, stderr[-2000:])
    kinds = [event["e"] for event in events]
    assert "UNSAFE_KILLPG" not in kinds, events
    assert "RESUMED_AFTER_SIGNAL" not in kinds
    assert "NOT_TERMINATED" not in kinds


def _wait_members_gone(events, timeout=10.0):
    members = [m for e in events if e["e"] == "spawned" for m in e["members"]]
    assert members
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_member_alive(pid, start) for pid, start in members):
            return
        time.sleep(0.05)
    raise AssertionError(f"managed group members survived: {members}")


def _sigterm_killpgs(events):
    return [
        e["pgid"]
        for e in events
        if e["e"] == "killpg" and e["sig"] == int(signal.SIGTERM)
    ]


# --------------------------------------------------------------------------
# A/B/C — SIGTERM inside a critical section (registration, unregistration,
# binding release) no longer deadlocks
# --------------------------------------------------------------------------


def test_a_sigterm_during_callback_registration(tmp_path):
    hung, code, events, stderr = _run(tmp_path, "registration")
    _assert_clean_sigterm_exit(hung, code, events, stderr)
    assert "signal_inside_critical_section" in [e["e"] for e in events]
    # Already-registered cleanup runs; the interrupted registration had not
    # appended yet, and its frame never resumes.
    assert _names(events) == ["first"]
    spawned = [e["pgid"] for e in events if e["e"] == "spawned"]
    assert _sigterm_killpgs(events) == spawned
    _wait_members_gone(events)


def test_b_sigterm_during_callback_unregistration(tmp_path):
    hung, code, events, stderr = _run(tmp_path, "unregistration")
    _assert_clean_sigterm_exit(hung, code, events, stderr)
    assert "signal_inside_critical_section" in [e["e"] for e in events]
    # The interrupted unregister had not removed its callback; cleanup
    # callbacks are idempotent, so both run exactly once.
    assert sorted(_names(events)) == ["first", "second"]
    _wait_members_gone(events)


def test_c_sigterm_during_binding_release_keeps_persistent_state(
    tmp_path, isolated_openclaw_binding_root
):
    state, config = _persistent_state(tmp_path)
    before = _snapshot(state)
    hung, code, events, stderr = _run(
        tmp_path, "release", config=config, root=isolated_openclaw_binding_root
    )
    _assert_clean_sigterm_exit(hung, code, events, stderr)
    assert "signal_inside_critical_section" in [e["e"] for e in events]
    assert _names(events) == ["sentinel"]
    bound = [Path(e["dir"]) for e in events if e["e"] == "bound"]
    assert bound and not any(path.exists() for path in bound)
    assert (
        recon.reconcile_binding_artifacts(isolated_openclaw_binding_root)["artifacts"]
        == []
    )
    assert _snapshot(state) == before


# --------------------------------------------------------------------------
# D/E — repeated SIGTERM and callback failures
# --------------------------------------------------------------------------


def test_d_repeated_sigterm_during_cleanup_does_not_abort_it(tmp_path):
    hung, code, events, stderr = _run(tmp_path, "repeated")
    _assert_clean_sigterm_exit(hung, code, events, stderr)
    assert _names(events) == ["a", "b"]
    spawned = [e["pgid"] for e in events if e["e"] == "spawned"]
    assert _sigterm_killpgs(events) == spawned  # killed exactly once
    _wait_members_gone(events)


def test_d_repeated_external_sigterm_burst(tmp_path, isolated_openclaw_binding_root):
    state, config = _persistent_state(tmp_path)
    before = _snapshot(state)
    proc, events_path = _start(
        tmp_path, "parent_signal", config=config, root=isolated_openclaw_binding_root
    )
    deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
    while "ready" not in [e["e"] for e in _read_events(events_path)]:
        assert proc.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    for _ in range(5):
        if proc.poll() is not None:
            break
        proc.send_signal(signal.SIGTERM)  # the child's own pid only
        time.sleep(0.05)
    hung, code, events, stderr = _finish(proc, events_path)
    _assert_clean_sigterm_exit(hung, code, events, stderr)

    # F: every binding released, every managed group killed exactly once.
    assert _names(events) == ["slow"]
    bound = [Path(e["dir"]) for e in events if e["e"] == "bound"]
    assert len(bound) == 3 and not any(path.exists() for path in bound)
    spawned = sorted(e["pgid"] for e in events if e["e"] == "spawned")
    assert sorted(_sigterm_killpgs(events)) == spawned
    _wait_members_gone(events)
    # N/O: nothing left for reconciliation, persistent state untouched.
    assert (
        recon.reconcile_binding_artifacts(isolated_openclaw_binding_root)["artifacts"]
        == []
    )
    assert _snapshot(state) == before


def test_e_callback_exception_does_not_block_remaining_cleanup(tmp_path):
    hung, code, events, stderr = _run(tmp_path, "exception")
    _assert_clean_sigterm_exit(hung, code, events, stderr)
    assert _names(events) == ["after"]
    assert "synthetic callback failure" in stderr


# --------------------------------------------------------------------------
# G — concurrent mutation and late registration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("iteration", range(3))
def test_g_sigterm_under_concurrent_registry_churn(tmp_path, iteration):
    proc, events_path = _start(tmp_path, "concurrent")
    deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
    while "ready" not in [e["e"] for e in _read_events(events_path)]:
        assert proc.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    hung, code, events, stderr = _finish(proc, events_path)
    _assert_clean_sigterm_exit(hung, code, events, stderr)
    assert "sentinel" in _names(events)
    # The steadily registered group is killed exactly once; churned groups
    # may or may not be registered at the snapshot (they are reaped by the
    # child's cleanup in _finish either way).
    stable = next(e for e in events if e["e"] == "spawned")
    assert _sigterm_killpgs(events).count(stable["pgid"]) == 1
    _wait_members_gone([stable])


def test_g_callback_registered_during_handler_is_drained(tmp_path):
    hung, code, events, stderr = _run(tmp_path, "drain")
    _assert_clean_sigterm_exit(hung, code, events, stderr)
    assert _names(events) == ["first", "late"]


# --------------------------------------------------------------------------
# H — legitimate managed-child termination (real signals, owned groups only)
# --------------------------------------------------------------------------


def _owned_group_with_grandchild():
    proc = subprocess.Popen(
        ["sh", "-c", "sleep 30 & echo $!; read _unused"],
        start_new_session=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    grandchild = int(proc.stdout.readline())
    return proc, grandchild, _proc_start(grandchild)[1]


def _owned_only_killpg(owned):
    real_killpg = os.killpg
    calls = []

    def _killpg(pgid, sig):
        calls.append((pgid, sig))
        if pgid not in owned:
            raise AssertionError(f"unexpected killpg target {pgid!r}")
        real_killpg(pgid, sig)

    return _killpg, calls


def test_h_kill_process_group_terminates_owned_group_and_grandchild():
    proc, grandchild, start = _owned_group_with_grandchild()
    try:
        sl.register_process_group(proc.pid)
        assert sl._active_process_groups[proc.pid] is not None
        killpg, calls = _owned_only_killpg({proc.pid})
        with patch.object(sl.os, "killpg", side_effect=killpg):
            sl.kill_process_group(proc.pid)
        assert proc.wait(timeout=10) is not None
        assert calls[0] == (proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while _member_alive(grandchild, start) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _member_alive(grandchild, start)
        assert proc.pid not in sl._active_process_groups
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if _member_alive(grandchild, start):
            os.kill(grandchild, signal.SIGKILL)


def test_h_exited_leader_group_residue_is_still_terminated():
    proc, grandchild, start = _owned_group_with_grandchild()
    try:
        sl.register_process_group(proc.pid)
        proc.stdin.close()  # leader exits; grandchild keeps the group alive
        proc.wait(timeout=10)
        assert _member_alive(grandchild, start)
        killpg, calls = _owned_only_killpg({proc.pid})
        with patch.object(sl.os, "killpg", side_effect=killpg):
            sl.kill_process_group(proc.pid)
        deadline = time.monotonic() + 10
        while _member_alive(grandchild, start) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _member_alive(grandchild, start)
        assert calls[0] == (proc.pid, signal.SIGTERM)
    finally:
        if _member_alive(grandchild, start):
            os.kill(grandchild, signal.SIGKILL)


# --------------------------------------------------------------------------
# I/J/K/L/M — refusals (os.killpg is a recorder; nothing is ever signaled)
# --------------------------------------------------------------------------


@pytest.fixture
def killpg_recorder():
    calls = []

    def _never(pgid, sig):
        calls.append((pgid, sig))
        raise AssertionError(f"killpg must not be reached: {pgid!r}")

    with patch.object(sl.os, "killpg", side_effect=_never), patch.object(
        sl.time, "sleep"
    ):
        yield calls


@pytest.fixture
def owned_child():
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.parametrize("pgid", [0, 1, -1, -4242, True, 2.5, "4242", None])
def test_i_j_invalid_and_reserved_pgids_are_refused(pgid, killpg_recorder, caplog):
    sl.register_process_group(pgid)
    assert sl._active_process_groups == {}
    sl.kill_process_group(pgid)
    assert killpg_recorder == []
    assert "invalid_pgid" in caplog.text


@pytest.mark.parametrize("which", ["pid", "pgrp"])
def test_k_current_process_and_group_are_refused(which, killpg_recorder, caplog):
    target = os.getpid() if which == "pid" else os.getpgrp()
    if target <= 1:
        pytest.skip("test runner leads pgid <= 1; covered by J")
    # Even with a (forged) matching identity the own group is refused.
    sl._active_process_groups[target] = _proc_start(target)[1]
    sl.kill_process_group(target)
    with patch.object(sl.os, "kill"), patch.object(sl.signal, "signal"):
        sl._active_process_groups[target] = _proc_start(target)[1]
        sl._handle_forced_termination(signal.SIGTERM, None)
    assert killpg_recorder == []
    assert "own_process_group" in caplog.text


def test_l_unregistered_group_is_refused(owned_child, killpg_recorder, caplog):
    sl.kill_process_group(owned_child.pid)  # live group, but never registered
    assert killpg_recorder == []
    assert "unregistered_pgid" in caplog.text


def test_l_registered_group_not_owned_by_this_process_is_refused(
    killpg_recorder, caplog
):
    # A real group whose leader is a grandchild, not our direct child.
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess,sys;"
            "p=subprocess.Popen(['sleep','30'],start_new_session=True);"
            "print(p.pid,flush=True);sys.stdin.read();p.kill();p.wait()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        foreign = int(launcher.stdout.readline())
        sl.register_process_group(foreign)
        assert sl._active_process_groups[foreign] is None
        sl.kill_process_group(foreign)
        assert killpg_recorder == []
        assert "unverified_ownership" in caplog.text
    finally:
        launcher.stdin.close()
        launcher.wait(timeout=10)


def test_m_reused_pgid_is_refused_before_sigterm(owned_child, killpg_recorder, caplog):
    sl.register_process_group(owned_child.pid)
    # Simulate reuse: the live leader is not the process that was registered.
    sl._active_process_groups[owned_child.pid] -= 1
    sl.kill_process_group(owned_child.pid)
    assert killpg_recorder == []
    assert "pgid_reused" in caplog.text


def test_m_reuse_during_grace_period_blocks_sigkill(owned_child, caplog):
    sl.register_process_group(owned_child.pid)
    identity = sl._active_process_groups[owned_child.pid]
    calls = []
    stats = iter(
        [
            (os.getpid(), owned_child.pid, identity),  # before SIGTERM: owned
            (1, owned_child.pid, identity + 5),  # before SIGKILL: reused
        ]
    )
    with patch.object(
        sl.os, "killpg", side_effect=lambda p, s: calls.append((p, s))
    ), patch.object(sl, "_read_proc_stat", side_effect=lambda _pid: next(stats)):
        sl.kill_process_group(owned_child.pid, grace_period_seconds=0)
    assert calls == [(owned_child.pid, signal.SIGTERM)]  # recorder: no signal
    assert '"phase": "sigkill"' in caplog.text and "pgid_reused" in caplog.text


def test_m_stale_registration_is_dropped_without_escalation():
    proc = subprocess.Popen(
        ["sh", "-c", "read _unused"], stdin=subprocess.PIPE, start_new_session=True
    )
    sl.register_process_group(proc.pid)
    assert sl._active_process_groups[proc.pid] is not None
    proc.stdin.close()
    proc.wait(timeout=10)  # leader reaped, group empty: registration is stale
    calls = []

    def _gone(pgid, sig):
        calls.append((pgid, sig))
        raise ProcessLookupError()

    with patch.object(sl.os, "killpg", side_effect=_gone):
        sl.kill_process_group(proc.pid)
    assert calls == [(proc.pid, signal.SIGTERM)]  # recorder: no signal
    assert proc.pid not in sl._active_process_groups


# --------------------------------------------------------------------------
# The handler never holds the lock across user code
# --------------------------------------------------------------------------


def test_lock_is_released_before_cleanup_callbacks_and_reentrant():
    held_during_callback = []

    def _callback():
        # Another thread must be able to take the lock while callbacks run.
        import threading

        acquired = []
        thread = threading.Thread(
            target=lambda: acquired.append(sl._lock.acquire(timeout=2))
            or sl._lock.release()
        )
        thread.start()
        thread.join(timeout=5)
        held_during_callback.append(acquired == [True])

    sl.register_forced_termination_cleanup(_callback)
    with patch.object(sl.os, "kill"), patch.object(sl.signal, "signal"):
        sl._handle_forced_termination(signal.SIGTERM, None)
    assert held_during_callback == [True]
    # Reentrant: the main thread re-acquires while already holding it (the
    # pre-S1 threading.Lock would block forever here).
    with sl._lock:
        assert sl._lock.acquire(timeout=1)
        sl._lock.release()

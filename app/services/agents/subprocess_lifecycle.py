"""Process-group lifecycle tracking for OpenClaw CLI subprocesses.

Phase 23D-3: closes the forced-termination gap Phase 23D-2 flagged and left
open. `openclaw` CLI subprocesses are spawned in their own process group
(`start_new_session=True`); this module tracks the currently-running group(s)
so that a hard `SIGTERM` to this worker process -- e.g. via
`revoke_session_celery_tasks(terminate=True)` on the intervention/pause path
-- kills the whole group instead of orphaning the child, and runs the
dispatch's own runtime-workspace cleanup (ephemeral OpenClaw config release +
sandbox disposal) only after that kill, before the process actually exits.

Phase 37 Pre-B-F2B-S1 concurrency contract:

* `_lock` is an RLock. Critical sections contain only builtin container
  operations on these registries -- no user code, I/O or logging -- so a
  holder on another thread always releases it promptly.
* The SIGTERM handler always runs on the main thread, between bytecodes. If
  it interrupts the main thread inside a critical section, the reentrant
  acquire succeeds (the old non-reentrant Lock deadlocked here) and the
  registries are consistent, because no builtin container operation is ever
  split by a Python-level signal handler. The interrupted frame
  never resumes: the handler always ends by re-delivering SIGTERM under the
  default disposition.
* The handler never holds `_lock` while killing groups or running callbacks,
  so a callback that unregisters itself, or another thread registering,
  never waits on the handler.

Phase 37 Pre-B-F2B-S1 PGID safety: a registered number alone never
authorizes `killpg`. See `_termination_refusal`.
"""

import json
import logging
import os
import signal
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.RLock()
# pgid -> leader start time (clock ticks since boot, /proc/<pid>/stat field
# 22), or None when ownership could not be established at registration.
_active_process_groups: Dict[int, Optional[int]] = {}
_active_cleanup_callbacks: List[Callable[[], None]] = []
_handler_installed = False
_forced_termination_in_progress = False
# Bounds the handler's drain loop: entries registered by other threads while
# the handler runs (e.g. a CLI lock-contention retry) are picked up by the
# next round instead of being dropped.
_MAX_DRAIN_ROUNDS = 5
_UNREGISTERED = object()


def _read_proc_stat(pid: int) -> Optional[tuple]:
    """Return (ppid, pgrp, starttime) for `pid`, or None if it does not exist."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as stat_file:
            raw = stat_file.read().decode("ascii", "replace")
    except OSError:
        return None
    # comm (field 2) may contain spaces and parentheses; fields after the
    # last ')' start at field 3 (state).
    fields = raw[raw.rfind(")") + 2 :].split()
    try:
        return int(fields[1]), int(fields[2]), int(fields[19])
    except (IndexError, ValueError):
        return None


def _is_valid_pgid(pid) -> bool:
    return type(pid) is int and pid > 1


def register_process_group(pid: int) -> None:
    """Record a live OpenClaw CLI process group (pgid == pid, start_new_session=True).

    Ownership is recorded only when `pid` is currently a direct child of this
    process leading its own group; otherwise the entry is kept for tracking
    but `kill_process_group` will refuse to signal it.
    """
    if not _is_valid_pgid(pid):
        _log_refusal(pid, "invalid_pgid", phase="register")
        return
    identity = None
    stat = _read_proc_stat(pid)
    if stat is not None:
        ppid, pgrp, starttime = stat
        if ppid == os.getpid() and pgrp == pid:
            identity = starttime
    with _lock:
        _active_process_groups[pid] = identity


def unregister_process_group(pid: int) -> None:
    with _lock:
        _active_process_groups.pop(pid, None)


def register_forced_termination_cleanup(
    callback: Callable[[], None],
) -> Callable[[], None]:
    """Register `callback` to run on a forced SIGTERM; returns an unregister fn.

    The caller must invoke the returned unregister function from its own
    normal-path `finally` once the dispatch completes on its own, so the
    callback never fires after a dispatch that already cleaned up normally.
    """
    with _lock:
        _active_cleanup_callbacks.append(callback)

    def _unregister() -> None:
        try:
            with _lock:
                _active_cleanup_callbacks.remove(callback)
        except ValueError:
            pass  # already unregistered, or already taken by the handler

    return _unregister


def _log_refusal(pid, reason: str, *, phase: str) -> None:
    # Non-sensitive: only the requested number's repr and a fixed reason.
    logger.warning(
        "[SUBPROCESS_LIFECYCLE] process_group_signal_refused %s",
        json.dumps({"pgid": repr(pid)[:32], "reason": reason, "phase": phase}),
    )


def _termination_refusal(pid, identity: Optional[int]) -> Optional[str]:
    """Return why `pid` must not be signaled, or None if it may be.

    `identity` is the leader start time recorded at registration. A leader
    that still exists must be that same process (defeats PID reuse); a leader
    that has exited and been reaped leaves a group that can only still exist
    if members of the original group remain (see residual note in the S1
    report).
    """
    if not _is_valid_pgid(pid):
        return "invalid_pgid"
    if pid == os.getpid() or pid == os.getpgrp():
        return "own_process_group"
    if identity is None:
        return "unverified_ownership"
    stat = _read_proc_stat(pid)
    if stat is None:
        return None  # leader gone; group residue (if any) is ours
    _ppid, pgrp, starttime = stat
    if starttime != identity or pgrp != pid:
        return "pgid_reused"
    return None


def _kill_owned_group(
    pid: int, identity: Optional[int], grace_period_seconds: float
) -> None:
    refusal = _termination_refusal(pid, identity)
    if refusal is not None:
        _log_refusal(pid, refusal, phase="sigterm")
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    time.sleep(grace_period_seconds)
    # Re-verify: the leader may have exited and been reaped during the grace
    # period; never escalate against a reused number.
    refusal = _termination_refusal(pid, identity)
    if refusal is not None:
        _log_refusal(pid, refusal, phase="sigkill")
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def kill_process_group(pid: int, *, grace_period_seconds: float = 0.2) -> None:
    """Best-effort SIGTERM-then-SIGKILL of a registered, owned process group.

    Idempotent and never raises -- safe to call from a signal handler or from
    normal async cleanup code. Unregistered, unverified or invalid groups are
    refused with a warning and never signaled.
    """
    if not _is_valid_pgid(pid):
        _log_refusal(pid, "invalid_pgid", phase="sigterm")
        return
    with _lock:
        identity = _active_process_groups.pop(pid, _UNREGISTERED)
    if identity is _UNREGISTERED:
        _log_refusal(pid, "unregistered_pgid", phase="sigterm")
        return
    _kill_owned_group(pid, identity, grace_period_seconds)


def _handle_forced_termination(signum, frame) -> None:
    global _forced_termination_in_progress
    if _forced_termination_in_progress:
        # A repeated SIGTERM interrupted this handler. Let the outer
        # invocation finish every kill and callback; it re-delivers SIGTERM
        # under the default disposition when done.
        return
    _forced_termination_in_progress = True

    for _round in range(_MAX_DRAIN_ROUNDS):
        with _lock:
            groups = list(_active_process_groups.items())
            _active_process_groups.clear()
        with _lock:
            callbacks = list(_active_cleanup_callbacks)
            _active_cleanup_callbacks.clear()
        if not groups and not callbacks:
            break

        for pid, identity in groups:
            try:
                _kill_owned_group(pid, identity, 0.2)
            except Exception:
                logger.exception(
                    "[SUBPROCESS_LIFECYCLE] Failed killing process group %s on SIGTERM",
                    pid,
                )

        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception(
                    "[SUBPROCESS_LIFECYCLE] Forced-termination cleanup callback failed"
                )

    # Restore default disposition and re-deliver so this worker process still
    # terminates exactly as `revoke(terminate=True, signal='SIGTERM')` expects
    # -- Celery's task_acks_late/reject_on_worker_lost requeue behavior is
    # unchanged, we have only ensured the child process group and the
    # runtime-workspace binding/sandbox are torn down first.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)


def install_sigterm_handler() -> None:
    """Install the forced-termination handler once per worker process."""
    global _handler_installed
    if _handler_installed:
        return
    signal.signal(signal.SIGTERM, _handle_forced_termination)
    _handler_installed = True


def _reset_for_tests() -> None:
    """Test-only: clear all module state between test cases."""
    global _handler_installed, _forced_termination_in_progress
    with _lock:
        _active_process_groups.clear()
        _active_cleanup_callbacks.clear()
    _handler_installed = False
    _forced_termination_in_progress = False

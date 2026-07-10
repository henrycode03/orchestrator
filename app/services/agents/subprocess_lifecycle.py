"""Process-group lifecycle tracking for OpenClaw CLI subprocesses.

Phase 23D-3: closes the forced-termination gap Phase 23D-2 flagged and left
open. `openclaw` CLI subprocesses are spawned in their own process group
(`start_new_session=True`); this module tracks the currently-running group(s)
so that a hard `SIGTERM` to this worker process -- e.g. via
`revoke_session_celery_tasks(terminate=True)` on the intervention/pause path
-- kills the whole group instead of orphaning the child, and runs the
dispatch's own runtime-workspace cleanup (ephemeral OpenClaw config release +
sandbox disposal) only after that kill, before the process actually exits.
"""

import logging
import os
import signal
import threading
import time
from typing import Callable, List, Set

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_active_process_groups: Set[int] = set()
_active_cleanup_callbacks: List[Callable[[], None]] = []
_handler_installed = False


def register_process_group(pid: int) -> None:
    """Record a live OpenClaw CLI process group (pgid == pid, start_new_session=True)."""
    with _lock:
        _active_process_groups.add(pid)


def unregister_process_group(pid: int) -> None:
    with _lock:
        _active_process_groups.discard(pid)


def register_forced_termination_cleanup(
    callback: Callable[[], None],
) -> Callable[[], None]:
    """Register cleanup to run if this worker process is SIGTERM'd mid-dispatch.

    Returns an unregister function the caller must invoke from its own
    normal-path `finally` once the dispatch completes on its own, so the
    callback never fires after a dispatch that already cleaned up normally.
    """
    with _lock:
        _active_cleanup_callbacks.append(callback)

    def _unregister() -> None:
        with _lock:
            if callback in _active_cleanup_callbacks:
                _active_cleanup_callbacks.remove(callback)

    return _unregister


def kill_process_group(pid: int, *, grace_period_seconds: float = 0.2) -> None:
    """Best-effort SIGTERM-then-SIGKILL of an entire process group.

    Idempotent and never raises -- safe to call from a signal handler or from
    normal async cleanup code.
    """
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        unregister_process_group(pid)
        return
    time.sleep(grace_period_seconds)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    unregister_process_group(pid)


def _handle_forced_termination(signum, frame) -> None:
    with _lock:
        pids = list(_active_process_groups)
        callbacks = list(_active_cleanup_callbacks)
        _active_cleanup_callbacks.clear()
        _active_process_groups.clear()

    for pid in pids:
        try:
            kill_process_group(pid)
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
    global _handler_installed
    with _lock:
        _active_process_groups.clear()
        _active_cleanup_callbacks.clear()
    _handler_installed = False

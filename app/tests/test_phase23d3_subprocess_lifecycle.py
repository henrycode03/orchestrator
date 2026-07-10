"""Phase 23D-3 — OpenClaw Subprocess Lifecycle Containment.

Phase 23D-2 found and flagged, not fixed: forcibly terminating a running
dispatch (SIGTERM to the Celery worker OS process via
``revoke_session_celery_tasks(terminate=True)``) can orphan the ``openclaw``
CLI child process the worker had already spawned; if the parent's cleanup
(ephemeral OpenClaw config release + sandbox disposal) races ahead of that
orphan, it can fall back to the real persistent ``openclaw.json`` and write
scaffold files into the real Project Workspace.

These tests pin the fix at the unit level: subprocesses are tracked as
process groups, a forced-termination signal kills every tracked group and
runs registered cleanup before the worker process actually exits, and the
registration/unregistration contract is idempotent.
"""

import os
import signal
import time
from unittest.mock import patch

import pytest

from app.services.agents import subprocess_lifecycle as sl


@pytest.fixture(autouse=True)
def _reset_module_state():
    sl._reset_for_tests()
    yield
    sl._reset_for_tests()


class TestProcessGroupRegistry:
    def test_register_and_unregister_round_trip(self):
        sl.register_process_group(4242)
        assert 4242 in sl._active_process_groups
        sl.unregister_process_group(4242)
        assert 4242 not in sl._active_process_groups

    def test_unregister_unknown_pid_is_a_noop(self):
        sl.unregister_process_group(999999)  # never raises

    def test_kill_process_group_sends_term_then_kill(self):
        calls = []

        def _fake_killpg(pid, sig):
            calls.append((pid, sig))
            if sig == signal.SIGTERM:
                return
            raise ProcessLookupError()

        sl.register_process_group(123)
        with patch.object(sl.os, "killpg", side_effect=_fake_killpg), patch.object(
            sl.time, "sleep"
        ) as mock_sleep:
            sl.kill_process_group(123)

        assert calls == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
        mock_sleep.assert_called_once()
        assert 123 not in sl._active_process_groups

    def test_kill_process_group_already_dead_is_idempotent(self):
        sl.register_process_group(123)
        with patch.object(sl.os, "killpg", side_effect=ProcessLookupError()):
            sl.kill_process_group(123)  # never raises
        assert 123 not in sl._active_process_groups

    def test_kill_process_group_permission_denied_never_raises(self):
        sl.register_process_group(123)
        with patch.object(sl.os, "killpg", side_effect=PermissionError()):
            sl.kill_process_group(123)


class TestForcedTerminationCleanupRegistration:
    def test_registered_callback_fires_on_forced_termination(self):
        fired = []
        unregister = sl.register_forced_termination_cleanup(lambda: fired.append(True))
        assert unregister is not None

        with patch.object(sl.os, "killpg"), patch.object(sl.os, "kill"), patch.object(
            sl.signal, "signal"
        ):
            sl._handle_forced_termination(signal.SIGTERM, None)

        assert fired == [True]

    def test_unregister_prevents_callback_from_firing(self):
        fired = []
        unregister = sl.register_forced_termination_cleanup(lambda: fired.append(True))
        unregister()

        with patch.object(sl.os, "killpg"), patch.object(sl.os, "kill"), patch.object(
            sl.signal, "signal"
        ):
            sl._handle_forced_termination(signal.SIGTERM, None)

        assert fired == []

    def test_callback_exception_does_not_block_process_termination(self):
        def _boom():
            raise RuntimeError("cleanup exploded")

        sl.register_forced_termination_cleanup(_boom)

        killed_self = []

        def _fake_kill(pid, sig):
            killed_self.append((pid, sig))

        with patch.object(sl.os, "killpg"), patch.object(
            sl.os, "kill", side_effect=_fake_kill
        ), patch.object(sl.signal, "signal") as mock_signal:
            sl._handle_forced_termination(signal.SIGTERM, None)

        # Default disposition restored and SIGTERM re-delivered to self even
        # though the cleanup callback raised.
        mock_signal.assert_called_once_with(signal.SIGTERM, signal.SIG_DFL)
        assert killed_self == [(os.getpid(), signal.SIGTERM)]


class TestForcedTerminationKillsProcessGroupsBeforeCleanup:
    def test_active_process_groups_killed_before_cleanup_callbacks(self):
        order = []
        sl.register_process_group(555)
        sl.register_forced_termination_cleanup(lambda: order.append("cleanup"))

        def _fake_killpg(pid, sig):
            if sig == signal.SIGTERM:
                order.append(f"kill:{pid}")
            else:
                raise ProcessLookupError()

        with patch.object(sl.os, "killpg", side_effect=_fake_killpg), patch.object(
            sl.time, "sleep"
        ), patch.object(sl.os, "kill"), patch.object(sl.signal, "signal"):
            sl._handle_forced_termination(signal.SIGTERM, None)

        assert order == ["kill:555", "cleanup"]

    def test_registry_cleared_after_forced_termination(self):
        sl.register_process_group(1)
        sl.register_process_group(2)
        sl.register_forced_termination_cleanup(lambda: None)

        with patch.object(sl.os, "killpg"), patch.object(sl.os, "kill"), patch.object(
            sl.signal, "signal"
        ):
            sl._handle_forced_termination(signal.SIGTERM, None)

        assert sl._active_process_groups == set()
        assert sl._active_cleanup_callbacks == []


class TestInstallSigtermHandler:
    def test_install_is_idempotent(self):
        with patch.object(sl.signal, "signal") as mock_signal:
            sl.install_sigterm_handler()
            sl.install_sigterm_handler()

        mock_signal.assert_called_once()

    def test_install_registers_handler_for_sigterm(self):
        with patch.object(sl.signal, "signal") as mock_signal:
            sl.install_sigterm_handler()

        args, _ = mock_signal.call_args
        assert args[0] == signal.SIGTERM
        assert args[1] is sl._handle_forced_termination


class TestWorkerDispatchRegistersForcedTerminationCleanup:
    """Regression guard: the canonical dispatch task must register its
    runtime-workspace cleanup as a forced-termination callback, and must
    unregister it in its own normal-path `finally` before doing its own
    disposal -- otherwise a forced SIGTERM after normal completion (or a
    normal completion after a registered-but-stale callback) could double-run
    disposal, or the callback could go missing entirely on a refactor."""

    def test_worker_registers_and_unregisters_cleanup_closure(self):
        import inspect

        import app.tasks.worker as worker

        source = inspect.getsource(worker.execute_orchestration_task)
        assert "register_forced_termination_cleanup(" in source
        assert "_unregister_forced_termination_cleanup()" in source
        # Unregistration must be the first statement in the outer `finally`,
        # ahead of the existing release_runtime_workspace_binding/dispose
        # calls, so a dispatch that completes normally can never also be
        # cleaned up a second time via the SIGTERM path.
        finally_index = source.index("\n    finally:")
        unregister_call_index = source.index(
            "_unregister_forced_termination_cleanup()", finally_index
        )
        # release_runtime_workspace_binding() appears twice: once inside the
        # forced-termination closure (defined before `try:`), once in the
        # normal-path `finally` -- the second occurrence is the one that must
        # come after unregistration.
        release_index = source.index(
            "release_runtime_workspace_binding()", unregister_call_index
        )
        assert finally_index < unregister_call_index < release_index

    def test_openclaw_service_spawns_subprocess_in_new_session(self):
        import inspect

        from app.services.agents import openclaw_service

        source = inspect.getsource(openclaw_service)
        assert source.count("start_new_session=True") == 2
        # No remaining bare (non-group) process.kill() calls on the real
        # asyncio subprocess paths -- all replaced with kill_process_group so
        # a forced kill takes the whole process group. `self.process.kill()`
        # is unrelated dead legacy code (`self.process` is never assigned to
        # a real subprocess anywhere in this file).
        bare_kills = [
            line
            for line in source.splitlines()
            if "process.kill()" in line and "self.process.kill()" not in line
        ]
        assert bare_kills == []

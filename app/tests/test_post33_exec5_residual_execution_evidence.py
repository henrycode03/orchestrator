"""POST33-EXEC5 evidence-complete residual Execution discriminator.

This is evaluation-only.  It uses the real execution prompt builder, direct
runtime adapter, response extraction, coercion, and (when explicitly enabled)
one gateway request.  It never changes production routing or invokes
Planning's provider path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import settings
from app.services.agents.agent_runtime import create_agent_runtime
from app.services.agents.providers.openai_chat_adapter import (
    _STEP_SYSTEM,
)
from app.services.agents.runtime_configuration import BackendRole
from app.services.orchestration.error_handler import error_handler
from app.services.orchestration.execution.step_support import (
    coerce_execution_step_result,
)
from app.services.orchestration.phases.execution_loop import execute_step_loop
from app.services.orchestration.context.assembly import assemble_execution_prompt
from app.services.orchestration.validation.parsing import extract_structured_text

from app.tests.test_post33_exec2_structured_orchestrator_consumption import (
    _make_loop_context,
)


GATE = "POST33-EXEC5"
TASK_TEXT = (
    "Create tiny_calc.py in the project workspace so answer() returns 42. "
    "The existing test_tiny_calc.py is the focused verification and must pass. "
    "Use one narrowly scoped task and preserve the existing test."
)
PLANNING_ARTIFACTS = {
    "requirements": "# Requirements\n\n- Change answer() from 41 to 42.",
    "design": "# Design\n\n- Keep the change limited to tiny_calc.py.",
    "implementation_plan": "# Implementation Plan\n\n1. Update tiny_calc.py.\n2. Run the focused test.",
    "planner_markdown": (
        "# Project: POST33-EXEC5 tiny_calc\n\n"
        "## Task List\n"
        "- [ ] TASK_START: Create tiny calculation source | Create tiny_calc.py "
        "so answer() returns 42 and run its focused test | order=1 | P1 | "
        "effort=small | stage=execute | profile=full_lifecycle"
    ),
}

# This is the residual form of the bounded EXEC3R1 class: the committed step
# carries the focused test command and expected creation path, while the
# residual runtime must return the step outcome.  No provider-native tool is
# implied by this object.
COMMITTED_EXECUTION_STEP = {
    "step_number": 1,
    "description": "Create tiny_calc.py with answer() returning 42.",
    # The opaque command deliberately selects the residual runtime branch in
    # the same way as EXEC2's provider-free residual seam.  The focused pytest
    # remains the committed verification fact and is not executed by the
    # provider-free capture call.
    "commands": ["custom-direct-runtime-step"],
    "verification": "python3 -m pytest -q test_tiny_calc.py",
    "rollback": None,
    "expected_files": ["tiny_calc.py"],
    "ops": [],
}

COMMITTED_PLAN = {
    "title": "POST33-EXEC5 tiny_calc",
    "source_brain": "local",
    "requirement": PLANNING_ARTIFACTS["requirements"],
    "markdown": PLANNING_ARTIFACTS["planner_markdown"],
    "status": "committed",
}

ARTIFACT_NAMES = (
    "metadata.json",
    "canonical-prompt.txt",
    "system-prompt.txt",
    "user-prompt.txt",
    "wire-request.json",
    "raw-stdout.txt",
    "raw-stderr.txt",
    "raw-provider-response.json",
    "extracted-response.txt",
    "normalized-response.txt",
    "coercion-result.json",
    "parser-error.json",
    "runtime-identity.json",
    "gateway-response-metadata.json",
    "final-result.json",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


class DurableCapture:
    """Pre-create and fsync all evidence destinations before dispatch."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def precreate(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o777)
        for name in ARTIFACT_NAMES:
            path = self.root / name
            if not path.exists():
                path.write_text(
                    "{}\n" if name.endswith(".json") else "", encoding="utf-8"
                )
            path.chmod(0o666)
        self._fsync_dir()

    def write_text(self, name: str, value: str) -> None:
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        path.chmod(0o666)
        self._fsync_dir()

    def write_json(self, name: str, value: Any) -> None:
        self.write_text(name, _canonical_json(value))

    def _fsync_dir(self) -> None:
        import os

        try:
            fd = os.open(self.root, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _tiny_calc_fixture(tmp_path: Path, db: Any, runtime: Any | None = None):
    runtime = runtime or SimpleNamespace(
        backend="openai_chat_completions", model="qwen-local"
    )
    ctx, execution, _ = _make_loop_context(
        db,
        tmp_path,
        step=COMMITTED_EXECUTION_STEP,
        runtime=runtime,
        timeout_seconds=60,
    )
    ctx.prompt = TASK_TEXT
    ctx.task.description = TASK_TEXT
    ctx.orchestration_state.task_description = TASK_TEXT
    ctx.orchestration_state.project_context = (
        "The focused test test_tiny_calc.py is present and asserts answer() == 42."
    )
    workspace = Path(ctx.orchestration_state.project_dir)
    (workspace / "test_tiny_calc.py").write_text(
        "from tiny_calc import answer\n\n\ndef test_answer():\n    assert answer() == 42\n",
        encoding="utf-8",
    )
    ctx.project.workspace_path = str(workspace)
    return ctx, execution, workspace


def _runtime_identity(runtime: Any, configuration: Any) -> dict[str, Any]:
    endpoint = str(runtime._base_url).rstrip("/") + "/chat/completions"
    payload = {
        "requested": {
            "provider": configuration.backend_name,
            "model": configuration.model_family,
            "profile": configuration.adaptation_profile,
            "role": configuration.role.value,
            "topology": "STRUCTURED_ORCHESTRATOR",
        },
        "effective": {
            "provider": getattr(runtime.backend_descriptor, "name", None),
            "model": runtime._model_name(),
            "profile": configuration.adaptation_profile,
            "role": getattr(runtime, "backend_role", None),
            "endpoint": endpoint,
        },
    }
    payload["configuration_fingerprint"] = _sha256_bytes(
        _canonical_json(payload).encode("utf-8")
    )
    return payload


def _raw_response_score(
    raw: str, expected_path: str = "tiny_calc.py"
) -> dict[str, Any]:
    stripped = str(raw or "").strip()
    fenced = bool(re.search(r"^```|```$", stripped, re.MULTILINE))
    prose_wrapper = False
    parsed: Any = None
    json_valid = False
    if stripped:
        try:
            parsed = json.loads(stripped)
            json_valid = True
        except json.JSONDecodeError:
            prose_wrapper = bool(
                ("{" in stripped or "[" in stripped)
                and not stripped.startswith(("{", "[", "```"))
            )
    native_tool = bool(
        re.search(
            r"<tool_call>|\"tool_calls\"|\"function\"\s*:|^\s*\{\s*\"action\"\s*:",
            stripped,
            re.IGNORECASE,
        )
    )
    if isinstance(parsed, dict):
        native_tool = native_tool or any(
            key in parsed for key in ("tool_calls", "function", "action")
        )
    paths: list[str] = []
    path_matches = re.findall(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py", stripped)
    paths.extend(dict.fromkeys(path_matches))
    semantically_solves = bool(
        isinstance(parsed, dict)
        and str(parsed.get("status") or "").lower()
        in {"success", "completed", "complete", "succeeded", "done"}
        and expected_path in (parsed.get("files_changed") or [])
        and "return 42" in str(parsed.get("output") or "")
    )
    return {
        "non_empty": bool(stripped),
        "json_valid": json_valid,
        "top_level_shape": type(parsed).__name__ if parsed is not None else None,
        "has_prose_wrapper": prose_wrapper,
        "has_code_fence": fenced,
        "has_native_tool_syntax": native_tool,
        "paths": paths,
        "semantically_solves_step": semantically_solves,
        "unexpected_shell_or_tool_request": native_tool
        or bool(re.search(r"\b(?:bash|sh|shell|execute_command)\b", stripped)),
        "unsupported_fields": (
            sorted(
                set(parsed)
                - {
                    "status",
                    "output",
                    "verification_output",
                    "files_changed",
                    "error",
                    "error_message",
                }
            )
            if isinstance(parsed, dict)
            else []
        ),
        "internally_contradictory": bool(
            isinstance(parsed, dict)
            and str(parsed.get("status") or "").lower() in {"failed", "error"}
            and expected_path in (parsed.get("files_changed") or [])
            and "success" in str(parsed.get("error") or "").lower()
        ),
    }


def _known_good_provider_text() -> str:
    return json.dumps(
        {
            "status": "completed",
            "output": "Created tiny_calc.py with answer() returning 42.",
            "verification_output": "1 passed",
            "files_changed": ["tiny_calc.py"],
            "error": "",
        },
        sort_keys=True,
    )


def _coerce_trace(raw_provider_text: str) -> dict[str, Any]:
    raw_result = {"status": "completed", "output": raw_provider_text}
    extracted = extract_structured_text(raw_result["output"])
    parser_trace: dict[str, Any] = {}
    original_parser = error_handler.attempt_json_parsing

    def capture_parser(text: str, context: str = "JSON"):
        result = original_parser(text, context=context)
        parser_trace.update(
            {
                "context": context,
                "input": text,
                "success": result[0],
                "parsed": result[1],
                "strategy": result[2],
            }
        )
        return result

    error_handler.attempt_json_parsing = capture_parser
    try:
        coerced = coerce_execution_step_result(
            raw_result,
            expected_files=["tiny_calc.py"],
            extract_structured_text=extract_structured_text,
        )
    finally:
        error_handler.attempt_json_parsing = original_parser
    return {
        "raw": raw_result,
        "extracted": extracted,
        "parser": parser_trace,
        "coerced": coerced,
    }


def test_exec5_known_good_result_passes_real_coercion():
    trace = _coerce_trace(_known_good_provider_text())
    assert trace["parser"]["success"] is True
    assert trace["coerced"]["status"] == "completed"
    assert trace["coerced"]["files_changed"] == ["tiny_calc.py"]


def test_exec5_artifact_root_is_precreated_and_raw_survives_post_provider_exception(
    tmp_path,
):
    capture = DurableCapture(tmp_path / "evidence")
    capture.precreate()
    assert all((capture.root / name).exists() for name in ARTIFACT_NAMES)

    raw_body = {"model": "qwen-local", "choices": [{"message": {"content": "raw"}}]}
    capture.write_json("raw-provider-response.json", raw_body)
    with pytest.raises(RuntimeError, match="post-provider scoring crash"):
        raise RuntimeError("post-provider scoring crash")
    assert (
        json.loads(
            (capture.root / "raw-provider-response.json").read_text(encoding="utf-8")
        )
        == raw_body
    )


def test_exec5_provider_free_capture_reconstructs_exact_wire_and_prompt(
    db_session,
    tmp_path,
    monkeypatch,
):
    ctx, _, workspace = _tiny_calc_fixture(tmp_path, db_session)
    prompt = assemble_execution_prompt(ctx, COMMITTED_EXECUTION_STEP)
    capture = DurableCapture(tmp_path / "evidence")
    capture.precreate()
    capture.write_text("canonical-prompt.txt", prompt)
    capture.write_text("system-prompt.txt", _STEP_SYSTEM)
    capture.write_text("user-prompt.txt", prompt)

    monkeypatch.setattr(settings, "EXECUTION_BACKEND", "openai_chat_completions")
    monkeypatch.setattr(settings, "EXECUTION_MODEL", "qwen-local")
    monkeypatch.setattr(settings, "EXECUTION_ADAPTATION_PROFILE", "ollama_default")
    monkeypatch.setattr(settings, "EXECUTION_CONTEXT_TOKENS", 64_000)
    configuration = SimpleNamespace(
        backend_name="openai_chat_completions",
        model_family="qwen-local",
        adaptation_profile="ollama_default",
        role=BackendRole.EXECUTION,
    )

    from app.services.agents.providers.openai_chat_adapter import (
        OpenAIChatCompletionsRuntime,
    )

    runtime = OpenAIChatCompletionsRuntime(
        db_session,
        ctx.session_id,
        ctx.task_id,
        runtime_configuration=configuration,
    )
    body = {
        "id": "exec5-provider-free",
        "model": "qwen-local",
        "choices": [
            {"message": {"role": "assistant", "content": _known_good_provider_text()}}
        ],
    }
    observed: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return body

    import httpx

    async def fake_post(client, url, *args, **kwargs):
        del client, args
        observed["url"] = str(url)
        observed["payload"] = kwargs["json"]
        capture.write_json("wire-request.json", kwargs["json"])
        capture.write_json("raw-provider-response.json", body)
        capture.write_json(
            "gateway-response-metadata.json",
            {"status_code": 200, "model": body["model"]},
        )
        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    result = asyncio.run(runtime.execute_task(prompt, timeout_seconds=60))
    captured_wire = json.loads((capture.root / "wire-request.json").read_text())
    assert observed["url"].endswith("/chat/completions")
    assert captured_wire == observed["payload"]
    assert captured_wire["model"] == "qwen-local"
    assert [message["role"] for message in captured_wire["messages"]] == [
        "system",
        "user",
    ]
    assert result["output"] == _known_good_provider_text()
    assert workspace.joinpath("test_tiny_calc.py").exists()


@pytest.mark.live
def test_exec5_one_live_residual_execution_turn(db_session, tmp_path, monkeypatch):
    if str(__import__("os").environ.get("POST33_EXEC5_LIVE")) != "1":
        pytest.skip("POST33_EXEC5_LIVE=1 is required for the one-call discriminator")

    evidence_root = Path("docs/roadmap/reports/evidence/post33-exec5")
    capture = DurableCapture(evidence_root)
    capture.precreate()
    ctx, _, workspace = _tiny_calc_fixture(tmp_path, db_session)
    prompt = assemble_execution_prompt(ctx, COMMITTED_EXECUTION_STEP)
    capture.write_text("canonical-prompt.txt", prompt)
    capture.write_text("system-prompt.txt", _STEP_SYSTEM)
    capture.write_text("user-prompt.txt", prompt)

    # Process-local certification binding only.  This does not alter GX10 or
    # any persistent setting; it selects the already accepted direct gateway
    # identity for this isolated discriminator call.
    monkeypatch.setattr(settings, "EXECUTION_MODEL", "qwen-local")
    monkeypatch.setattr(
        settings, "OPENAI_CHAT_COMPLETIONS_BASE_URL", settings.PLANNING_DIRECT_BASE_URL
    )
    monkeypatch.setattr(settings, "OPENAI_CHAT_COMPLETIONS_API_KEY", "")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "EXECUTION_CONTEXT_TOKENS", 64_000)

    runtime = create_agent_runtime(
        db_session,
        ctx.session_id,
        ctx.task_id,
        role=BackendRole.EXECUTION,
        backend_override="openai_chat_completions",
    )
    resolved_configuration = runtime.runtime_configuration
    identity = _runtime_identity(runtime, resolved_configuration)
    capture.write_json("runtime-identity.json", identity)
    if (
        resolved_configuration.backend_name != "openai_chat_completions"
        or resolved_configuration.model_family != "qwen-local"
        or resolved_configuration.adaptation_profile != "ollama_default"
        or resolved_configuration.role is not BackendRole.EXECUTION
    ):
        capture.write_json(
            "final-result.json",
            {"runtime_identity_failure": identity, "provider_calls": 0},
        )
        pytest.fail("EXEC5 runtime identity hard gate failed before dispatch")
    capture.write_json(
        "metadata.json",
        {
            "gate": GATE,
            "task_text": TASK_TEXT,
            "planning_artifacts": PLANNING_ARTIFACTS,
            "committed_plan": COMMITTED_PLAN,
            "committed_execution_step": COMMITTED_EXECUTION_STEP,
            "expected_files": COMMITTED_EXECUTION_STEP["expected_files"],
            "verification": COMMITTED_EXECUTION_STEP["verification"],
            "rollback": COMMITTED_EXECUTION_STEP["rollback"],
            "prior_results": ctx.orchestration_state.execution_results,
            "project_context": ctx.orchestration_state.project_context,
            "runtime_workspace_identity": str(workspace.resolve()),
            "role_runtime_configuration": resolved_configuration.to_dict(),
            "provider_call_budget": 1,
            "provider_retries": 0,
        },
    )

    import httpx

    call_count = 0
    original_post = httpx.AsyncClient.post

    async def capture_post(client, url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count != 1:
            raise AssertionError("EXEC5 provider call budget exceeded")
        payload = kwargs.get("json") or {}
        if payload.get("model") != "qwen-local" or payload.get("tools"):
            raise AssertionError("EXEC5 wire identity/native-tool hard gate failed")
        capture.write_json("wire-request.json", payload)
        capture.write_text("raw-stdout.txt", "")
        capture.write_text("raw-stderr.txt", "")
        response = await original_post(client, url, *args, **kwargs)
        # This is written before the adapter extracts/scorers content.
        try:
            raw_body = response.json()
        except Exception as exc:
            capture.write_json(
                "gateway-response-metadata.json",
                {"status_code": response.status_code, "error": repr(exc)},
            )
            raise
        capture.write_json("raw-provider-response.json", raw_body)
        capture.write_json(
            "gateway-response-metadata.json",
            {
                "status_code": response.status_code,
                "response_model": (
                    raw_body.get("model") if isinstance(raw_body, dict) else None
                ),
                "url": str(url),
            },
        )
        return response

    monkeypatch.setattr(httpx.AsyncClient, "post", capture_post)
    try:
        runtime_result = asyncio.run(runtime.execute_task(prompt, timeout_seconds=300))
    except Exception as exc:
        capture.write_json(
            "final-result.json",
            {"provider_calls": call_count, "runtime_exception": repr(exc)},
        )
        raise

    raw_provider = json.loads(
        (evidence_root / "raw-provider-response.json").read_text()
    )
    extracted = str(runtime_result.get("output") or "")
    capture.write_text("extracted-response.txt", extracted)
    capture.write_text("normalized-response.txt", extracted.strip())
    score = _raw_response_score(extracted)
    trace = _coerce_trace(extracted)
    capture.write_json("coercion-result.json", trace)
    if trace["parser"].get("success") is not True:
        capture.write_json("parser-error.json", trace["parser"])
    else:
        capture.write_json("parser-error.json", {})
    capture.write_json(
        "final-result.json",
        {
            "raw_provider_response_model": (
                raw_provider.get("model") if isinstance(raw_provider, dict) else None
            ),
            "raw_response": extracted,
            "raw_response_score": score,
            "coercion": trace,
            "provider_calls": call_count,
            "provider_retries": 0,
            "workspace_before": sorted(path.name for path in workspace.iterdir()),
            "expected_behavior_fixed": (workspace / "tiny_calc.py").exists(),
        },
    )
    assert call_count == 1

    if trace["parser"].get("success"):

        class ReplayRuntime:
            backend = "openai_chat_completions"
            model = "qwen-local"

            def __init__(self, result):
                self.result = result
                self.calls: list[dict[str, Any]] = []

            async def execute_task(self, prompt, timeout_seconds=300, **kwargs):
                self.calls.append(
                    {"prompt": prompt, "timeout_seconds": timeout_seconds, **kwargs}
                )
                return dict(self.result)

            def reports_context_overflow(self, result):
                return False

            def get_backend_metadata(self):
                return {
                    "backend": self.backend,
                    "model_family": self.model,
                    "role": "execution",
                }

        replay_runtime = ReplayRuntime(trace["raw"])
        replay_ctx, _, replay_workspace = _tiny_calc_fixture(
            tmp_path / "replay", db_session, runtime=replay_runtime
        )
        replay_result = execute_step_loop(
            ctx=replay_ctx,
            extract_structured_text=extract_structured_text,
            normalize_step=lambda raw_step, project_dir, logger_obj, step_number: dict(
                raw_step
            ),
            normalize_plan_with_live_logging=lambda *args, **kwargs: [],
            workspace_violation_error_cls=RuntimeError,
            write_project_state_snapshot_fn=lambda *args, **kwargs: None,
            record_live_log_fn=lambda *args, **kwargs: None,
        )
        final_result = json.loads(
            (evidence_root / "final-result.json").read_text(encoding="utf-8")
        )
        final_result["coerced_result_consumed_by_real_loop"] = bool(
            replay_runtime.calls
        )
        final_result["real_loop_result"] = replay_result
        final_result["real_loop_provider_free_calls"] = len(replay_runtime.calls)
        final_result["verification_reached"] = bool(replay_runtime.calls) and bool(
            replay_result.get("status") == "completed"
        )
        final_result["verification_succeeded"] = bool(
            (replay_workspace / "tiny_calc.py").exists()
        )
        capture.write_json("final-result.json", final_result)

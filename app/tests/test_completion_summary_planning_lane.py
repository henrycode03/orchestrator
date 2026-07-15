"""Completion-summary repair-lane registry compatibility tests."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.orchestration.phases.completion_summary import (
    _call_planning_lane,
    _deterministic_task_summary,
    _generate_task_summary_with_fallback,
)


def _make_ctx(state: Any = None) -> MagicMock:
    ctx = MagicMock()
    ctx.db = MagicMock()
    ctx.orchestration_state = state or MagicMock(execution_results=[], plan=[])
    ctx.emit_live = MagicMock()
    return ctx


def _run_lane(output: str):
    runtime = MagicMock()
    runtime.invoke_prompt = AsyncMock(
        return_value={"status": "completed", "output": output}
    )
    with patch(
        "app.services.agents.agent_runtime.create_agent_runtime",
        return_value=runtime,
    ) as factory:
        result = asyncio.run(_call_planning_lane("summary prompt", db=MagicMock()))
    return result, factory, runtime


class TestCallPlanningLane:
    def test_returns_adapter_output_and_explicit_repair_role(self):
        result, factory, runtime = _run_lane("LLM summary text")

        assert result == "LLM summary text"
        assert factory.call_args.kwargs["role"].value == "repair"
        options = runtime.invoke_prompt.call_args.kwargs["invocation_options"]
        assert options.max_output_tokens == 512
        assert options.temperature == 0.0
        assert options.reasoning_enabled is False

    def test_empty_adapter_output_is_preserved_for_outer_fallback(self):
        result, _, _ = _run_lane("")
        assert result == ""

    def test_adapter_list_content_is_already_extracted(self):
        result, _, _ = _run_lane("part1part2")
        assert result == "part1part2"


class TestGenerateWithFallback:
    def _patch_planning_lane(self, output: str):
        return patch(
            "app.services.orchestration.phases.completion_summary._call_planning_lane",
            new=AsyncMock(return_value=output),
        )

    def test_flag_off_returns_deterministic(self):
        ctx = _make_ctx()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY", None)
            result = _generate_task_summary_with_fallback(ctx=ctx, summary_prompt="p")
        assert result["fallback"] is True
        assert "Task completed" in result["output"]

    def test_flag_on_calls_repair_registry_not_execution_runtime(self):
        ctx = _make_ctx()
        ctx.runtime_service = MagicMock()

        with patch.dict(os.environ, {"ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY": "1"}):
            with self._patch_planning_lane("LLM generated summary"):
                result = _generate_task_summary_with_fallback(
                    ctx=ctx, summary_prompt="prompt"
                )

        ctx.runtime_service.execute_task.assert_not_called()
        assert result["output"] == "LLM generated summary"
        assert result.get("fallback") is not True

    def test_planning_lane_exception_triggers_fallback(self):
        ctx = _make_ctx()

        async def _fail(prompt, **kwargs):
            raise RuntimeError("connection refused")

        with patch.dict(os.environ, {"ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY": "1"}):
            with patch(
                "app.services.orchestration.phases.completion_summary._call_planning_lane",
                new=_fail,
            ):
                result = _generate_task_summary_with_fallback(
                    ctx=ctx, summary_prompt="prompt"
                )

        assert result["fallback"] is True
        assert "Task completed" in result["output"]
        assert "connection refused" in result.get("error", "")
        ctx.emit_live.assert_called_once()

    def test_empty_llm_output_triggers_fallback(self):
        ctx = _make_ctx()

        with patch.dict(os.environ, {"ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY": "1"}):
            with self._patch_planning_lane(""):
                result = _generate_task_summary_with_fallback(
                    ctx=ctx, summary_prompt="prompt"
                )

        assert result["fallback"] is True
        assert "Task completed" in result["output"]

    def test_timeout_triggers_fallback(self):
        ctx = _make_ctx()

        async def _slow(prompt, **kwargs):
            await asyncio.sleep(9999)
            return "never"

        with patch.dict(os.environ, {"ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY": "1"}):
            with patch(
                "app.services.orchestration.phases.completion_summary._call_planning_lane",
                new=_slow,
            ):
                with patch(
                    "app.services.orchestration.phases.completion_summary.SUMMARY_TIMEOUT_SECONDS",
                    0.01,
                ):
                    result = _generate_task_summary_with_fallback(
                        ctx=ctx, summary_prompt="prompt"
                    )

        assert result["fallback"] is True

    def test_runtime_service_never_called_when_flag_on(self):
        ctx = _make_ctx()
        ctx.runtime_service = MagicMock()

        with patch.dict(os.environ, {"ORCHESTRATOR_GENERATE_LLM_TASK_SUMMARY": "1"}):
            with self._patch_planning_lane("summary text"):
                _generate_task_summary_with_fallback(ctx=ctx, summary_prompt="p")

        ctx.runtime_service.assert_not_called()
        ctx.runtime_service.execute_task.assert_not_called()

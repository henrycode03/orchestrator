"""Tests for planning context provenance collection.

Verifies that collect_planning_context_provenance() records injected and
omitted context accurately without changing any planning behavior.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.schemas.knowledge import (
    KnowledgeContext,
    KnowledgeItemRef,
    RecommendedAction,
)
from app.services.orchestration.context.provenance import (
    collect_planning_context_provenance,
)
from app.services.orchestration.events.event_types import EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_knowledge_context(
    items: list[dict],
    *,
    matched_failure_memory: bool = False,
) -> KnowledgeContext:
    return KnowledgeContext(
        retrieved_items=[
            KnowledgeItemRef(
                id=it["id"],
                title=it["title"],
                knowledge_type=it["knowledge_type"],
                content=it.get("content", "body"),
                priority=it.get("priority", 1),
                confidence=it.get("confidence", 0.9),
            )
            for it in items
        ],
        query="test query",
        trigger_phase="planning",
        retrieval_reason="test",
        confidence=0.9,
        matched_failure_memory=matched_failure_memory,
        recommended_action=RecommendedAction.none,
    )


def _base_call(tmp_path: Path, **overrides):
    defaults = dict(
        task_description="Build a CLI tool",
        project_context="Python project",
        project_dir=tmp_path,
        workspace_review={
            "file_count": 3,
            "source_file_count": 2,
            "has_existing_files": True,
        },
        knowledge_context=None,
        planning_prompt="## PLAN\n...",
    )
    defaults.update(overrides)
    return collect_planning_context_provenance(**defaults)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_provenance_has_required_keys(tmp_path):
    result = _base_call(tmp_path)
    for key in (
        "task_description",
        "project_context",
        "workspace_files",
        "knowledge_items_injected",
        "failure_memories_injected",
        "matched_failure_memory",
        "omitted_sources",
    ):
        assert key in result, f"missing key: {key}"


def test_task_description_shape(tmp_path):
    result = _base_call(tmp_path, task_description="Fix the bug")
    assert result["task_description"]["chars"] == len("Fix the bug")
    assert result["task_description"]["preview"] == "Fix the bug"


def test_task_description_preview_truncated_at_200(tmp_path):
    long_desc = "x" * 500
    result = _base_call(tmp_path, task_description=long_desc)
    assert result["task_description"]["chars"] == 500
    assert len(result["task_description"]["preview"]) == 200


def test_project_context_chars(tmp_path):
    result = _base_call(tmp_path, project_context="abc")
    assert result["project_context"]["chars"] == 3


def test_workspace_files_from_review(tmp_path):
    result = _base_call(
        tmp_path,
        workspace_review={
            "file_count": 7,
            "source_file_count": 4,
            "has_existing_files": True,
        },
    )
    wf = result["workspace_files"]
    assert wf["file_count"] == 7
    assert wf["source_file_count"] == 4
    assert wf["has_existing_files"] is True


# ---------------------------------------------------------------------------
# Knowledge injected
# ---------------------------------------------------------------------------


def test_knowledge_items_injected_when_present(tmp_path):
    kctx = _make_knowledge_context(
        [
            {
                "id": "k1",
                "title": "Format Guide",
                "knowledge_type": "format_guide",
                "confidence": 0.8,
            },
            {
                "id": "k2",
                "title": "Task Example",
                "knowledge_type": "task_example",
                "confidence": 0.75,
            },
        ]
    )
    result = _base_call(tmp_path, knowledge_context=kctx)
    items = result["knowledge_items_injected"]
    assert len(items) == 2
    ids = {it["id"] for it in items}
    assert ids == {"k1", "k2"}


def test_failure_memories_filtered_correctly(tmp_path):
    kctx = _make_knowledge_context(
        [
            {"id": "f1", "title": "Failure A", "knowledge_type": "failure_memory"},
            {"id": "k1", "title": "Guide", "knowledge_type": "format_guide"},
        ],
        matched_failure_memory=True,
    )
    result = _base_call(tmp_path, knowledge_context=kctx)
    assert len(result["knowledge_items_injected"]) == 2
    assert len(result["failure_memories_injected"]) == 1
    assert result["failure_memories_injected"][0]["id"] == "f1"
    assert result["matched_failure_memory"] is True


def test_no_failure_memories_when_none_injected(tmp_path):
    kctx = _make_knowledge_context(
        [
            {"id": "k1", "title": "Guide", "knowledge_type": "format_guide"},
        ]
    )
    result = _base_call(tmp_path, knowledge_context=kctx)
    assert result["failure_memories_injected"] == []
    assert result["matched_failure_memory"] is False


def test_empty_knowledge_items_when_no_context(tmp_path):
    result = _base_call(tmp_path, knowledge_context=None)
    assert result["knowledge_items_injected"] == []
    assert result["failure_memories_injected"] == []
    assert result["matched_failure_memory"] is False


# ---------------------------------------------------------------------------
# Omitted sources
# ---------------------------------------------------------------------------


def test_knowledge_omitted_when_retrieval_failed(tmp_path):
    result = _base_call(tmp_path, knowledge_context=None)
    assert "knowledge" in result["omitted_sources"]
    assert result["omitted_sources"]["knowledge"] == "retrieval_failed_or_skipped"


def test_knowledge_omitted_when_empty_items(tmp_path):
    kctx = _make_knowledge_context([])
    result = _base_call(tmp_path, knowledge_context=kctx)
    assert "knowledge" in result["omitted_sources"]
    assert result["omitted_sources"]["knowledge"] == "no_items_retrieved"


def test_knowledge_not_omitted_when_items_present(tmp_path):
    kctx = _make_knowledge_context(
        [
            {"id": "k1", "title": "G", "knowledge_type": "format_guide"},
        ]
    )
    result = _base_call(tmp_path, knowledge_context=kctx)
    assert "knowledge" not in result["omitted_sources"]


def test_python_source_omitted_when_no_test_files(tmp_path):
    with patch(
        "app.services.orchestration.context.provenance.python_test_source_context_from_tests",
        return_value=None,
    ):
        result = _base_call(tmp_path)
    assert "python_source_context" in result["omitted_sources"]
    assert result["omitted_sources"]["python_source_context"] == "no_test_files_found"


def test_python_source_not_omitted_when_test_files_present(tmp_path):
    with patch(
        "app.services.orchestration.context.provenance.python_test_source_context_from_tests",
        return_value="# test content",
    ):
        result = _base_call(tmp_path)
    assert "python_source_context" not in result["omitted_sources"]


def test_planning_prompt_omitted_when_runtime_service_unavailable(tmp_path):
    result = _base_call(tmp_path, planning_prompt=None)
    assert "planning_prompt" in result["omitted_sources"]
    assert result["omitted_sources"]["planning_prompt"] == "runtime_service_unavailable"


def test_no_omissions_when_all_sources_present(tmp_path):
    kctx = _make_knowledge_context(
        [
            {"id": "k1", "title": "G", "knowledge_type": "format_guide"},
        ]
    )
    with patch(
        "app.services.orchestration.context.provenance.python_test_source_context_from_tests",
        return_value="# content",
    ):
        result = _base_call(tmp_path, knowledge_context=kctx, planning_prompt="prompt")
    assert result["omitted_sources"] == {}


# ---------------------------------------------------------------------------
# EventType constant
# ---------------------------------------------------------------------------


def test_planning_context_provenance_event_type_defined():
    assert hasattr(EventType, "PLANNING_CONTEXT_PROVENANCE")
    assert EventType.PLANNING_CONTEXT_PROVENANCE == "planning_context_provenance"


def test_planning_context_provenance_in_all_event_types():
    from app.services.orchestration.events.event_types import _ALL_EVENT_TYPES

    assert "planning_context_provenance" in _ALL_EVENT_TYPES


# ---------------------------------------------------------------------------
# Robustness: None / empty inputs
# ---------------------------------------------------------------------------


def test_handles_none_task_description(tmp_path):
    result = _base_call(tmp_path, task_description=None)
    assert result["task_description"]["chars"] == 0
    assert result["task_description"]["preview"] == ""


def test_handles_none_project_context(tmp_path):
    result = _base_call(tmp_path, project_context=None)
    assert result["project_context"]["chars"] == 0


def test_handles_empty_workspace_review(tmp_path):
    result = _base_call(tmp_path, workspace_review={})
    wf = result["workspace_files"]
    assert wf["file_count"] == 0
    assert wf["source_file_count"] == 0
    assert wf["has_existing_files"] is False


def test_python_source_exception_treated_as_omitted(tmp_path):
    with patch(
        "app.services.orchestration.context.provenance.python_test_source_context_from_tests",
        side_effect=RuntimeError("filesystem error"),
    ):
        result = _base_call(tmp_path)
    assert "python_source_context" in result["omitted_sources"]

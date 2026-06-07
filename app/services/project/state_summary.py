"""Read-only ProjectStateSummary: durable task memory for multi-task sessions.

Consolidates completed-task outcomes, pending work, planning artifact constraints,
and the canonical workspace root into a single diagnostic payload. Nothing here
changes planning, validation, or execution behavior.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as DbSession

from app.models import (
    PlanningArtifact,
    PlanningSession,
    Project,
    Task,
    TaskExecutionChangeSet,
    TaskStatus,
)

_ARTIFACT_EXCERPT_CHARS = 400
_DESCRIPTION_PREVIEW_CHARS = 300
_MAX_FILES_PER_TASK = 40
_PROMOTION_NOTE_CHARS = 300

# Injection renderer limits — conservative defaults to stay inside context budget.
_PSS_BLOCK_MAX_CHARS = 1800
_PSS_FILES_MAX = 20
_PSS_ARTIFACT_EXCERPT_CHARS = 200


def _files_for_task(task_id: int, db: DbSession) -> List[str]:
    """Aggregate added + modified filenames from accepted change sets for a task."""
    change_sets = (
        db.query(TaskExecutionChangeSet)
        .filter(TaskExecutionChangeSet.task_id == task_id)
        .order_by(TaskExecutionChangeSet.id.asc())
        .all()
    )
    seen: set[str] = set()
    result: List[str] = []
    for cs in change_sets:
        for f in (cs.added_files or []) + (cs.modified_files or []):
            name = str(f).strip() if f else ""
            if name and name not in seen:
                seen.add(name)
                result.append(name)
                if len(result) >= _MAX_FILES_PER_TASK:
                    return result
    return result


def _latest_artifacts(planning_session_id: int, db: DbSession) -> Dict[str, str]:
    """Return {artifact_type: content_excerpt} for the latest planning artifacts."""
    rows = (
        db.query(PlanningArtifact)
        .filter(
            PlanningArtifact.planning_session_id == planning_session_id,
            PlanningArtifact.is_latest.is_(True),
        )
        .all()
    )
    # Fallback: if no is_latest rows, take highest version per type
    if not rows:
        rows = (
            db.query(PlanningArtifact)
            .filter(PlanningArtifact.planning_session_id == planning_session_id)
            .order_by(
                PlanningArtifact.artifact_type.asc(),
                PlanningArtifact.version.desc(),
            )
            .all()
        )
    seen_types: set[str] = set()
    result: Dict[str, str] = {}
    for row in rows:
        if row.artifact_type not in seen_types:
            seen_types.add(row.artifact_type)
            result[row.artifact_type] = (row.content or "")[:_ARTIFACT_EXCERPT_CHARS]
    return result


def build_project_state_summary(project_id: int, db: DbSession) -> Dict[str, Any]:
    """Return a read-only ProjectStateSummary for diagnostic purposes."""
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.deleted_at.is_(None))
        .first()
    )
    if not project:
        return {
            "error": "project_not_found",
            "project_id": project_id,
            "computed_at": datetime.now(UTC).isoformat(),
        }

    all_tasks = (
        db.query(Task)
        .filter(Task.project_id == project_id)
        .order_by(
            Task.plan_position.asc().nullslast(),
            Task.id.asc(),
        )
        .all()
    )

    completed_tasks = [t for t in all_tasks if t.status == TaskStatus.DONE]
    pending_tasks = [t for t in all_tasks if t.status == TaskStatus.PENDING]

    completed_task_records: List[Dict[str, Any]] = []
    all_files: List[str] = []
    for task in completed_tasks:
        task_files = _files_for_task(task.id, db)
        all_files.extend(f for f in task_files if f not in all_files)
        completed_task_records.append(
            {
                "task_id": task.id,
                "plan_position": task.plan_position,
                "title": task.title,
                "workspace_status": task.workspace_status,
                "task_subfolder": task.task_subfolder,
                "completed_at": (
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                "promotion_note": (
                    (task.promotion_note or "")[:_PROMOTION_NOTE_CHARS]
                    if task.promotion_note
                    else None
                ),
                "files_created_or_modified": task_files,
            }
        )

    pending_task_records: List[Dict[str, Any]] = [
        {
            "task_id": t.id,
            "plan_position": t.plan_position,
            "title": t.title,
            "description": (t.description or "")[:_DESCRIPTION_PREVIEW_CHARS],
        }
        for t in pending_tasks
    ]

    # Planning artifact constraints: latest planning session for this project
    latest_ps = (
        db.query(PlanningSession)
        .filter(PlanningSession.project_id == project_id)
        .order_by(PlanningSession.id.desc())
        .first()
    )
    known_constraints: Dict[str, Any] = {
        "source": "latest_planning_session_artifacts",
        "planning_session_id": latest_ps.id if latest_ps else None,
        "requirements_excerpt": None,
        "design_excerpt": None,
        "implementation_plan_excerpt": None,
    }
    if latest_ps:
        artifacts = _latest_artifacts(latest_ps.id, db)
        known_constraints["requirements_excerpt"] = artifacts.get("requirements")
        known_constraints["design_excerpt"] = artifacts.get("design")
        known_constraints["implementation_plan_excerpt"] = artifacts.get(
            "implementation_plan"
        )

    next_task = pending_tasks[0] if pending_tasks else None
    next_task_recommendation: Optional[Dict[str, Any]] = None
    if next_task:
        next_task_recommendation = {
            "task_id": next_task.id,
            "plan_position": next_task.plan_position,
            "title": next_task.title,
            "description": (next_task.description or "")[:_DESCRIPTION_PREVIEW_CHARS],
        }

    return {
        "project_id": project_id,
        "project_name": project.name,
        "canonical_root": project.workspace_path or "unknown",
        "computed_at": datetime.now(UTC).isoformat(),
        "completed_tasks": completed_task_records,
        "pending_tasks": pending_task_records,
        "files_created_or_modified": all_files[:100],
        "known_constraints": known_constraints,
        "next_task_recommendation": next_task_recommendation,
    }


def render_project_state_summary_block(
    summary: Dict[str, Any],
    *,
    max_chars: int = _PSS_BLOCK_MAX_CHARS,
) -> str:
    """Render a ProjectStateSummary dict as a compact planning-prompt block.

    The block is prefixed so the planner can locate it in the context. It is
    intentionally terse: only completed-task file lists and planning-artifact
    excerpts are included. Verbose fields (full descriptions, timestamps) are
    omitted.

    Returns an empty string when the summary has no completed tasks (avoids
    injecting a useless block for single-task sessions).
    """
    completed = summary.get("completed_tasks") or []
    if not completed:
        return ""

    lines: List[str] = [
        "=== PROJECT STATE SUMMARY ===",
        f"Project: {summary.get('project_name', 'unknown')}",
        f"Canonical root: {summary.get('canonical_root', 'unknown')}",
    ]

    lines.append("Completed tasks:")
    for t in completed:
        pos = t.get("plan_position")
        pos_label = f"#{pos}" if pos is not None else "manual"
        title = t.get("title") or "untitled"
        files = t.get("files_created_or_modified") or []
        file_str = ", ".join(files[:_PSS_FILES_MAX]) if files else "no files recorded"
        lines.append(f"  {pos_label} {title} → {file_str}")

    constraints = summary.get("known_constraints") or {}
    req = (constraints.get("requirements_excerpt") or "")[:_PSS_ARTIFACT_EXCERPT_CHARS]
    impl = (constraints.get("implementation_plan_excerpt") or "")[
        :_PSS_ARTIFACT_EXCERPT_CHARS
    ]
    if req:
        lines.append(f"Requirements: {req.strip()}")
    if impl:
        lines.append(f"Implementation plan: {impl.strip()}")

    next_rec = summary.get("next_task_recommendation")
    if next_rec:
        npos = next_rec.get("plan_position")
        npos_label = f"#{npos}" if npos is not None else "next"
        lines.append(f"Next task ({npos_label}): {next_rec.get('title', 'untitled')}")

    lines.append("=== END PROJECT STATE SUMMARY ===")
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[: max_chars - 3] + "..."
    return block


def _inject_project_state_summary_into_context(
    *,
    orchestration_state: Any,
    db: DbSession,
    project_id: int,
    logger: Any,
    task_position: Optional[int] = None,
) -> None:
    """Experimental: prepend a ProjectStateSummary block to project_context.

    Only active when PSS_CONTINUATION_INJECTION_ENABLED=True. Guarded by
    task_position: does not inject for plan_position == 1 because there is
    no prior completed work to summarise.

    This function does not change any planning schema, validator, or repair
    logic. It only prepends a diagnostic text block to project_context before
    the planning prompt is assembled.
    """
    if task_position is not None and task_position == 1:
        return
    try:
        summary = build_project_state_summary(project_id, db)
        if summary.get("error"):
            logger.debug(
                "[PSS_INJECT] Skipping: build_project_state_summary returned error=%s",
                summary.get("error"),
            )
            return
        block = render_project_state_summary_block(summary)
        if not block:
            logger.debug("[PSS_INJECT] Skipping: no completed tasks in PSS")
            return
        current = orchestration_state.project_context or ""
        injected = (block + "\n\n" + current)[:8000]
        orchestration_state.project_context = injected
        logger.info(
            "[PSS_INJECT] Injected ProjectStateSummary block (%d chars) into planning context",
            len(block),
        )
    except Exception as exc:
        logger.warning("[PSS_INJECT] Failed to inject PSS (non-fatal): %s", exc)

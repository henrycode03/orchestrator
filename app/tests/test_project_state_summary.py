"""Tests for ProjectStateSummary diagnostic service.

Covers: shape, completed/pending task listing, files-changed aggregation,
planning artifact constraint extraction, next-task recommendation,
and the Garden Story Task 1 → Task 2 transition scenario.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import (
    Base,
    PlanningArtifact,
    PlanningSession,
    Project,
    Task,
    TaskExecutionChangeSet,
    TaskExecution,
    TaskStatus,
    Session as OrchestratorSession,
    SessionTask,
)
from app.services.project.state_summary import (
    _files_for_task,
    _latest_artifacts,
    build_project_state_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _make_project(db, *, name="Test Project", workspace_path="/tmp/test") -> Project:
    p = Project(name=name, description="desc", workspace_path=workspace_path)
    db.add(p)
    db.flush()
    return p


def _make_task(
    db,
    project: Project,
    *,
    title: str,
    status: TaskStatus = TaskStatus.PENDING,
    plan_position: int = 1,
    description: str = "",
    task_subfolder: str | None = None,
) -> Task:
    t = Task(
        project_id=project.id,
        title=title,
        description=description,
        status=status,
        plan_position=plan_position,
        task_subfolder=task_subfolder,
    )
    db.add(t)
    db.flush()
    return t


def _make_task_execution(db, project: Project, task: Task) -> TaskExecution:
    session = OrchestratorSession(
        project_id=project.id,
        name="test-session",
    )
    db.add(session)
    db.flush()
    te = TaskExecution(
        session_id=session.id,
        task_id=task.id,
        attempt_number=1,
        status=TaskStatus.DONE,
    )
    db.add(te)
    db.flush()
    return te


def _make_change_set(
    db,
    project: Project,
    task: Task,
    task_execution: TaskExecution,
    *,
    added: list[str] | None = None,
    modified: list[str] | None = None,
) -> TaskExecutionChangeSet:
    cs = TaskExecutionChangeSet(
        project_id=project.id,
        task_id=task.id,
        task_execution_id=task_execution.id,
        base_snapshot_key="base",
        added_files=added or [],
        modified_files=modified or [],
        deleted_files=[],
        warning_flags=[],
    )
    db.add(cs)
    db.flush()
    return cs


def _make_planning_session(db, project: Project) -> PlanningSession:
    ps = PlanningSession(
        project_id=project.id,
        title="Plan",
        prompt="Build a project",
        status="completed",
        source_brain="local",
    )
    db.add(ps)
    db.flush()
    return ps


def _make_artifact(
    db,
    planning_session: PlanningSession,
    *,
    artifact_type: str,
    content: str,
    is_latest: bool = True,
) -> PlanningArtifact:
    pa = PlanningArtifact(
        planning_session_id=planning_session.id,
        artifact_type=artifact_type,
        filename=f"{artifact_type}.md",
        content=content,
        version=1,
        is_latest=is_latest,
    )
    db.add(pa)
    db.flush()
    return pa


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_summary_has_required_keys(db):
    p = _make_project(db)
    db.commit()
    result = build_project_state_summary(p.id, db)
    for key in (
        "project_id",
        "project_name",
        "canonical_root",
        "computed_at",
        "completed_tasks",
        "pending_tasks",
        "files_created_or_modified",
        "known_constraints",
        "next_task_recommendation",
    ):
        assert key in result, f"missing key: {key}"


def test_summary_project_not_found_returns_error(db):
    result = build_project_state_summary(9999, db)
    assert "error" in result
    assert result["error"] == "project_not_found"
    assert result["project_id"] == 9999


def test_summary_project_name_and_root(db):
    p = _make_project(db, name="My Project", workspace_path="/mnt/ws")
    db.commit()
    result = build_project_state_summary(p.id, db)
    assert result["project_name"] == "My Project"
    assert result["canonical_root"] == "/mnt/ws"


def test_summary_unknown_root_when_workspace_path_null(db):
    p = Project(name="NoRoot", workspace_path=None)
    db.add(p)
    db.commit()
    result = build_project_state_summary(p.id, db)
    assert result["canonical_root"] == "unknown"


# ---------------------------------------------------------------------------
# Completed task records
# ---------------------------------------------------------------------------


def test_completed_tasks_listed(db):
    p = _make_project(db)
    t1 = _make_task(db, p, title="Task A", status=TaskStatus.DONE, plan_position=1)
    _make_task(db, p, title="Task B", status=TaskStatus.PENDING, plan_position=2)
    t1.completed_at = datetime.now(UTC)
    db.commit()
    result = build_project_state_summary(p.id, db)
    assert len(result["completed_tasks"]) == 1
    assert result["completed_tasks"][0]["title"] == "Task A"
    assert result["completed_tasks"][0]["task_id"] == t1.id


def test_completed_tasks_ordered_by_plan_position(db):
    p = _make_project(db)
    _make_task(db, p, title="Task B", status=TaskStatus.DONE, plan_position=2)
    _make_task(db, p, title="Task A", status=TaskStatus.DONE, plan_position=1)
    db.commit()
    result = build_project_state_summary(p.id, db)
    titles = [t["title"] for t in result["completed_tasks"]]
    assert titles == ["Task A", "Task B"]


def test_completed_task_record_has_expected_keys(db):
    p = _make_project(db)
    _make_task(db, p, title="T", status=TaskStatus.DONE, plan_position=1)
    db.commit()
    result = build_project_state_summary(p.id, db)
    rec = result["completed_tasks"][0]
    for key in (
        "task_id",
        "plan_position",
        "title",
        "workspace_status",
        "task_subfolder",
        "completed_at",
        "promotion_note",
        "files_created_or_modified",
    ):
        assert key in rec, f"missing key in completed task: {key}"


# ---------------------------------------------------------------------------
# Pending tasks
# ---------------------------------------------------------------------------


def test_pending_tasks_listed(db):
    p = _make_project(db)
    _make_task(db, p, title="Task A", status=TaskStatus.DONE, plan_position=1)
    _make_task(db, p, title="Task B", status=TaskStatus.PENDING, plan_position=2)
    db.commit()
    result = build_project_state_summary(p.id, db)
    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0]["title"] == "Task B"


def test_no_tasks_produces_empty_lists(db):
    p = _make_project(db)
    db.commit()
    result = build_project_state_summary(p.id, db)
    assert result["completed_tasks"] == []
    assert result["pending_tasks"] == []
    assert result["next_task_recommendation"] is None


# ---------------------------------------------------------------------------
# Files created/modified
# ---------------------------------------------------------------------------


def test_files_aggregated_from_change_sets(db):
    p = _make_project(db)
    t1 = _make_task(db, p, title="T1", status=TaskStatus.DONE, plan_position=1)
    te = _make_task_execution(db, p, t1)
    _make_change_set(db, p, t1, te, added=["index.html", "style.css"])
    db.commit()
    result = build_project_state_summary(p.id, db)
    assert "index.html" in result["files_created_or_modified"]
    assert "style.css" in result["files_created_or_modified"]


def test_files_per_task_in_completed_record(db):
    p = _make_project(db)
    t1 = _make_task(db, p, title="T1", status=TaskStatus.DONE, plan_position=1)
    te = _make_task_execution(db, p, t1)
    _make_change_set(db, p, t1, te, added=["foo.py"], modified=["bar.py"])
    db.commit()
    result = build_project_state_summary(p.id, db)
    task_files = result["completed_tasks"][0]["files_created_or_modified"]
    assert "foo.py" in task_files
    assert "bar.py" in task_files


def test_files_deduped_across_change_sets(db):
    p = _make_project(db)
    t1 = _make_task(db, p, title="T1", status=TaskStatus.DONE, plan_position=1)
    te1 = _make_task_execution(db, p, t1)
    # second attempt = second TaskExecution → second change set
    session = (
        db.query(OrchestratorSession)
        .filter(OrchestratorSession.project_id == p.id)
        .first()
    )
    te2 = TaskExecution(
        session_id=session.id,
        task_id=t1.id,
        attempt_number=2,
        status=TaskStatus.DONE,
    )
    db.add(te2)
    db.flush()
    _make_change_set(db, p, t1, te1, added=["index.html"])
    _make_change_set(db, p, t1, te2, modified=["index.html"])
    db.commit()
    files = _files_for_task(t1.id, db)
    assert files.count("index.html") == 1


def test_no_change_sets_yields_empty_files(db):
    p = _make_project(db)
    t1 = _make_task(db, p, title="T1", status=TaskStatus.DONE, plan_position=1)
    db.commit()
    files = _files_for_task(t1.id, db)
    assert files == []


# ---------------------------------------------------------------------------
# Planning artifact constraints
# ---------------------------------------------------------------------------


def test_known_constraints_populated_from_artifacts(db):
    p = _make_project(db)
    ps = _make_planning_session(db, p)
    _make_artifact(
        db, ps, artifact_type="requirements", content="# Requirements\n- Use Flask"
    )
    _make_artifact(db, ps, artifact_type="design", content="# Design\n- REST API")
    db.commit()
    result = build_project_state_summary(p.id, db)
    kc = result["known_constraints"]
    assert kc["planning_session_id"] == ps.id
    assert "Requirements" in (kc["requirements_excerpt"] or "")
    assert "Design" in (kc["design_excerpt"] or "")


def test_known_constraints_null_when_no_planning_session(db):
    p = _make_project(db)
    db.commit()
    result = build_project_state_summary(p.id, db)
    kc = result["known_constraints"]
    assert kc["planning_session_id"] is None
    assert kc["requirements_excerpt"] is None
    assert kc["design_excerpt"] is None


def test_latest_artifacts_uses_is_latest_flag(db):
    p = _make_project(db)
    ps = _make_planning_session(db, p)
    # older version
    pa_old = PlanningArtifact(
        planning_session_id=ps.id,
        artifact_type="requirements",
        filename="requirements.md",
        content="old content",
        version=1,
        is_latest=False,
    )
    # newer version
    pa_new = PlanningArtifact(
        planning_session_id=ps.id,
        artifact_type="requirements",
        filename="requirements.md",
        content="new content",
        version=2,
        is_latest=True,
    )
    db.add_all([pa_old, pa_new])
    db.commit()
    artifacts = _latest_artifacts(ps.id, db)
    assert "new content" in artifacts.get("requirements", "")


# ---------------------------------------------------------------------------
# Next task recommendation
# ---------------------------------------------------------------------------


def test_next_task_recommendation_is_first_pending(db):
    p = _make_project(db)
    _make_task(db, p, title="T1", status=TaskStatus.DONE, plan_position=1)
    _make_task(db, p, title="T2", status=TaskStatus.PENDING, plan_position=2)
    _make_task(db, p, title="T3", status=TaskStatus.PENDING, plan_position=3)
    db.commit()
    result = build_project_state_summary(p.id, db)
    rec = result["next_task_recommendation"]
    assert rec is not None
    assert rec["title"] == "T2"
    assert rec["plan_position"] == 2


def test_next_task_recommendation_none_when_all_done(db):
    p = _make_project(db)
    _make_task(db, p, title="T1", status=TaskStatus.DONE, plan_position=1)
    db.commit()
    result = build_project_state_summary(p.id, db)
    assert result["next_task_recommendation"] is None


# ---------------------------------------------------------------------------
# Garden Story Task 1 → Task 2 transition scenario
# ---------------------------------------------------------------------------


def test_garden_story_task1_to_task2_transition(db):
    """Task 1 completes; summary shows it done, Task 2 as next recommendation."""
    p = _make_project(db, name="Garden Story", workspace_path="/projects/garden")
    ps = _make_planning_session(db, p)
    _make_artifact(
        db,
        ps,
        artifact_type="requirements",
        content=(
            "# Requirements\n"
            "- Create a static flower landing page.\n"
            "- Files: index.html, css/style.css, images/flower-bg.svg."
        ),
    )
    _make_artifact(
        db,
        ps,
        artifact_type="implementation_plan",
        content=(
            "# Implementation Plan\n"
            "1. Task 1: Create HTML and CSS files.\n"
            "2. Task 2: Add SVG and wire stylesheet."
        ),
    )
    task1 = _make_task(
        db,
        p,
        title="Create flower landing page HTML and CSS",
        status=TaskStatus.DONE,
        plan_position=1,
        task_subfolder="task-1-landing-page",
    )
    task1.completed_at = datetime.now(UTC)
    task2 = _make_task(
        db,
        p,
        title="Add SVG illustration and wire stylesheet",
        status=TaskStatus.PENDING,
        plan_position=2,
        description="Add images/flower-bg.svg and ensure index.html references it correctly.",
    )
    # Simulate Task 1 produced files
    te = _make_task_execution(db, p, task1)
    _make_change_set(db, p, task1, te, added=["index.html", "css/style.css"])
    db.commit()

    result = build_project_state_summary(p.id, db)

    # Project metadata
    assert result["project_name"] == "Garden Story"
    assert result["canonical_root"] == "/projects/garden"

    # Task 1 complete
    assert len(result["completed_tasks"]) == 1
    ct = result["completed_tasks"][0]
    assert ct["title"] == "Create flower landing page HTML and CSS"
    assert ct["plan_position"] == 1
    assert "index.html" in ct["files_created_or_modified"]
    assert "css/style.css" in ct["files_created_or_modified"]

    # Task 2 pending
    assert len(result["pending_tasks"]) == 1
    assert (
        result["pending_tasks"][0]["title"]
        == "Add SVG illustration and wire stylesheet"
    )

    # Planning constraints from artifacts
    kc = result["known_constraints"]
    assert kc["planning_session_id"] == ps.id
    assert kc["requirements_excerpt"] is not None
    assert "flower" in kc["requirements_excerpt"].lower()
    assert kc["implementation_plan_excerpt"] is not None

    # Next task recommendation points to Task 2
    rec = result["next_task_recommendation"]
    assert rec is not None
    assert rec["task_id"] == task2.id
    assert rec["plan_position"] == 2
    assert "SVG" in rec["title"]

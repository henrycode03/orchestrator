from __future__ import annotations

from pathlib import Path

from app.models import Project, Session as SessionModel, Task, TaskExecution, TaskStatus
from app.services.orchestration.execution.runtime import (
    snapshot_workspace_before_run,
    workspace_snapshot_key,
)
from app.services.tasks.service import TaskService


def test_reject_restore_preserves_nested_tracked_product_metadata(
    db_session,
    tmp_path: Path,
):
    """Reproduce the A3-R1 loss of excluded-but-tracked ProductRoot paths."""

    project_root = tmp_path / "project"
    (project_root / "frontend").mkdir(parents=True)
    (project_root / "relay").mkdir(parents=True)
    (project_root / "scripts" / "fixture" / ".openclaw" / "events").mkdir(parents=True)
    baseline_files = {
        "normal.txt": "baseline\n",
        "frontend/keep.py": "print('keep')\n",
        "frontend/.gitignore": "generated/\n",
        "relay/.gitignore": "cache/\n",
        "scripts/fixture/.openclaw/events/.gitkeep": "",
    }
    for relative, content in baseline_files.items():
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    project = Project(
        name="phase35s1-rejection-restore",
        workspace_path=str(project_root),
    )
    db_session.add(project)
    db_session.flush()
    task = Task(
        project_id=project.id,
        title="S1 restore reproduction",
        description="Reproduce excluded tracked paths disappearing on reject",
        status=TaskStatus.DONE,
        workspace_status="ready",
        task_subfolder="task-phase35s1-repro",
    )
    session = SessionModel(project_id=project.id, name="phase35s1-session")
    db_session.add_all([task, session])
    db_session.commit()
    db_session.refresh(task)
    db_session.refresh(session)
    execution = TaskExecution(
        session_id=session.id,
        task_id=task.id,
        attempt_number=1,
        status=TaskStatus.DONE,
    )
    db_session.add(execution)
    db_session.commit()
    db_session.refresh(execution)

    task_service = TaskService(db_session)
    snapshot = snapshot_workspace_before_run(
        task_service,
        project,
        task.id,
        project_root,
        task_execution_id=execution.id,
        preserve_project_root_rules=True,
    )
    snapshot_root = Path(snapshot["snapshot_path"])

    assert (snapshot_root / "normal.txt").read_text(encoding="utf-8") == "baseline\n"
    assert (snapshot_root / "frontend/keep.py").exists()
    assert not (snapshot_root / "frontend/.gitignore").exists()
    assert not (snapshot_root / "relay/.gitignore").exists()
    assert not (snapshot_root / "scripts/fixture/.openclaw/events/.gitkeep").exists()

    (project_root / "normal.txt").write_text("candidate\n", encoding="utf-8")
    result = task_service.reject_task_execution_change_set(
        project,
        task,
        task_execution_id=execution.id,
        snapshot_key=workspace_snapshot_key(task.id, execution.id),
    )

    assert result["rejected"] is True
    observations = {
        "normal_file_restored": (
            (project_root / "normal.txt").read_text(encoding="utf-8") == "baseline\n"
        ),
        "frontend_keep_survives": (project_root / "frontend/keep.py").exists(),
        "frontend_gitignore_survives": (project_root / "frontend/.gitignore").exists(),
        "relay_gitignore_survives": (project_root / "relay/.gitignore").exists(),
        "nested_openclaw_survives": (
            project_root / "scripts/fixture/.openclaw/events/.gitkeep"
        ).exists(),
    }
    assert observations == {
        "normal_file_restored": True,
        "frontend_keep_survives": True,
        "frontend_gitignore_survives": True,
        "relay_gitignore_survives": True,
        "nested_openclaw_survives": True,
    }

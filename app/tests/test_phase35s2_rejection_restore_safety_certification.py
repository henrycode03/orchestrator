from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from app.models import Project, Session as SessionModel, Task, TaskExecution, TaskStatus
from app.services.orchestration.execution.runtime import (
    snapshot_workspace_before_run,
    workspace_snapshot_key,
)
from app.services.tasks.service import TaskService


def _manifest(root: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def test_reject_restore_certification_with_real_git_product_root(
    db_session,
    tmp_path: Path,
):
    """Certify rollback against a temporary real Git ProductRoot."""

    project_root = tmp_path / "real-product"
    for relative in (
        "frontend",
        "relay",
        "scripts/fixture/.openclaw/events",
    ):
        (project_root / relative).mkdir(parents=True, exist_ok=True)
    baseline = {
        "normal.txt": b"baseline\n",
        "frontend/keep.py": b"print('keep')\n",
        "frontend/.gitignore": b"generated/\n",
        "relay/.gitignore": b"cache/\n",
        "scripts/fixture/.openclaw/events/.gitkeep": b"",
        "scripts/fixture/runtime.json": b'{"product": true}\n',
        "runtime.json": b'{"runtime": true}\n',
    }
    for relative, content in baseline.items():
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    subprocess.run(["git", "init", "-q"], cwd=project_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "phase35s2@example.test"],
        cwd=project_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Phase35 S2"],
        cwd=project_root,
        check=True,
    )

    project = Project(
        name="phase35s2-real-product",
        workspace_path=str(project_root),
    )
    db_session.add(project)
    db_session.flush()
    task = Task(
        project_id=project.id,
        title="S2 rollback certification",
        description="Certify rejection restores ProductRoot exactly",
        status=TaskStatus.DONE,
        workspace_status="ready",
        task_subfolder="task-phase35s2-certification",
    )
    session = SessionModel(project_id=project.id, name="phase35s2-session")
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
    task_service.ensure_project_gitignore_guard(project)
    subprocess.run(["git", "add", "-A"], cwd=project_root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "phase35-s2-baseline"],
        cwd=project_root,
        check=True,
    )
    (project_root / ".git" / "phase35-protected-sentinel").write_bytes(b"protected\n")

    special_paths = (
        "frontend/keep.py",
        "frontend/.gitignore",
        "relay/.gitignore",
        "scripts/fixture/.openclaw/events/.gitkeep",
        "scripts/fixture/runtime.json",
    )
    baseline_manifest = _manifest(project_root, special_paths)
    snapshot = snapshot_workspace_before_run(
        task_service,
        project,
        task.id,
        project_root,
        task_execution_id=execution.id,
        preserve_project_root_rules=True,
    )
    snapshot_root = Path(snapshot["snapshot_path"])

    assert all((snapshot_root / relative).exists() for relative in special_paths)
    assert not (snapshot_root / ".git").exists()
    assert not (snapshot_root / ".gitignore").exists()
    assert not (snapshot_root / "runtime.json").exists()

    (project_root / "normal.txt").write_bytes(b"candidate\n")
    (project_root / "candidate-only.txt").write_bytes(b"candidate\n")
    change_set = task_service.persist_task_execution_change_set(
        project,
        task,
        session_id=session.id,
        task_execution_id=execution.id,
        snapshot_key=workspace_snapshot_key(task.id, execution.id),
        target_dir=project_root,
        preserve_project_root_rules=True,
    )
    assert "normal.txt" in change_set["modified_files"]
    assert "candidate-only.txt" in change_set["added_files"]

    result = task_service.reject_task_execution_change_set(
        project,
        task,
        task_execution_id=execution.id,
        snapshot_key=workspace_snapshot_key(task.id, execution.id),
    )

    assert result["rejected"] is True
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    assert (project_root / "normal.txt").read_bytes() == b"baseline\n"
    assert not (project_root / "candidate-only.txt").exists()
    assert _manifest(project_root, special_paths) == baseline_manifest
    assert (project_root / ".git" / "phase35-protected-sentinel").read_bytes() == (
        b"protected\n"
    )

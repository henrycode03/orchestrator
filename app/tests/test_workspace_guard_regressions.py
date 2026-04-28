from __future__ import annotations

import logging

from app.services.orchestration.workspace_guard import normalize_command, normalize_step


def test_normalize_command_allows_dev_null_redirection(tmp_path):
    project_dir = tmp_path / "skillsync"
    project_dir.mkdir(parents=True)

    command = (
        "cat backend/package.json | python3 -m json.tool > /dev/null "
        "&& test -f backend/src/index.ts && echo 'Backend foundation verified'"
    )

    normalized = normalize_command(command, project_dir)

    assert "/dev/null" in normalized


def test_normalize_step_allows_verification_commands_that_sink_to_dev_null(tmp_path):
    project_dir = tmp_path / "skillsync"
    project_dir.mkdir(parents=True)
    logger = logging.getLogger("workspace-guard-test")

    step = {
        "step_number": 2,
        "description": "Verify backend foundation",
        "commands": [
            "mkdir -p backend/src && touch backend/src/index.ts backend/package.json"
        ],
        "verification": (
            "cat backend/package.json | python3 -m json.tool > /dev/null "
            "&& test -f backend/src/index.ts && echo 'Backend foundation verified'"
        ),
        "rollback": "rm -rf backend",
        "expected_files": ["backend/package.json", "backend/src/index.ts"],
    }

    normalized = normalize_step(step, project_dir, logger, step_index=2)

    assert normalized["verification"] is not None
    assert "/dev/null" in normalized["verification"]

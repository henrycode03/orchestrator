"""Executor-stage helpers for orchestration."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.models import LogEntry


class ExecutorService:
    """Step execution support helpers."""

    TOOL_FAILURE_PATTERNS = (
        "read failed: ENOENT",
        "read failed: EISDIR",
        "exec failed: exec preflight",
        "complex interpreter invocation detected",
        "no such file or directory, access",
        "illegal operation on a directory, read",
    )

    @classmethod
    def recent_step_tool_failures(
        cls,
        db: Session,
        session_id: int,
        task_id: int,
        started_at: datetime,
    ) -> List[str]:
        recent_logs = (
            db.query(LogEntry)
            .filter(
                LogEntry.session_id == session_id,
                LogEntry.task_id == task_id,
                LogEntry.created_at >= started_at,
            )
            .order_by(LogEntry.created_at.asc(), LogEntry.id.asc())
            .all()
        )
        matches: List[str] = []
        for log in recent_logs:
            message = str(log.message or "")
            lowered = message.lower()
            if any(pattern.lower() in lowered for pattern in cls.TOOL_FAILURE_PATTERNS):
                matches.append(message[:500])
        return matches

    @staticmethod
    def tool_failure_correction_hints(
        tool_failures: List[str], project_dir: Path
    ) -> List[str]:
        hints: List[str] = []

        for failure in tool_failures:
            message = str(failure or "")

            raw_params_match = re.search(r"raw_params=(\{.*\})", message)
            raw_params: Dict[str, Any] = {}
            if raw_params_match:
                try:
                    raw_params = json.loads(raw_params_match.group(1))
                except json.JSONDecodeError:
                    raw_params = {}

            raw_path = str(raw_params.get("path") or "").strip()
            if raw_path and not Path(raw_path).is_absolute():
                corrected_path = (project_dir / raw_path).resolve()
                hints.append(
                    "File-tool paths are being resolved against the wrong root. "
                    f"Retry the file read/write using the absolute task-workspace path "
                    f"`{corrected_path}` instead of `{raw_path}`."
                )
            elif raw_path and Path(raw_path).is_absolute():
                corrected_path = (project_dir / raw_path.lstrip("/")).resolve()
                if not Path(raw_path).exists() and corrected_path.is_relative_to(
                    project_dir
                ):
                    hints.append(
                        "The file-tool path looks like a truncated absolute path. "
                        f"Do not shorten the workspace root. Retry with the real absolute "
                        f"task-workspace path `{project_dir}` or a file inside it, not `{raw_path}`."
                    )

            raw_command = str(raw_params.get("command") or "").strip()
            if raw_command.startswith("cd ") and "&&" in raw_command:
                hints.append(
                    "The execution tool rejected a wrapped shell command. "
                    "Retry with a direct command such as `node dist/server.js` and rely "
                    f"on the task working directory `{project_dir}` instead of `cd ... &&`."
                )

            if "read failed: eisd" in message.lower():
                hints.append(
                    "A directory path was passed to the file-read tool. Retry by reading "
                    "an actual file path inside the task workspace, not the folder itself."
                )
            elif raw_path and re.search(r"/task-[^/]+/?$", raw_path):
                hints.append(
                    "A task workspace directory was passed to the file-read tool. "
                    "Read a specific file inside that directory, not the directory path itself."
                )

        deduped: List[str] = []
        seen = set()
        for hint in hints:
            if hint not in seen:
                seen.add(hint)
                deduped.append(hint)
        return deduped

    @staticmethod
    def is_repeated_tool_path_failure(
        debug_attempts: List[Dict[str, Any]], error_message: str
    ) -> bool:
        combined = str(error_message or "").lower()
        if not any(
            marker in combined
            for marker in (
                "raw_params",
                "wrong root",
                "absolute task-workspace path",
                "read failed: enoent",
                "read failed: eisdir",
                "exec failed: exec preflight",
            )
        ):
            return False

        prior_related = 0
        for attempt in debug_attempts:
            prior_text = " ".join(
                [
                    str(attempt.get("error", "")),
                    str(attempt.get("analysis", "")),
                    str(attempt.get("fix", "")),
                ]
            ).lower()
            if any(
                marker in prior_text
                for marker in (
                    "raw_params",
                    "absolute task-workspace path",
                    "read failed: enoent",
                    "read failed: eisdir",
                    "exec failed: exec preflight",
                )
            ):
                prior_related += 1
        return prior_related >= 2

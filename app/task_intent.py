"""Small, durable task-intent vocabulary used by planning admission."""

from __future__ import annotations

from enum import Enum
from typing import Any


class TaskIntentMode(str, Enum):
    """The only task distinction needed before source discovery."""

    DEFAULT = "default"
    CREATE_ONLY = "create_only"


def normalize_task_intent(value: Any) -> str:
    """Normalize persisted/API values without allowing a new capability."""

    if isinstance(value, TaskIntentMode):
        return value.value
    return (
        TaskIntentMode.CREATE_ONLY.value
        if str(value or "").strip().lower() == TaskIntentMode.CREATE_ONLY.value
        else TaskIntentMode.DEFAULT.value
    )


def render_create_only_guidance(value: Any) -> str:
    """Return concise product-level guidance for a CREATE_ONLY task."""

    if normalize_task_intent(value) != TaskIntentMode.CREATE_ONLY.value:
        return ""
    return (
        "Task constraint: This task is creation-only. You may propose new "
        "project-relative files as needed, but must not modify or delete "
        "existing project files. Every created path must be listed in "
        "expected_files and all normal verification, containment, and Plan "
        "rules remain active."
    )

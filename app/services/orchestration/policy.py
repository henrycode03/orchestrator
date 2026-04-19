"""Centralized orchestration policy knobs and thresholds."""

from __future__ import annotations

PLANNING_TIMEOUT_MIN_SECONDS = 180
PLANNING_TIMEOUT_MAX_SECONDS = 300
STALE_RUN_GUARD_SECONDS = 300
MAX_STEP_ATTEMPTS = 3
DEBUG_TIMEOUT_SECONDS = 180
SUMMARY_TIMEOUT_SECONDS = 60


def clamp_planning_timeout(timeout_seconds: int) -> int:
    """Bound planning time so dense tasks fail faster and more predictably."""

    return max(
        PLANNING_TIMEOUT_MIN_SECONDS,
        min(timeout_seconds, PLANNING_TIMEOUT_MAX_SECONDS),
    )

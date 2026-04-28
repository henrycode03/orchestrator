from app.services.orchestration.planner import PlannerService
from app.services.orchestration.policy import (
    MINIMAL_PLANNING_TIMEOUT_SECONDS,
    ORCHESTRATION_TASK_SOFT_TIME_LIMIT_SECONDS,
    ORCHESTRATION_TASK_TIME_LIMIT_SECONDS,
    PLANNING_REPAIR_TIMEOUT_SECONDS,
    ULTRA_MINIMAL_PLANNING_TIMEOUT_SECONDS,
)
from app.tasks.worker import execute_orchestration_task


def test_build_task_with_clean_architecture_does_not_start_minimal_first():
    assert (
        PlannerService.should_start_with_minimal_prompt(
            "Set up frontend (React or Vite) and backend (Node.js or FastAPI) with clean architecture.",
            "",
        )
        is False
    )


def test_true_inspection_task_still_starts_minimal_first():
    assert (
        PlannerService.should_start_with_minimal_prompt(
            "Inspect current project structure and review architecture before changes.",
            "",
        )
        is True
    )


def test_planning_fallback_timeouts_are_relaxed_for_local_models():
    assert MINIMAL_PLANNING_TIMEOUT_SECONDS == 300
    assert PLANNING_REPAIR_TIMEOUT_SECONDS == 300
    assert ULTRA_MINIMAL_PLANNING_TIMEOUT_SECONDS == 240


def test_worker_soft_time_limit_allows_planning_retries_and_execution_headroom():
    assert ORCHESTRATION_TASK_SOFT_TIME_LIMIT_SECONDS == 3300
    assert ORCHESTRATION_TASK_TIME_LIMIT_SECONDS == 3600
    assert execute_orchestration_task.soft_time_limit == 3300
    assert execute_orchestration_task.time_limit == 3600

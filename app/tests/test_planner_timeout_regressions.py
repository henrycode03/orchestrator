from app.services.orchestration.planner import PlannerService
from app.services.orchestration.planning_flow import (
    _should_repair_truncated_single_step_plan,
)
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


def test_qwen_local_prompt_profile_enforces_array_only_output():
    profile = PlannerService.select_prompt_profile("local_openclaw", "qwen-local")
    prompt = PlannerService.build_minimal_planning_prompt(
        "Build a hiring platform",
        project_dir=__import__("pathlib").Path("/tmp/project"),
        prompt_profile=profile,
    )

    assert profile == "local_qwen_json_array"
    assert "first non-whitespace character must be `[`" in prompt
    assert "Do not wrap it in an object" in prompt


def test_planning_repair_prompt_forbids_duplicated_workspace_roots():
    prompt = PlannerService.build_planning_repair_prompt(
        "Build frontend and backend scaffolding",
        malformed_output='[{"step_number":1,"commands":["mkdir -p frontend/src/frontend/src"]}]',
        project_dir=__import__("pathlib").Path("/tmp/project"),
    )

    assert "frontend/src/frontend/src" in prompt
    assert "backend/src/backend/src" in prompt
    assert "rooted exactly once" in prompt


def test_local_qwen_single_step_plan_is_routed_to_repair():
    assert (
        _should_repair_truncated_single_step_plan(
            prompt_profile="local_qwen_json_array",
            execution_profile="full_lifecycle",
            extracted_plan=[
                {
                    "step_number": 1,
                    "description": "Set up frontend and backend foundations",
                    "commands": ["mkdir -p frontend backend"],
                    "verification": "test -d frontend && test -d backend",
                    "rollback": "rm -rf frontend backend",
                    "expected_files": ["frontend/src/main.tsx", "backend/src/index.ts"],
                }
            ],
        )
        is True
    )


def test_non_qwen_or_non_full_lifecycle_single_step_plan_still_uses_retry_guard():
    single_step_plan = [
        {
            "step_number": 1,
            "description": "Do work",
            "commands": ["echo hi"],
            "verification": "test -n hi",
            "rollback": "true",
            "expected_files": [],
        }
    ]

    assert (
        _should_repair_truncated_single_step_plan(
            prompt_profile="default",
            execution_profile="full_lifecycle",
            extracted_plan=single_step_plan,
        )
        is False
    )
    assert (
        _should_repair_truncated_single_step_plan(
            prompt_profile="local_qwen_json_array",
            execution_profile="review_only",
            extracted_plan=single_step_plan,
        )
        is False
    )

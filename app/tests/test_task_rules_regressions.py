from app.services.orchestration.task_rules import (
    get_workflow_profile,
    should_force_review_execution_profile,
)


def test_force_review_profile_for_true_inspection_task():
    assert (
        should_force_review_execution_profile(
            "full_lifecycle",
            "Inspect current project architecture and inventory extension points.",
            "Inspect current project architecture",
            "Review the real files before implementation.",
        )
        is True
    )


def test_do_not_force_review_profile_for_build_task_with_clean_architecture():
    assert (
        should_force_review_execution_profile(
            "full_lifecycle",
            "Set up frontend (React or Vite) and backend (Node.js or FastAPI) with clean architecture.",
            "SkillSync AI Hiring Platform",
            "Set up frontend (React or Vite) and backend (Node.js or FastAPI) with clean architecture.",
        )
        is False
    )


def test_fullstack_scaffold_task_resolves_workflow_profile():
    assert (
        get_workflow_profile(
            "full_lifecycle",
            "SkillSync AI Hiring Platform",
            "Set up frontend (React or Vite) and backend (FastAPI) with clean architecture.",
        )
        == "fullstack_scaffold"
    )

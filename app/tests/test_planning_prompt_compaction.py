from __future__ import annotations

from app.models import PlanningMessage, PlanningSession
from app.services.openclaw_service import OpenClawSessionService
from app.services.planning_session_service import PlanningSessionService


def test_context_overflow_detection_matches_embedded_agent_variant():
    assert OpenClawSessionService._is_context_overflow_result(
        {"error": "Context size has been exceeded."}
    )
    assert OpenClawSessionService._is_context_overflow_result(
        {"output": "Prompt is too long for the model."}
    )
    assert not OpenClawSessionService._is_context_overflow_result(
        {"error": "Connection refused"}
    )


def test_synthesis_prompt_is_compacted_for_long_transcripts():
    service = PlanningSessionService(db=None)  # type: ignore[arg-type]
    session = PlanningSession(
        id=1,
        project_id=1,
        title="Long planning session",
        prompt="Build an execution-ready plan for a large dashboard rewrite with auth, background jobs, logs, websockets, reporting, settings, and migration safety.",
        status="active",
        source_brain="local",
    )
    session.messages = [
        PlanningMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=("very long planning detail " * 40) + f" message {index}",
        )
        for index in range(8)
    ]
    project = type(
        "ProjectStub",
        (),
        {"name": "Big Project", "description": "existing project " * 80},
    )()

    prompt = service._build_synthesis_prompt(session, project)

    assert len(prompt) <= service.SYNTHESIS_PROMPT_CHAR_BUDGET + 200
    assert "Conversation transcript:" in prompt
    assert prompt.count("Planner:") + prompt.count("User:") <= 6

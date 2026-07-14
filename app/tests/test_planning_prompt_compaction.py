from __future__ import annotations

from app.config import settings
from app.models import SystemSetting
from app.models import PlanningMessage, PlanningSession
from app.services.agents.openclaw_service import OpenClawSessionService
from app.services.agents.runtime_configuration import RuntimeConfiguration
from app.services.planning.planning_session_service import PlanningSessionService
from app.services.workspace.system_settings import (
    ADAPTATION_PROFILE_KEY,
    set_setting_value,
)

PLANNING_ADAPTATION_PROFILE_KEY = "orchestrator_planning_adaptation_profile"


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
        {
            "name": "Big Project",
            "description": "existing project " * 80,
            "project_rules": None,
        },
    )()

    prompt = service._build_synthesis_prompt(session, project)

    assert len(prompt) <= service.SYNTHESIS_PROMPT_CHAR_BUDGET + 200
    assert "Conversation transcript:" in prompt
    assert prompt.count("Planner:") + prompt.count("User:") <= 6


def test_synthesis_prompt_uses_active_adaptation_profile(monkeypatch):
    service = PlanningSessionService(db=None)  # type: ignore[arg-type]
    session = PlanningSession(
        id=1,
        project_id=1,
        title="Adapted planning session",
        prompt="Add a resilient auth flow with better session recovery.",
        status="active",
        source_brain="local",
    )
    session.messages = [PlanningMessage(role="user", content=session.prompt)]
    project = type(
        "ProjectStub",
        (),
        {
            "name": "Adapted Project",
            "description": "Backend and frontend auth work.",
            "project_rules": None,
        },
    )()

    monkeypatch.setattr(
        "app.services.planning.planning_session_service.resolve_planning_runtime_configuration",
        lambda db: RuntimeConfiguration(
            role="planning",
            backend_name="planning-backend",
            model_family="planning-model",
            adaptation_profile="openai_responses_default",
        ),
    )

    prompt = service._build_synthesis_prompt(session, project)

    assert prompt.startswith('{"objective":"Create implementation-planning artifacts')
    assert '"execution_mode":"planning_synthesis"' in prompt
    assert '"context":{"Project":"Adapted Project"' in prompt


def test_synthesis_prompt_uses_planning_role_profile(db_session, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "PLANNING_BACKEND", "openai_responses_api")
    monkeypatch.setattr(settings, "PLANNER_MODEL", "gpt-5")
    set_setting_value(db_session, ADAPTATION_PROFILE_KEY, "openclaw_default")
    set_setting_value(
        db_session,
        PLANNING_ADAPTATION_PROFILE_KEY,
        "openai_responses_default",
    )
    service = PlanningSessionService(db_session)

    prompt = service._render_adapted_prompt(
        objective="Create a plan",
        execution_mode="planning_synthesis",
        instructions=["Return a structured plan."],
        context={"Project": "Role-specific planning"},
        expected_output="A plan",
    )

    assert prompt.startswith('{"objective":"Create a plan"')
    assert '"execution_mode":"planning_synthesis"' in prompt


def test_clarification_payload_parser_uses_model_decision():
    service = PlanningSessionService(db=None)  # type: ignore[arg-type]

    parsed = service._parse_clarification_payload(
        {
            "status": "completed",
            "output": '{"needs_clarification": true, "question": "Which rollout constraints matter most?"}',
        },
        fallback_needs=False,
        fallback_question="Fallback question",
    )

    assert parsed == {
        "needs_clarification": True,
        "question": "Which rollout constraints matter most?",
    }


def test_clarification_payload_parser_falls_back_on_invalid_json():
    service = PlanningSessionService(db=None)  # type: ignore[arg-type]

    parsed = service._parse_clarification_payload(
        {"status": "completed", "output": "not-json"},
        fallback_needs=True,
        fallback_question="Fallback question",
    )

    assert parsed == {
        "needs_clarification": True,
        "question": "Fallback question",
    }


def test_synthesis_prompt_uses_db_selected_adaptation_profile(db_session):
    db_session.add(
        SystemSetting(
            key="orchestrator_adaptation_profile",
            value="openai_responses_default",
        )
    )
    db_session.commit()

    service = PlanningSessionService(db=db_session)
    session = PlanningSession(
        id=2,
        project_id=1,
        title="DB adapted planning session",
        prompt="Improve retries and diagnostics.",
        status="active",
        source_brain="local",
    )
    session.messages = [PlanningMessage(role="user", content=session.prompt)]
    project = type(
        "ProjectStub",
        (),
        {
            "name": "DB Adapted Project",
            "description": "Queue and worker cleanup.",
            "project_rules": None,
        },
    )()

    prompt = service._build_synthesis_prompt(session, project)

    assert prompt.startswith('{"objective":"Create implementation-planning artifacts')

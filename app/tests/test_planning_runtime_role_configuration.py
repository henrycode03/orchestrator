"""End-to-end ownership checks for the provider-neutral Planning Runtime."""

from __future__ import annotations

import pytest

from app.config import settings
from app.models import Project
from app.services.agents.agent_runtime import (
    BackendRole,
    create_agent_runtime,
    resolve_planning_runtime_configuration,
)
from app.services.agents.openclaw_service import OpenClawSessionService
from app.services.agents.providers.ollama_adapter import OllamaRuntime
from app.services.agents.providers.openai_adapter import OpenAIResponsesRuntime
from app.services.agents.providers.openai_chat_adapter import (
    OpenAIChatCompletionsRuntime,
)
from app.services.model_adaptation import render_prompt_for_profile
from app.services.model_adaptation.schemas import PromptEnvelope
from app.services.observability.build_identity import build_identity_payload
from app.services.observability.planning_identity import _fingerprint
from app.services.planning.planning_session_service import PlanningSessionService
from app.services.workspace.system_settings import (
    ADAPTATION_PROFILE_KEY,
    AGENT_BACKEND_KEY,
    PLANNING_ADAPTATION_PROFILE_KEY,
    set_setting_value,
)


@pytest.mark.parametrize(
    ("planning_backend", "planner_model", "planning_profile", "adapter_type"),
    [
        ("direct_ollama", "planner-model-a", "planning_default", OllamaRuntime),
        (
            "openai_responses_api",
            "planner-model-b",
            "openai_responses_default",
            OpenAIResponsesRuntime,
        ),
    ],
)
def test_planning_role_configuration_owns_every_planning_surface(
    db_session,
    monkeypatch,
    planning_backend,
    planner_model,
    planning_profile,
    adapter_type,
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "PLANNING_BACKEND", planning_backend)
    monkeypatch.setattr(settings, "EXECUTION_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "PLANNER_MODEL", planner_model)
    set_setting_value(db_session, AGENT_BACKEND_KEY, "local_openclaw")
    set_setting_value(db_session, ADAPTATION_PROFILE_KEY, "openclaw_default")
    set_setting_value(
        db_session,
        PLANNING_ADAPTATION_PROFILE_KEY,
        planning_profile,
    )
    project = Project(
        name=f"planning-role-{planning_backend}",
        workspace_path=f"planning-role-{planning_backend}",
    )
    db_session.add(project)
    db_session.commit()
    monkeypatch.setattr(PlanningSessionService, "schedule_processing", lambda *_: None)

    service = PlanningSessionService(db_session)
    configuration = resolve_planning_runtime_configuration(db_session)
    planning_session = service.start_session(project, "Create a bounded plan")
    planning_runtime = create_agent_runtime(
        db_session,
        session_id=None,
        role=BackendRole.PLANNING,
    )
    execution_runtime = create_agent_runtime(
        db_session,
        session_id=None,
        role=BackendRole.EXECUTION,
    )
    envelope = PromptEnvelope(
        objective="Create a bounded plan",
        execution_mode="planning_synthesis",
        instructions=["Return the plan."],
        context={"Project": project.name},
        expected_output="A plan",
    )
    rendered_prompt = service._render_adapted_prompt(
        objective=envelope.objective,
        execution_mode=envelope.execution_mode,
        instructions=envelope.instructions,
        context=envelope.context,
        expected_output=envelope.expected_output,
    )
    identity_payload = build_identity_payload(
        db_session,
        planning_configuration=configuration,
    )
    runtime_metadata = planning_runtime.get_backend_metadata()

    assert configuration.role == BackendRole.PLANNING.value
    assert configuration.backend_name == planning_backend
    assert configuration.model_family == planner_model
    assert configuration.adaptation_profile == planning_profile
    assert planning_session.planning_backend == planning_backend
    assert planning_session.planner_model == planner_model
    assert planning_session.reasoning_profile == planning_profile
    assert planning_session.configuration_fingerprint == _fingerprint(
        identity_payload,
        planning_profile,
    )
    assert rendered_prompt == render_prompt_for_profile(planning_profile, envelope)
    assert isinstance(planning_runtime, adapter_type)
    assert runtime_metadata["backend"] == planning_backend
    assert runtime_metadata["model_family"] == planner_model
    assert runtime_metadata["adaptation_profile"] == planning_profile
    assert isinstance(execution_runtime, OpenClawSessionService)


def test_non_role_adapter_metadata_keeps_legacy_interface_shape(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "direct_ollama")
    direct_runtime = create_agent_runtime(db_session, session_id=None)
    direct_metadata = direct_runtime.get_backend_metadata()
    direct_interface = direct_runtime.describe_interface()

    monkeypatch.setattr(settings, "AGENT_BACKEND", "openai_chat_completions")
    chat_runtime = create_agent_runtime(db_session, session_id=None)
    chat_metadata = chat_runtime.get_backend_metadata()
    chat_interface = chat_runtime.describe_interface()

    assert isinstance(direct_runtime, OllamaRuntime)
    assert "adaptation_profile" not in direct_metadata
    assert direct_interface.prompt_dialect == "ollama_chat"
    assert direct_interface.tool_shape == "none"
    assert (
        direct_interface.context_window_policy.compaction_strategy == "truncate_context"
    )
    assert isinstance(chat_runtime, OpenAIChatCompletionsRuntime)
    assert "adaptation_profile" not in chat_metadata
    assert chat_interface.prompt_dialect == "openai_chat_completions"
    assert chat_interface.tool_shape == "none"
    assert (
        chat_interface.context_window_policy.compaction_strategy == "truncate_context"
    )

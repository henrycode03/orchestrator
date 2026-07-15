"""Focused regressions for Phase 26D-2 planning runtime hardening."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import settings
from app.services.agents.agent_runtime import (
    BackendRole,
    _diagnostic_category,
    invoke_runtime_prompt,
    resolve_planning_runtime_configuration,
)
from app.services.agents.interfaces import AgentRuntimeError
from app.services.agents.runtime_configuration import RoleRuntimeConfiguration
from app.services.planning.planning_session_service import PlanningSessionService
from app.services.workspace.system_settings import (
    ADAPTATION_PROFILE_KEY,
    AGENT_BACKEND_KEY,
    AGENT_MODEL_FAMILY_KEY,
    PLANNING_ADAPTATION_PROFILE_KEY,
    PLANNING_MODEL_FAMILY_KEY,
    get_setting_value,
    set_setting_value,
)


def _configure_planning(db_session, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_BACKEND", "local_openclaw")
    monkeypatch.setattr(settings, "PLANNING_BACKEND", "direct_ollama")
    monkeypatch.setattr(settings, "PLANNER_MODEL", "")
    monkeypatch.setattr(settings, "AGENT_MODEL", "global-env-model")
    set_setting_value(db_session, AGENT_BACKEND_KEY, "local_openclaw")
    set_setting_value(db_session, AGENT_MODEL_FAMILY_KEY, "global-db-model")
    set_setting_value(db_session, ADAPTATION_PROFILE_KEY, "openclaw_default")
    set_setting_value(
        db_session,
        PLANNING_ADAPTATION_PROFILE_KEY,
        "planning_default",
    )


def test_persistent_planning_model_preserves_environment_precedence(
    db_session, monkeypatch
):
    _configure_planning(db_session, monkeypatch)
    set_setting_value(
        db_session,
        PLANNING_MODEL_FAMILY_KEY,
        "persistent-planning-model",
    )

    configuration = resolve_planning_runtime_configuration(db_session)
    assert configuration.model_family == "persistent-planning-model"

    monkeypatch.setattr(settings, "PLANNER_MODEL", "env-planning-model")
    assert (
        resolve_planning_runtime_configuration(db_session).model_family
        == "env-planning-model"
    )


def test_settings_api_persists_planning_model(
    authenticated_client, db_session, monkeypatch
):
    monkeypatch.setattr(settings, "PLANNER_MODEL", "")
    response = authenticated_client.patch(
        "/api/v1/settings/system",
        json={"planning_model_family": "operator-planning-model"},
    )

    assert response.status_code == 200
    assert response.json()["system"]["planning_model_family"] == (
        "operator-planning-model"
    )
    assert (
        get_setting_value(db_session, PLANNING_MODEL_FAMILY_KEY)
        == "operator-planning-model"
    )


def test_planning_runtime_forwards_bounded_no_output_timeout(db_session, monkeypatch):
    captured = {}

    def fake_invoke_runtime_prompt(db, prompt, **kwargs):
        captured.update(kwargs)
        return {"status": "completed", "output": "{}"}

    monkeypatch.setattr(
        "app.services.planning.planning_session_service.invoke_runtime_prompt",
        fake_invoke_runtime_prompt,
    )
    service = PlanningSessionService(db_session)

    service._run_openclaw(
        "Return JSON",
        source_brain="local",
        timeout_seconds=37,
        project_id=42,
    )

    assert captured["timeout_seconds"] == 37
    assert captured["no_output_timeout_seconds"] == 37
    assert captured["role"] is BackendRole.PLANNING


def test_runtime_diagnostics_classify_silent_timeout_provider_and_slow_paths():
    assert _diagnostic_category({"no_output_timeout": True}) == "silent_inference"
    assert _diagnostic_category({"timed_out": True}) == "timeout"
    assert _diagnostic_category({}, error=AgentRuntimeError("provider failed")) == (
        "provider_failure"
    )
    assert _diagnostic_category({"first_output_after_seconds": 8}) == ("slow_inference")


def test_runtime_invocation_exposes_secret_free_planning_diagnostics(
    db_session, monkeypatch, caplog
):
    captured = {}
    configuration = RoleRuntimeConfiguration(
        role=BackendRole.PLANNING,
        backend_name="test-provider",
        model_family="test-model",
        adaptation_profile="test-profile",
    )

    class FakeRuntime:
        backend_descriptor = SimpleNamespace(name="test-provider")
        runtime_configuration = configuration

        async def invoke_prompt(self, prompt, **kwargs):
            captured.update(kwargs)
            return {
                "status": "completed",
                "output": "{}",
                "runtime_diagnostics": {"first_output_after_seconds": 8},
            }

    monkeypatch.setattr(
        "app.services.agents.agent_runtime.create_agent_runtime",
        lambda *args, **kwargs: FakeRuntime(),
    )

    with caplog.at_level("INFO"):
        result = invoke_runtime_prompt(
            db_session,
            "Return JSON",
            role=BackendRole.PLANNING,
            timeout_seconds=37,
            no_output_timeout_seconds=20,
        )

    assert captured["no_output_timeout_seconds"] == 20
    assert result["runtime_diagnostics"]["diagnostic_category"] == "slow_inference"
    assert result["runtime_diagnostics"]["model_family"] == "test-model"
    assert "Return JSON" not in caplog.text


def test_runtime_configuration_failure_is_classified_without_secrets(
    db_session, monkeypatch
):
    error = ValueError("planning profile has api_key=top-secret")
    monkeypatch.setattr(
        "app.services.agents.agent_runtime.create_agent_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ValueError) as exc_info:
        invoke_runtime_prompt(
            db_session,
            "Return JSON",
            role=BackendRole.PLANNING,
        )

    diagnostics = exc_info.value.runtime_diagnostics
    assert diagnostics["diagnostic_category"] == "configuration_failure"
    assert "top-secret" not in diagnostics["error"]

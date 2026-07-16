"""Phase 26E-5 Execution role authority regression coverage."""

import pytest
from types import SimpleNamespace

from app.config import settings
from app.models import Project, Session, Task
from app.services.agents.agent_runtime import (
    BackendRole,
    create_agent_runtime,
    resolve_runtime_configuration,
)
from app.services.observability.planning_identity import active_execution_identity
from app.services.observability.runtime_identity import (
    build_runtime_identity_projection,
)
from app.services.tasks.execution import create_task_execution
from app.services.workspace.system_settings import (
    ADAPTATION_PROFILE_KEY,
    AGENT_BACKEND_KEY,
    AGENT_MODEL_FAMILY_KEY,
    set_setting_value,
)

PLANNING_ADAPTATION_PROFILE_KEY = "orchestrator_planning_adaptation_profile"


def _configure_execution(
    db_session,
    monkeypatch,
    *,
    backend: str,
    global_model: str = "qwen3.6:27B",
    execution_model: str = "",
    ollama_model: str = "qwen3-coder:30b",
):
    monkeypatch.setattr(settings, "AGENT_BACKEND", backend)
    monkeypatch.setattr(settings, "EXECUTION_BACKEND", backend)
    monkeypatch.setattr(settings, "EXECUTION_MODEL", execution_model)
    monkeypatch.setattr(settings, "OLLAMA_AGENT_MODEL", ollama_model)
    monkeypatch.setattr(settings, "EXECUTION_ADAPTATION_PROFILE", None)
    set_setting_value(db_session, AGENT_BACKEND_KEY, backend)
    set_setting_value(db_session, AGENT_MODEL_FAMILY_KEY, global_model)
    set_setting_value(db_session, ADAPTATION_PROFILE_KEY, "openclaw_default")


def test_blank_execution_model_inherits_global_for_local_openclaw(
    db_session, monkeypatch
):
    _configure_execution(db_session, monkeypatch, backend="local_openclaw")

    configuration = resolve_runtime_configuration(db_session, BackendRole.EXECUTION)

    assert configuration.model_family == "qwen3.6:27B"


def test_local_openclaw_execution_inherits_global_profile_without_qwen_inference(
    db_session, monkeypatch
):
    _configure_execution(db_session, monkeypatch, backend="local_openclaw")

    configuration = resolve_runtime_configuration(db_session, BackendRole.EXECUTION)

    assert configuration.adaptation_profile == "openclaw_default"


def test_execution_readback_uses_the_authoritative_role_configuration(
    db_session, monkeypatch
):
    _configure_execution(db_session, monkeypatch, backend="local_openclaw")
    configuration = resolve_runtime_configuration(db_session, BackendRole.EXECUTION)

    identity = active_execution_identity(db_session)

    assert identity["execution_backend"] == configuration.backend_name
    assert identity["executor_model"] == configuration.model_family
    assert identity["execution_adaptation_profile"] == configuration.adaptation_profile


def test_task_execution_creation_persists_authoritative_execution_identity(
    db_session, monkeypatch
):
    _configure_execution(db_session, monkeypatch, backend="local_openclaw")
    project = Project(name="phase26e5-execution-authority")
    db_session.add(project)
    db_session.flush()
    task = Task(project_id=project.id, title="Persist Execution authority")
    session = Session(project_id=project.id, name="Execution authority")
    db_session.add_all([task, session])
    db_session.flush()
    configuration = resolve_runtime_configuration(db_session, BackendRole.EXECUTION)

    execution = create_task_execution(
        db_session,
        session_id=session.id,
        task_id=task.id,
    )

    assert execution.execution_backend == configuration.backend_name
    assert execution.executor_model == configuration.model_family

    projection = build_runtime_identity_projection(
        db_session,
        task_execution=execution,
    )
    assert projection.execution_backend == configuration.backend_name
    assert projection.executor_model == configuration.model_family
    assert projection.identity_sources["execution_backend"] == "stored_task_execution"
    assert projection.identity_sources["executor_model"] == "stored_task_execution"


@pytest.mark.parametrize(
    "backend",
    [
        "local_openclaw",
        "openai_responses_api",
        "openai_chat_completions",
    ],
)
def test_ollama_model_is_isolated_from_unrelated_execution_providers(
    db_session, monkeypatch, backend
):
    _configure_execution(db_session, monkeypatch, backend=backend)

    first_configuration = resolve_runtime_configuration(
        db_session, BackendRole.EXECUTION
    )
    first_identity = active_execution_identity(db_session)
    first_runtime = create_agent_runtime(
        db_session, session_id=None, role=BackendRole.EXECUTION
    )
    monkeypatch.setattr(settings, "OLLAMA_AGENT_MODEL", "unrelated-ollama-change")
    second_configuration = resolve_runtime_configuration(
        db_session, BackendRole.EXECUTION
    )
    second_identity = active_execution_identity(db_session)
    second_runtime = create_agent_runtime(
        db_session, session_id=None, role=BackendRole.EXECUTION
    )

    assert first_configuration == second_configuration
    assert first_identity == second_identity
    assert first_runtime.get_backend_metadata()["runtime_configuration"] == (
        second_runtime.get_backend_metadata()["runtime_configuration"]
    )


def test_planning_setting_drift_does_not_change_execution_fingerprint(
    db_session, monkeypatch
):
    _configure_execution(db_session, monkeypatch, backend="local_openclaw")
    monkeypatch.setattr(settings, "PLANNER_MODEL", "planning-model-one")
    first = active_execution_identity(db_session)

    monkeypatch.setattr(settings, "PLANNER_MODEL", "planning-model-two")
    set_setting_value(
        db_session,
        PLANNING_ADAPTATION_PROFILE_KEY,
        "qwen_compact_json",
    )
    second = active_execution_identity(db_session)

    for field_name in (
        "execution_backend",
        "executor_model",
        "execution_adaptation_profile",
        "configuration_fingerprint",
    ):
        assert first[field_name] == second[field_name]


def test_direct_ollama_keeps_ollama_model_fallback_through_factory_and_readback(
    db_session, monkeypatch
):
    _configure_execution(db_session, monkeypatch, backend="direct_ollama")

    configuration = resolve_runtime_configuration(db_session, BackendRole.EXECUTION)
    identity = active_execution_identity(db_session)
    runtime = create_agent_runtime(
        db_session, session_id=None, role=BackendRole.EXECUTION
    )
    metadata = runtime.get_backend_metadata()

    assert configuration.model_family == "qwen3-coder:30b"
    assert configuration.adaptation_profile == "ollama_default"
    assert identity["executor_model"] == configuration.model_family
    assert identity["execution_adaptation_profile"] == (
        configuration.adaptation_profile
    )
    assert metadata["model_family"] == configuration.model_family
    assert metadata["adaptation_profile"] == configuration.adaptation_profile


def test_explicit_local_openclaw_execution_model_and_profile_remain_highest_priority(
    db_session, monkeypatch
):
    _configure_execution(
        db_session,
        monkeypatch,
        backend="local_openclaw",
        execution_model="explicit-execution-model",
    )
    monkeypatch.setattr(
        settings,
        "EXECUTION_ADAPTATION_PROFILE",
        "qwen_compact_json",
    )

    configuration = resolve_runtime_configuration(db_session, BackendRole.EXECUTION)
    identity = active_execution_identity(db_session)
    runtime = create_agent_runtime(
        db_session, session_id=None, role=BackendRole.EXECUTION
    )

    assert configuration.model_family == "explicit-execution-model"
    assert configuration.adaptation_profile == "qwen_compact_json"
    assert identity["executor_model"] == configuration.model_family
    assert identity["execution_adaptation_profile"] == (
        configuration.adaptation_profile
    )
    assert runtime.get_backend_metadata()["runtime_configuration"] == (
        configuration.to_dict()
    )


def test_legacy_global_fallback_does_not_leak_ollama_model_to_local_openclaw(
    db_session, monkeypatch
):
    _configure_execution(db_session, monkeypatch, backend="local_openclaw")
    monkeypatch.setattr(
        settings,
        "EXECUTION_ADAPTATION_PROFILE",
        "ollama_default",
    )
    legacy_execution = SimpleNamespace(
        id=91,
        planning_session_id=None,
        planning_backend="historical-planning",
        planner_model="historical-planner",
        reasoning_profile="historical-profile",
        configuration_fingerprint="a" * 64,
        execution_backend=None,
        executor_model=None,
    )

    projection = build_runtime_identity_projection(
        db_session,
        task_execution=legacy_execution,
    )

    assert projection.execution_backend == "local_openclaw"
    assert projection.executor_model == "qwen3.6:27B"
    assert projection.identity_sources["executor_model"] == "legacy_global_fallback"

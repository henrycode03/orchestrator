"""POST33-ROUTE2 regressions for the bounded planning role routing repair.

ROUTE1-D1: ``OpenAIChatCompletionsRuntime.execute_task`` selected its system
contract from ``diagnostic_label`` prose, so the production Discovery label
``PLANNING_DISCOVERY`` -- the only planning-family label not ending in
"PLANNING" -- received the step-execution contract.

ROUTE1-D2: ``_base_url``/``_api_key`` were role-scoped only for the three
repair roles, so a planning-role direct runtime fell through to the generic
``OPENAI_*`` settings and could address the public OpenAI API.

Every test here is provider-free: the outbound request is captured at the
httpx boundary and no network call is made.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import settings
from app.services.agents.agent_backends import (
    ExecutionTopology,
    get_backend_descriptor,
    require_backend_descriptor,
)
from app.services.agents.agent_runtime import (
    RuntimeCapabilityError,
    UnsupportedRuntimeProfileError,
    resolve_runtime_configuration,
    validate_runtime_capabilities,
    validate_runtime_provider_contract,
)
from app.services.agents.interfaces import AgentRuntimeError
from app.services.agents.providers import openai_chat_adapter
from app.services.agents.providers.openai_chat_adapter import (
    _GENERIC_SYSTEM,
    _STEP_SYSTEM,
    OpenAIChatCompletionsRuntime,
)
from app.services.agents.runtime_configuration import (
    BackendRole,
    RoleRuntimeConfiguration,
)
from app.services.orchestration.planning.read_only_discovery import (
    build_discovery_prompt,
)

PLANNING_GATEWAY = "http://ai-gateway.test:8000/v1"
PLANNING_MODEL = "qwen-local"
PLANNING_PROFILE = "ollama_default"

# The exact label run_discovery_stage() sends. It must never select the step
# contract, and it must not be special-cased by string in production code.
DISCOVERY_LABEL = "PLANNING_DISCOVERY"
FINAL_PLANNING_LABEL = "PLANNING"


def _runtime(db, role: BackendRole, *, model: str = PLANNING_MODEL):
    configuration = RoleRuntimeConfiguration(
        role=role,
        backend_name="openai_chat_completions",
        model_family=model,
        adaptation_profile=PLANNING_PROFILE,
    )
    return OpenAIChatCompletionsRuntime(
        db, session_id=None, runtime_configuration=configuration
    )


def _planning_endpoint_configured(monkeypatch):
    """Configure a local planning gateway plus hostile generic OpenAI values."""

    monkeypatch.setattr(settings, "PLANNING_DIRECT_BASE_URL", PLANNING_GATEWAY)
    monkeypatch.setattr(settings, "PLANNING_DIRECT_API_KEY", "")
    # If the planning role ever falls back to these, the assertions below fail.
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-placeholder-must-not-be-used")
    monkeypatch.setattr(settings, "OPENAI_CHAT_COMPLETIONS_BASE_URL", "")
    monkeypatch.setattr(settings, "OPENAI_CHAT_COMPLETIONS_API_KEY", "")
    monkeypatch.setattr(settings, "OPENAI_CHAT_COMPLETIONS_MODEL", "wrong-model")
    monkeypatch.setattr(settings, "OPENAI_CHAT_COMPLETIONS_TEMPERATURE", 0.0)
    monkeypatch.setattr(settings, "OPENAI_CHAT_COMPLETIONS_TOP_P", None)
    monkeypatch.setattr(settings, "OPENAI_CHAT_COMPLETIONS_REPEAT_PENALTY", None)


def _capture_dispatch(monkeypatch, response_content: str = '{"action":"stop"}'):
    """Capture the outbound request immediately before network dispatch."""

    captured: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": response_content}}]}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            captured.update({"url": url, "headers": dict(headers or {}), "json": json})
            return _FakeResponse()

    monkeypatch.setattr(openai_chat_adapter.httpx, "AsyncClient", _FakeAsyncClient)
    return captured


# ---------------------------------------------------------------------------
# ROUTE1-D1 -- invocation classification comes from role ownership
# ---------------------------------------------------------------------------


def test_production_shaped_direct_discovery_replay_uses_planning_contract(
    db_session, monkeypatch
):
    """The most important ROUTE2 regression.

    MODEL4's direct arm exercised invoke_prompt(), which is unconditionally
    generic. Production Discovery goes through execute_task(), which is where
    D1 lived. This replays the production shape end to end.
    """

    _planning_endpoint_configured(monkeypatch)
    captured = _capture_dispatch(monkeypatch)

    canonical_prompt = build_discovery_prompt(
        "Add a bounded retry to the planning repair lane.",
        "Existing planning context.",
    )
    runtime = _runtime(db_session, BackendRole.PLANNING)

    asyncio.run(
        runtime.execute_task(
            canonical_prompt,
            timeout_seconds=120,
            # The exact kwargs run_discovery_stage() passes through
            # PlannerService._execute_task_with_planning_lock.
            reuse_task_session=False,
            diagnostic_label=DISCOVERY_LABEL,
            diagnostic_metadata={"stage": "read_only_discovery"},
        )
    )

    payload = captured["json"]
    system_messages = [m for m in payload["messages"] if m["role"] == "system"]
    user_messages = [m for m in payload["messages"] if m["role"] == "user"]

    # Planning/generic system contract, and the step contract is absent.
    assert system_messages == [{"role": "system", "content": _GENERIC_SYSTEM}]
    assert _STEP_SYSTEM not in str(payload)
    assert "Execute the given step exactly as described" not in str(payload)

    # Canonical discovery user prompt is unchanged, byte for byte.
    assert user_messages == [{"role": "user", "content": canonical_prompt}]
    assert "Return exactly one JSON object and no prose" in canonical_prompt

    # No tools field is introduced.
    assert "tools" not in payload
    assert "tool_choice" not in payload

    # Role-scoped planning endpoint and planning model.
    assert captured["url"] == f"{PLANNING_GATEWAY}/chat/completions"
    assert "api.openai.com" not in captured["url"]
    assert payload["model"] == PLANNING_MODEL

    # No OpenClaw runtime, bootstrap, or system envelope is introduced.
    serialized = str(payload).lower()
    for marker in ("openclaw", "soul.md", "agents.md", "tools.md", "bootstrap"):
        assert marker not in serialized


def test_production_shaped_direct_final_planning_replay_matches_discovery(
    db_session, monkeypatch
):
    _planning_endpoint_configured(monkeypatch)
    captured = _capture_dispatch(monkeypatch, response_content="[]")

    runtime = _runtime(db_session, BackendRole.PLANNING)
    asyncio.run(
        runtime.execute_task(
            "Return the plan as a JSON array.",
            timeout_seconds=180,
            reuse_task_session=False,
            diagnostic_label=FINAL_PLANNING_LABEL,
            diagnostic_metadata={"planning_attempt": "initial"},
        )
    )

    payload = captured["json"]
    assert {"role": "system", "content": _GENERIC_SYSTEM} in payload["messages"]
    assert _STEP_SYSTEM not in str(payload)
    # Same endpoint, model, and profile as the Discovery replay above.
    assert captured["url"] == f"{PLANNING_GATEWAY}/chat/completions"
    assert payload["model"] == PLANNING_MODEL
    assert runtime.runtime_configuration.adaptation_profile == PLANNING_PROFILE
    assert "tools" not in payload
    assert "openclaw" not in str(payload).lower()


def test_planning_role_cannot_receive_step_contract_for_any_label(db_session):
    """Role ownership, not the label, decides the contract."""

    runtime = _runtime(db_session, BackendRole.PLANNING)
    for label in (
        DISCOVERY_LABEL,
        FINAL_PLANNING_LABEL,
        "MINIMAL_PLANNING",
        "ULTRA_MINIMAL_PLANNING",
        None,
        "SOMETHING_ELSE",
    ):
        assert runtime._execute_task_system_prompt(label, None) is _GENERIC_SYSTEM


def test_execution_role_still_receives_step_contract(db_session):
    runtime = _runtime(db_session, BackendRole.EXECUTION, model="local")
    assert runtime._execute_task_system_prompt("STEP", None) is _STEP_SYSTEM
    # Even a planning-shaped label cannot make execution reasoning-shaped.
    assert runtime._execute_task_system_prompt(FINAL_PLANNING_LABEL, None) is (
        _STEP_SYSTEM
    )


@pytest.mark.parametrize(
    "role",
    [BackendRole.REPAIR, BackendRole.DEBUG_REPAIR, BackendRole.COMPLETION_REPAIR],
)
def test_repair_roles_keep_existing_execute_task_contract(db_session, role):
    """Repair behavior on execute_task is preserved exactly.

    execute_task() is reached by repair only to repair step-shaped output;
    reasoning-shaped repair uses invoke_prompt(), which is unconditionally
    generic. Broadening repair here would be an unrequested behavior change.
    """

    runtime = _runtime(db_session, role)
    assert runtime._execute_task_system_prompt(None, None) is _STEP_SYSTEM
    assert (
        runtime._execute_task_system_prompt("BOUNDED_EXECUTION_DEBUG_REPAIR", None)
        is _STEP_SYSTEM
    )


def test_role_less_legacy_callers_keep_the_label_heuristic(db_session):
    runtime = OpenAIChatCompletionsRuntime(db_session, session_id=None)
    assert runtime.backend_role is None
    assert runtime._execute_task_system_prompt("MINIMAL_PLANNING", None) is (
        _GENERIC_SYSTEM
    )
    assert (
        runtime._execute_task_system_prompt(None, {"planning_attempt": "initial"})
        is _GENERIC_SYSTEM
    )
    assert runtime._execute_task_system_prompt(None, None) is _STEP_SYSTEM


def test_invoke_prompt_remains_unconditionally_generic(db_session, monkeypatch):
    _planning_endpoint_configured(monkeypatch)
    captured = _capture_dispatch(monkeypatch)
    runtime = _runtime(db_session, BackendRole.PLANNING)

    asyncio.run(runtime.invoke_prompt("probe", timeout_seconds=30))

    assert {"role": "system", "content": _GENERIC_SYSTEM} in captured["json"][
        "messages"
    ]


# ---------------------------------------------------------------------------
# ROUTE1-D2 -- role-scoped planning endpoint resolution
# ---------------------------------------------------------------------------


def test_planning_endpoint_fails_closed_when_unconfigured(db_session, monkeypatch):
    monkeypatch.setattr(settings, "PLANNING_DIRECT_BASE_URL", "")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-placeholder-must-not-be-used")

    runtime = _runtime(db_session, BackendRole.PLANNING)
    with pytest.raises(AgentRuntimeError) as excinfo:
        runtime._base_url  # noqa: B018 -- property access is the assertion

    assert "PLANNING_DIRECT_BASE_URL" in str(excinfo.value)


def test_planning_never_falls_back_to_generic_openai_settings(db_session, monkeypatch):
    _planning_endpoint_configured(monkeypatch)
    # Even a populated generic chat endpoint must not win for planning.
    monkeypatch.setattr(
        settings, "OPENAI_CHAT_COMPLETIONS_BASE_URL", "https://generic.invalid/v1"
    )
    monkeypatch.setattr(settings, "OPENAI_CHAT_COMPLETIONS_API_KEY", "generic-key")

    runtime = _runtime(db_session, BackendRole.PLANNING)

    assert runtime._base_url == PLANNING_GATEWAY
    assert runtime._api_key() == ""
    assert runtime._invocation_base_url(None) == PLANNING_GATEWAY
    assert runtime._invocation_api_key(None) == ""


def test_planning_api_key_uses_planning_owned_value(db_session, monkeypatch):
    _planning_endpoint_configured(monkeypatch)
    monkeypatch.setattr(settings, "PLANNING_DIRECT_API_KEY", "planning-dummy")

    runtime = _runtime(db_session, BackendRole.PLANNING)
    assert runtime._api_key() == "planning-dummy"


def test_repair_roles_are_not_redirected_through_planning_settings(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "PLANNING_DIRECT_BASE_URL", PLANNING_GATEWAY)
    monkeypatch.setattr(settings, "PLANNING_DIRECT_API_KEY", "planning-dummy")
    monkeypatch.setattr(
        settings, "PLANNING_REPAIR_BASE_URL", "http://repair.test:8000/v1"
    )
    monkeypatch.setattr(settings, "PLANNING_REPAIR_API_KEY", "repair-dummy")
    monkeypatch.setattr(settings, "DEBUG_REPAIR_BASE_URL", "http://debug.test:8000/v1")
    monkeypatch.setattr(settings, "DEBUG_REPAIR_API_KEY", "debug-dummy")

    repair = _runtime(db_session, BackendRole.REPAIR)
    debug = _runtime(db_session, BackendRole.DEBUG_REPAIR)
    completion = _runtime(db_session, BackendRole.COMPLETION_REPAIR)

    assert repair._base_url == "http://repair.test:8000/v1"
    assert repair._api_key() == "repair-dummy"
    assert debug._base_url == "http://debug.test:8000/v1"
    assert debug._api_key() == "debug-dummy"
    assert completion._base_url == "http://repair.test:8000/v1"
    assert completion._api_key() == "repair-dummy"
    for runtime in (repair, debug, completion):
        assert runtime._base_url != PLANNING_GATEWAY


def test_planning_provider_contract_validates_the_planning_endpoint(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "PLANNING_BACKEND", "openai_chat_completions")
    monkeypatch.setattr(settings, "PLANNING_ADAPTATION_PROFILE", PLANNING_PROFILE)
    monkeypatch.setattr(settings, "PLANNER_MODEL", PLANNING_MODEL)

    monkeypatch.setattr(settings, "PLANNING_DIRECT_BASE_URL", "")
    with pytest.raises(RuntimeCapabilityError) as excinfo:
        validate_runtime_provider_contract(db_session, BackendRole.PLANNING)
    assert excinfo.value.code == "provider_endpoint_incompatible"

    monkeypatch.setattr(
        settings, "PLANNING_DIRECT_BASE_URL", f"{PLANNING_GATEWAY}/chat/completions"
    )
    with pytest.raises(RuntimeCapabilityError):
        validate_runtime_provider_contract(db_session, BackendRole.PLANNING)

    monkeypatch.setattr(settings, "PLANNING_DIRECT_BASE_URL", PLANNING_GATEWAY)
    contract = validate_runtime_provider_contract(db_session, BackendRole.PLANNING)
    assert contract["base_url"] == PLANNING_GATEWAY
    assert contract["endpoint"] == f"{PLANNING_GATEWAY}/chat/completions"
    assert contract["model"] == PLANNING_MODEL
    assert contract["adaptation_profile"] == PLANNING_PROFILE


# ---------------------------------------------------------------------------
# Model resolution and D3 fail-closed profile behavior
# ---------------------------------------------------------------------------


def test_planning_model_family_wins_over_generic_chat_model(db_session, monkeypatch):
    _planning_endpoint_configured(monkeypatch)
    runtime = _runtime(db_session, BackendRole.PLANNING)

    assert settings.OPENAI_CHAT_COMPLETIONS_MODEL == "wrong-model"
    assert runtime._model_name() == PLANNING_MODEL
    assert runtime.get_backend_metadata()["model_family"] == PLANNING_MODEL
    assert runtime.get_backend_metadata()["adaptation_profile"] == PLANNING_PROFILE


def test_planning_profile_backend_mismatch_still_fails_closed(db_session, monkeypatch):
    """D3 is deliberately NOT repaired: fail-closed is the correct behavior."""

    monkeypatch.setattr(settings, "PLANNING_BACKEND", "openai_chat_completions")
    monkeypatch.setattr(settings, "PLANNER_MODEL", PLANNING_MODEL)

    monkeypatch.setattr(settings, "PLANNING_ADAPTATION_PROFILE", "openclaw_default")
    with pytest.raises(UnsupportedRuntimeProfileError):
        resolve_runtime_configuration(db_session, BackendRole.PLANNING)

    monkeypatch.setattr(settings, "PLANNING_ADAPTATION_PROFILE", PLANNING_PROFILE)
    configuration = resolve_runtime_configuration(db_session, BackendRole.PLANNING)
    assert configuration.backend_name == "openai_chat_completions"
    assert configuration.model_family == PLANNING_MODEL
    # No auto-coercion: openclaw_default was rejected, never rewritten.
    assert configuration.adaptation_profile == PLANNING_PROFILE
    validate_runtime_capabilities(
        require_backend_descriptor(configuration.backend_name),
        BackendRole.PLANNING,
        dispatch=True,
    )


# ---------------------------------------------------------------------------
# Execution non-regression
# ---------------------------------------------------------------------------


def test_agent_runtime_execution_remains_openclaw_only(monkeypatch):
    """POST33-EXEC1 narrowed this: *agent* execution stays OpenClaw-only.

    ROUTE2 asserted that ``openai_chat_completions`` could not own the
    EXECUTION role at all, because ``supports_step_execution`` conflated
    "can produce a bounded step result" with "is an agent runtime with native
    tools".  EXEC1 split those.  The direct backend may now serve the
    structured-orchestrator topology the execution loop actually runs; the
    agent topology still fails closed, which is what this test guards.
    """

    # Keep this capability assertion independent of the CI execution-model
    # setting; an incapable backend must fail closed on its role boundary.
    monkeypatch.setattr(settings, "EXECUTION_MODEL", "capability-test-model")
    openclaw = get_backend_descriptor("local_openclaw").capabilities
    assert openclaw.supports_step_reasoning is True
    assert openclaw.supports_step_execution is True
    assert openclaw.supports_tool_execution is True
    assert openclaw.supports_agent_workspace_binding is True
    assert openclaw.supports_checkpoint_resume is True
    assert openclaw.supports_streaming is True

    chat = get_backend_descriptor("openai_chat_completions")
    assert chat.capabilities.supports_step_execution is False
    assert chat.capabilities.supports_tool_execution is False
    assert chat.capabilities.supports_agent_workspace_binding is False
    with pytest.raises(RuntimeCapabilityError) as excinfo:
        validate_runtime_capabilities(
            chat,
            BackendRole.EXECUTION,
            effective_context_tokens=200_000,
            dispatch=True,
            execution_topology=ExecutionTopology.AGENT_RUNTIME,
        )
    assert excinfo.value.code == "provider_endpoint_incompatible"


def test_no_discovery_backend_role_or_setting_was_added():
    assert [role.value for role in BackendRole] == [
        "planning",
        "execution",
        "debug_repair",
        "repair",
        "completion_repair",
    ]
    assert not hasattr(settings, "DISCOVERY_BACKEND")

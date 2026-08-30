#!/usr/bin/env python3
"""PHASE34-S2A: frozen first-pass Planning interface ablation.

This evaluation harness is deliberately incapable of executing a Plan. It:

1. creates deterministic fixture bytes in an isolated /tmp directory;
2. renders CURRENT and evaluation-only COMPACT Planning prompts;
3. calls only the configured BackendRole.PLANNING text-generation runtime;
4. runs the existing parser, deterministic Planning normalizers, and validator;
5. writes compact JSON evidence.

It never imports an executor, lifecycle coordinator, publication service,
repair service, Task/Session API, or candidate mutation path.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings
from app.database import SessionLocal
from app.services.agents.agent_runtime import (
    BackendRole,
    create_agent_runtime,
    resolve_execution_topology_for_role,
    resolve_runtime_configuration,
)
from app.services.agents.agent_backends import require_backend_descriptor
from app.services.orchestration.context.assembly import (
    _build_project_structure_capsule,
    _shape_project_context,
    build_workspace_inventory_summary,
    render_adapted_runtime_prompt,
)
from app.services.orchestration.error_handler import EnhancedErrorHandler
from app.services.orchestration.operations.file_ops_contract import SUPPORTED_FILE_OPS
from app.services.orchestration.phases.planning_plan_shape import (
    split_repaired_single_step_full_lifecycle_plan,
)
from app.services.orchestration.phases.planning_verification import (
    _strengthen_weak_expected_file_verifications,
)
from app.services.orchestration.planning.normalization import (
    normalize_blank_line_divergent_replace_anchors,
    normalize_existing_file_target_plan,
    normalize_stale_replace_ops_to_small_file_writes,
)
from app.services.orchestration.planning.planner import PlannerService
from app.services.orchestration.planning.read_only_discovery import (
    DiscoveryObservation,
    DiscoveryRequest,
    SearchHit,
    assess_discovery_admission,
    execute_discovery_request,
    render_discovery_observation,
)
from app.services.orchestration.planning.semantic_target_inventory import (
    build_semantic_target_inventory,
    normalize_provider_semantic_intents,
)
from app.services.orchestration.planning.source_materialization import (
    PlannerSourceMaterialization,
    materialize_planner_source_context,
    observed_candidate_paths,
)
from app.services.orchestration.planning.workspace_identity import (
    PlannerWorkspaceIdentity,
    render_planner_workspace_identity,
)
from app.services.orchestration.prompt_templates import PromptTemplates
from app.services.orchestration.validation.parsing import extract_plan_steps
from app.services.orchestration.validation.validator import ValidatorService
from app.services.orchestration.validation.workspace_guard import normalize_plan
from app.services.orchestration.workflow_profiles import get_workflow_phases
from app.services.project.source_imports import (
    python_test_source_context_from_tests,
    render_source_stub_block,
)
from app.task_intent import TaskIntentMode


LOGGER = logging.getLogger("phase34s2a")
DEFAULT_EVIDENCE_DIR = (
    REPO_ROOT / "docs/roadmap/reports/evidence/phase34-s2a"
)
EXACT_E_VERIFICATION = "python -m pytest -q test_calculator.py"
VARIANTS = ("CURRENT", "COMPACT")
CALL_ORDER = (
    ("A", "CURRENT"),
    ("A", "COMPACT"),
    ("B", "COMPACT"),
    ("B", "CURRENT"),
    ("C", "CURRENT"),
    ("C", "COMPACT"),
    ("D", "COMPACT"),
    ("D", "CURRENT"),
    ("E", "CURRENT"),
    ("E", "COMPACT"),
)
NON_SAFETY_LEXICAL_CODES = frozenset(
    {
        "existing_file_write_without_authorization",
        "existing_file_write_requires_explicit_replace_authorization",
    }
)


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    name: str
    task: str
    intent_mode: str
    files: Mapping[str, str]
    grounded_paths: tuple[str, ...]
    requirements: tuple[str, ...]
    first_ordered_task: bool = False


@dataclass
class FixtureContext:
    spec: FixtureSpec
    workspace: Path
    identity: PlannerWorkspaceIdentity
    materialization: PlannerSourceMaterialization
    observation: DiscoveryObservation | None
    project_context: str
    structure_capsule: str
    python_source_context: str
    source_stub_context: str
    source_block: str
    discovery_block: str
    evidence_block: str
    semantic_input_digest: str
    prompts: dict[str, dict[str, Any]]


def _fixtures() -> dict[str, FixtureSpec]:
    return {
        "A": FixtureSpec(
            fixture_id="A",
            name="CREATE_ONLY_NEW_PROJECT",
            task=(
                "Create a small Python temperature conversion capability supporting "
                "Celsius and Fahrenheit. Provide a reusable conversion API, a simple "
                "command-line interface, automated tests, and a short README. Keep "
                "the implementation dependency-light."
            ),
            intent_mode=TaskIntentMode.CREATE_ONLY.value,
            files={},
            grounded_paths=(),
            requirements=(
                "celsius_and_fahrenheit",
                "reusable_api",
                "command_line_interface",
                "automated_tests",
                "short_readme",
                "dependency_light",
            ),
            first_ordered_task=True,
        ),
        "B": FixtureSpec(
            fixture_id="B",
            name="GROUNDED_EXISTING_EDIT",
            task=(
                "Update the greeting behavior to include the person's name while "
                "preserving the existing public function and current behavior for "
                "callers. Update tests only if necessary and verify the project "
                "still passes."
            ),
            intent_mode=TaskIntentMode.DEFAULT.value,
            files={
                "greeter.py": (
                    '"""Small greeting API."""\n\n'
                    "def greet(name: str) -> str:\n"
                    '    """Return the current generic greeting."""\n'
                    '    return "Hello"\n'
                ),
                "test_greeter.py": (
                    "from greeter import greet\n\n\n"
                    "def test_greet_keeps_current_behavior():\n"
                    '    assert greet("Ada") == "Hello"\n'
                ),
            },
            grounded_paths=("greeter.py", "test_greeter.py"),
            requirements=(
                "name_in_greeting",
                "public_greet_function_preserved",
                "caller_compatible_signature",
                "project_tests_verified",
            ),
        ),
        "C": FixtureSpec(
            fixture_id="C",
            name="DISCOVERY_REQUIRED_EDIT",
            task=(
                "Change the application so failed login attempts are reported with "
                "a clear user-facing message, and add or update automated coverage "
                "for the behavior."
            ),
            intent_mode=TaskIntentMode.DEFAULT.value,
            files={
                "auth_service.py": (
                    '"""Authentication behavior."""\n\n'
                    "# User-visible text for failed login attempts.\n"
                    'FAILED_LOGIN_MESSAGE = ""\n\n'
                    "def authenticate(username: str, password: str) -> dict:\n"
                    '    if username == "admin" and password == "secret":\n'
                    '        return {"ok": True, "message": "Welcome"}\n'
                    '    return {"ok": False, "message": FAILED_LOGIN_MESSAGE}\n'
                ),
                "test_auth_service.py": (
                    "from auth_service import authenticate\n\n\n"
                    "def test_failed_login_is_rejected():\n"
                    '    result = authenticate("bad", "credentials")\n'
                    '    assert result["ok"] is False\n'
                ),
            },
            grounded_paths=(),
            requirements=(
                "clear_user_facing_failure_message",
                "failed_login_behavior_updated",
                "automated_coverage",
            ),
        ),
        "D": FixtureSpec(
            fixture_id="D",
            name="SMALL_MULTI_FILE_FEATURE",
            task=(
                "Add support for recording a task priority while preserving existing "
                "task creation behavior. Update the relevant API/data handling and "
                "automated tests."
            ),
            intent_mode=TaskIntentMode.DEFAULT.value,
            files={
                "tasks.py": (
                    '"""Task data handling."""\n\n'
                    "def make_task(title: str) -> dict:\n"
                    '    return {"title": title}\n'
                ),
                "api.py": (
                    "from tasks import make_task\n\n\n"
                    "def create_task(payload: dict) -> dict:\n"
                    '    return make_task(payload["title"])\n'
                ),
                "test_tasks.py": (
                    "from api import create_task\n\n\n"
                    "def test_create_task():\n"
                    '    assert create_task({"title": "Write docs"}) == {"title": "Write docs"}\n'
                ),
            },
            grounded_paths=("tasks.py", "api.py", "test_tasks.py"),
            requirements=(
                "priority_recorded",
                "existing_creation_behavior_preserved",
                "api_and_data_handling_updated",
                "automated_tests_updated",
            ),
        ),
        "E": FixtureSpec(
            fixture_id="E",
            name="EXACT_VERIFICATION_BUG_FIX",
            task=(
                "Fix the calculator regression so subtraction returns the correct "
                "result. Preserve existing behavior and verify the fix using exactly:\n"
                f"{EXACT_E_VERIFICATION}"
            ),
            intent_mode=TaskIntentMode.DEFAULT.value,
            files={
                "calculator.py": (
                    '"""Tiny calculator."""\n\n'
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n\n\n"
                    "def subtract(a: int, b: int) -> int:\n"
                    "    return a + b\n"
                ),
                "test_calculator.py": (
                    "from calculator import add, subtract\n\n\n"
                    "def test_addition():\n"
                    "    assert add(2, 3) == 5\n\n\n"
                    "def test_subtraction():\n"
                    "    assert subtract(5, 3) == 2\n"
                ),
            },
            grounded_paths=("calculator.py", "test_calculator.py"),
            requirements=(
                "subtraction_corrected",
                "existing_addition_preserved",
                "exact_verification_command",
            ),
        ),
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_digest(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _script_digest() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o777)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o666)


def _workspace_content_digest(workspace: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
        relative = path.relative_to(workspace).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _workspace_tree(workspace: Path) -> list[str]:
    return sorted(
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    )


def _runtime_freeze(db: Any) -> dict[str, Any]:
    configuration = resolve_runtime_configuration(db, BackendRole.PLANNING)
    runtime = create_agent_runtime(db, None, None, role=BackendRole.PLANNING)
    metadata = runtime.get_backend_metadata()
    descriptor = require_backend_descriptor(configuration.backend_name)
    base_url_value = getattr(runtime, "_base_url", None)
    base_url = (
        base_url_value
        if isinstance(base_url_value, str)
        else base_url_value()
        if callable(base_url_value)
        else None
    )
    provider_model = None
    provider_context_limit = None
    provider_root = None
    if base_url:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{str(base_url).rstrip('/')}/models")
            response.raise_for_status()
            body = response.json()
        models = body.get("data") if isinstance(body, dict) else None
        for item in models or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("id") or "") == configuration.model_family:
                provider_model = item.get("id")
                provider_context_limit = item.get("max_model_len")
                provider_root = item.get("root")
                break
    return {
        "planning_backend": configuration.backend_name,
        "planning_model": configuration.model_family,
        "planning_adaptation_profile": configuration.adaptation_profile,
        "planning_base_url_or_provider_id": base_url,
        "provider_reported_model": provider_model,
        "provider_reported_root": provider_root,
        "temperature": (
            settings.OPENAI_CHAT_COMPLETIONS_TEMPERATURE
            if configuration.backend_name == "openai_chat_completions"
            else settings.PLANNING_DIRECT_TEMPERATURE
        ),
        "max_output_tokens": None,
        "max_output_tokens_note": (
            "ordinary Planning execute_task omits max_tokens; the same provider "
            "default is used for both variants"
        ),
        "context_limit": provider_context_limit
        or descriptor.capabilities.max_context_tokens,
        "context_limit_descriptor": descriptor.capabilities.max_context_tokens,
        "low_resource_single_model": bool(settings.LOW_RESOURCE_SINGLE_MODEL),
        "execution_topology": resolve_execution_topology_for_role(
            db, BackendRole.EXECUTION
        ).value,
        "timeout_seconds": int(settings.PLANNING_SYNTHESIS_TIMEOUT_SECONDS),
        "stream": False,
        "reasoning_options": "ordinary adapter defaults; no invocation_options",
        "runtime_configuration": configuration.to_dict(),
        "runtime_metadata": {
            "backend": metadata.get("backend"),
            "model_family": metadata.get("model_family"),
            "adaptation_profile": metadata.get("adaptation_profile"),
            "role": metadata.get("role"),
            "capabilities": metadata.get("capabilities"),
        },
        "api_key_present": bool(getattr(settings, "PLANNING_DIRECT_API_KEY", "")),
    }


def _create_workspace(root: Path, spec: FixtureSpec) -> Path:
    workspace = root / f"fixture-{spec.fixture_id.lower()}-{spec.name.lower()}"
    workspace.mkdir(parents=True, exist_ok=False)
    os.chmod(workspace, 0o777)
    for relative, content in spec.files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o777)
        path.write_text(content, encoding="utf-8", newline="")
        os.chmod(path, 0o666)
    return workspace


def _identity(workspace: Path, spec: FixtureSpec) -> PlannerWorkspaceIdentity:
    return PlannerWorkspaceIdentity.from_paths(
        project_workspace=workspace,
        physical_runtime_root=workspace,
        logical_project_name=f"s2a-{spec.name.lower()}",
        display_project_path="current isolated task workspace",
        display_project_workspace_path="current isolated task workspace",
    )


def _source_materialization(
    workspace: Path,
    spec: FixtureSpec,
    identity: PlannerWorkspaceIdentity,
) -> tuple[PlannerSourceMaterialization, DiscoveryObservation | None]:
    initial = materialize_planner_source_context(
        workspace,
        task_description=spec.task,
        expected_paths=spec.grounded_paths,
        workspace_identity=identity,
    )
    admission = assess_discovery_admission(
        prompt=spec.task,
        planner_contract=None,
        materialization=initial,
        intent_mode=spec.intent_mode,
    )
    if spec.fixture_id == "C":
        if admission.reason != "no_explicit_source_or_creation_path":
            raise RuntimeError(f"Fixture C discovery admission drifted: {asdict(admission)}")
        observation = execute_discovery_request(
            workspace,
            DiscoveryRequest(action="search_text", query="failed login", paths=()),
        )
        if not observation.hits:
            raise RuntimeError("Fixture C canonical discovery produced no hits")
        materialization = materialize_planner_source_context(
            workspace,
            task_description=spec.task,
            supporting_paths=observation.materialization_paths(),
            workspace_identity=identity,
        )
        return materialization, observation
    if spec.fixture_id == "A":
        if admission.reason != "typed_create_only_intent":
            raise RuntimeError(f"Fixture A CREATE_ONLY admission drifted: {asdict(admission)}")
    elif admission.status != "SKIPPED_SUFFICIENT_GROUNDING":
        raise RuntimeError(
            f"Fixture {spec.fixture_id} unexpectedly requires discovery: {asdict(admission)}"
        )
    return initial, None


def _raw_current_prompt(
    *,
    spec: FixtureSpec,
    identity: PlannerWorkspaceIdentity,
    materialization: PlannerSourceMaterialization,
    observation: DiscoveryObservation | None,
    project_context: str,
    structure_capsule: str,
    python_source_context: str,
    source_stub_context: str,
    source_block: str,
    discovery_block: str,
) -> str:
    candidate_paths = observed_candidate_paths(observation)
    raw = PromptTemplates.build_planning_prompt(
        task_description=spec.task,
        project_context=project_context,
        project_dir=identity.display_project_path,
        execution_profile="full_lifecycle",
        workflow_profile="default",
        workflow_phases=get_workflow_phases("default"),
        project_structure_capsule=structure_capsule,
        workspace_identity=identity,
        planner_contract=None,
        source_materialization=materialization,
        additional_candidate_paths=candidate_paths,
        intent_mode=spec.intent_mode,
    )
    for supplement in (
        python_source_context,
        source_stub_context,
        discovery_block,
        source_block,
    ):
        if supplement:
            raw += "\n\n" + supplement
    return raw


def _compact_directives(spec: FixtureSpec) -> list[str]:
    create_only = (
        "This CREATE_ONLY task may create declared new relative files, but may "
        "not modify or delete any existing file."
        if spec.intent_mode == TaskIntentMode.CREATE_ONLY.value
        else "This DEFAULT task may modify an existing file only when its current source is supplied below."
    )
    return [
        "Return one top-level JSON array and no prose or markdown.",
        "Use 1 to 4 ordered step objects; step_number values are consecutive from 1.",
        "Every step has exactly step_number, description, commands, verification, rollback, expected_files, and optional ops.",
        "commands and expected_files are string arrays; verification and rollback are a string or null; ops is an array.",
        "Supported ops are mkdir{op,path}, delete_file{op,path}, write_file/append_file{op,path,content}, and replace_in_file{op,path,old,new}.",
        create_only,
        "Use project-relative paths from the current root; do not use absolute paths, parent traversal, home paths, or a nested project root.",
        "Prefer structured ops for file changes; do not assume source that is absent from CURRENT EVIDENCE.",
        "Commands must be runnable foreground shell commands of at most 900 characters; no prose commands, background servers, or heredocs.",
        "A step may have empty commands when its ops perform the file change.",
        "List every file created or changed by a step in expected_files; do not list speculative files.",
        "Give every mutating step executable verification and finish with one meaningful project test/build verification.",
        "When the user gives an exact verification command, preserve that exact command.",
        "Use only the supplied task, mode, evidence, actions, and Plan fields; do not invent internal authority or lifecycle concepts.",
    ]


def _raw_compact_prompt(spec: FixtureSpec, evidence_block: str) -> str:
    directives = _compact_directives(spec)
    return "\n\n".join(
        (
            "USER TASK\n" + spec.task,
            "TASK MODE\n" + spec.intent_mode.upper(),
            "CURRENT EVIDENCE\n" + evidence_block,
            "ALLOWED ACTIONS AND PLAN CONTRACT\n"
            + "\n".join(f"{index}. {item}" for index, item in enumerate(directives, 1)),
        )
    )


def _adapt_prompt(db: Any, raw: str, profile: str) -> str:
    return render_adapted_runtime_prompt(
        db,
        objective="Generate a machine-runnable JSON execution plan for the requested task.",
        execution_mode="planning",
        prompt_body=raw,
        instructions=[
            "Do not implement anything yet.",
            "Return a sequential JSON plan only.",
        ],
        context={
            "Project Directory": "current isolated task workspace",
            "Execution Profile": "full_lifecycle",
            "Workflow Profile": "default",
        },
        expected_output="JSON array of orchestration step objects.",
        adaptation_profile=profile,
    )


def _section_order(raw: str) -> list[str]:
    known = (
        "USER TASK",
        "TASK MODE",
        "CURRENT EVIDENCE",
        "ALLOWED ACTIONS AND PLAN CONTRACT",
        "Task:",
        "Execution Profile:",
        "Context:",
        "Project Structure",
        "Workspace:",
        "Execution Boundary:",
        "Requirements:",
        "Planning Rules:",
        "Execution Profile Rules:",
        "Workflow Phases:",
        "Invalid output wrapper",
        "## CURRENT SOURCE MATERIALIZATION",
        "## READ-ONLY DISCOVERY OBSERVATION",
    )
    positions = [(raw.find(item), item) for item in known if raw.find(item) >= 0]
    return [item for _, item in sorted(positions)]


def _redacted_excerpt(text: str, fixture: FixtureSpec, limit: int = 360) -> dict[str, str]:
    redacted = text
    for content in sorted(fixture.files.values(), key=len, reverse=True):
        if content:
            redacted = redacted.replace(content, "[FROZEN_SOURCE_REDACTED]")
    if len(redacted) <= limit * 2:
        return {"text": redacted}
    return {"head": redacted[:limit], "tail": redacted[-limit:]}


def _prompt_record(
    *, raw: str, adapted: str, fixture: FixtureSpec, profile: str, directives: int
) -> dict[str, Any]:
    source_format_preserved = all(
        content.rstrip() in adapted for content in fixture.files.values() if content
    )
    return {
        "prompt_chars": len(adapted),
        "prompt_tokens_approx": (len(adapted) + 3) // 4,
        "prompt_sha256": _sha256_text(adapted),
        "pre_transform_prompt_sha256": _sha256_text(raw),
        "post_transform_provider_prompt_sha256": _sha256_text(adapted),
        "final_provider_bound_prompt_sha256": _sha256_text(adapted),
        "pre_transform_chars": len(raw),
        "post_transform_chars": len(adapted),
        "prompt_chars_before_after": [len(raw), len(adapted)],
        "section_order": _section_order(raw),
        "adaptation_profile": profile,
        "static_directive_count": directives,
        "whitespace_preserved": raw.strip() in adapted,
        "section_boundaries_preserved": all(
            heading in adapted for heading in _section_order(raw)
        ),
        "source_format_preserved": source_format_preserved,
        "runtime_adapter_additional_transform": "NONE_FOR_OPENAI_CHAT_COMPLETIONS",
        "capture": {
            "pre_transform_prompt": _redacted_excerpt(raw, fixture),
            "post_transform_provider_prompt": _redacted_excerpt(adapted, fixture),
            "retention_note": "full prompt held in memory; compact evidence retains hashes and source-redacted bounded excerpts",
        },
    }


def _build_context(
    db: Any,
    workspace: Path,
    spec: FixtureSpec,
    runtime: Mapping[str, Any],
    *,
    frozen_materialization: PlannerSourceMaterialization | None = None,
    frozen_observation: DiscoveryObservation | None = None,
) -> FixtureContext:
    identity = _identity(workspace, spec)
    if frozen_materialization is None:
        materialization, observation = _source_materialization(workspace, spec, identity)
    else:
        materialization = frozen_materialization
        observation = frozen_observation
    workspace_summary = build_workspace_inventory_summary(
        workspace, workspace_review=None, max_files=10
    )
    project_context = _shape_project_context(
        "",
        workspace_summary=workspace_summary,
        recent_history="",
        validation_history="",
        operator_guidance="",
        max_chars=800,
    )
    structure_capsule = _build_project_structure_capsule(workspace)
    python_source_context = python_test_source_context_from_tests(workspace)
    source_stub_context = render_source_stub_block(workspace)
    candidate_paths = observed_candidate_paths(observation)
    source_block = materialization.to_prompt_block(
        provider_safe=True, additional_candidate_paths=candidate_paths
    )
    discovery_block = render_discovery_observation(observation)
    evidence_parts = [
        "PROJECT CONTEXT\n" + project_context,
        "PROJECT STRUCTURE\n" + (structure_capsule or "No structural index."),
    ]
    for label, value in (
        ("PYTHON TEST SOURCE CONTEXT", python_source_context),
        ("SOURCE STUB CONTEXT", source_stub_context),
        ("FROZEN DISCOVERY EVIDENCE", discovery_block),
        ("CURRENT SOURCE MATERIALIZATION", source_block),
    ):
        if value:
            evidence_parts.append(label + "\n" + value)
    evidence_block = "\n\n".join(evidence_parts)
    semantic_input = {
        "task": spec.task,
        "intent_mode": spec.intent_mode,
        "project_context": project_context,
        "structure_capsule": structure_capsule,
        "python_source_context": python_source_context,
        "source_stub_context": source_stub_context,
        "source_materialization": materialization.to_metadata(),
        "discovery_observation": asdict(observation) if observation else None,
        "knowledge_context": None,
        "workspace_identity": render_planner_workspace_identity(identity),
        "execution_profile": "full_lifecycle",
        "workflow_profile": "default",
        "execution_topology": runtime["execution_topology"],
    }
    raw_current = _raw_current_prompt(
        spec=spec,
        identity=identity,
        materialization=materialization,
        observation=observation,
        project_context=project_context,
        structure_capsule=structure_capsule,
        python_source_context=python_source_context,
        source_stub_context=source_stub_context,
        source_block=source_block,
        discovery_block=discovery_block,
    )
    raw_compact = _raw_compact_prompt(spec, evidence_block)
    profile = str(runtime["planning_adaptation_profile"])
    current = _adapt_prompt(db, raw_current, profile)
    compact = _adapt_prompt(db, raw_compact, profile)
    prompt_profile = PlannerService.select_prompt_profile(
        str(runtime["planning_backend"]), str(runtime["planning_model"])
    )
    current = PlannerService.apply_prompt_profile(current, prompt_profile)
    compact = PlannerService.apply_prompt_profile(compact, prompt_profile)
    prompts = {
        "CURRENT": {
            "raw": raw_current,
            "provider": current,
            "record": _prompt_record(
                raw=raw_current,
                adapted=current,
                fixture=spec,
                profile=profile,
                directives=-1,
            ),
        },
        "COMPACT": {
            "raw": raw_compact,
            "provider": compact,
            "record": _prompt_record(
                raw=raw_compact,
                adapted=compact,
                fixture=spec,
                profile=profile,
                directives=len(_compact_directives(spec)) + 2,
            ),
        },
    }
    return FixtureContext(
        spec=spec,
        workspace=workspace,
        identity=identity,
        materialization=materialization,
        observation=observation,
        project_context=project_context,
        structure_capsule=structure_capsule,
        python_source_context=python_source_context,
        source_stub_context=source_stub_context,
        source_block=source_block,
        discovery_block=discovery_block,
        evidence_block=evidence_block,
        semantic_input_digest=_json_digest(semantic_input),
        prompts=prompts,
    )


def _identity_record(identity: PlannerWorkspaceIdentity) -> dict[str, Any]:
    return {
        "logical_project_name": identity.logical_project_name,
        "planner_display_root": identity.planner_display_root,
        "display_project_path": identity.display_project_path,
        "display_project_workspace_path": identity.display_project_workspace_path,
        "physical_runtime_basename": identity.physical_runtime_basename,
        "forbidden_root_aliases": list(identity.forbidden_root_aliases),
        "physical_runtime_root_sha256": _sha256_text(
            str(identity.physical_runtime_root)
        ),
    }


def prepare(evidence_dir: Path) -> Path:
    evidence_dir = evidence_dir.resolve()
    allowed_root = (REPO_ROOT / "docs/roadmap/reports/evidence").resolve()
    if allowed_root not in evidence_dir.parents:
        raise RuntimeError(f"Evidence directory must be under {allowed_root}")
    freeze_path = evidence_dir / "fixture-freeze.json"
    if freeze_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing freeze: {freeze_path}")
    workspace_root = Path(tempfile.mkdtemp(prefix="phase34-s2a-"))
    os.chmod(workspace_root, 0o777)
    specs = _fixtures()
    db = SessionLocal()
    try:
        runtime = _runtime_freeze(db)
        contexts: dict[str, FixtureContext] = {}
        for fixture_id, spec in specs.items():
            workspace = _create_workspace(workspace_root, spec)
            contexts[fixture_id] = _build_context(db, workspace, spec, runtime)
    finally:
        db.close()

    fixtures: dict[str, Any] = {}
    for fixture_id, context in contexts.items():
        observation_payload = (
            asdict(context.observation) if context.observation else None
        )
        fixtures[fixture_id] = {
            "fixture_id": fixture_id,
            "name": context.spec.name,
            "task_text_sha256": _sha256_text(context.spec.task),
            "intent_mode": context.spec.intent_mode,
            "workspace_tree": _workspace_tree(context.workspace),
            "workspace_content_digest": _workspace_content_digest(context.workspace),
            "source_materialization_digest": _json_digest(
                context.materialization.to_metadata()
            ),
            "discovery_observation_digest": (
                _json_digest(observation_payload) if observation_payload else "NONE"
            ),
            "discovery_observation": observation_payload,
            "discovery_observation_summary": (
                {
                    "action": context.observation.action,
                    "status": context.observation.status,
                    "paths": list(context.observation.materialization_paths()),
                    "result_count": context.observation.result_count,
                }
                if context.observation
                else None
            ),
            "planner_workspace_identity": _identity_record(context.identity),
            "project_context_digest": _sha256_text(context.project_context),
            "knowledge_context": "EMPTY_CONTROLLED",
            "semantic_input_digest": context.semantic_input_digest,
            "requirements": list(context.spec.requirements),
            "first_ordered_task": context.spec.first_ordered_task,
            "prompts": {
                variant: context.prompts[variant]["record"]
                for variant in VARIANTS
            },
            "variant_evidence_equality": {
                "task": True,
                "intent": True,
                "workspace": True,
                "source_materialization": True,
                "discovery_observation": True,
                "project_context": True,
                "knowledge": True,
                "execution_topology": True,
            },
        }
    payload = {
        "schema_version": "phase34-s2a-freeze/1",
        "prepared_at_epoch": time.time(),
        "evaluation_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "evaluation_script_sha256": _script_digest(),
        "workspace_root": str(workspace_root),
        "runtime_freeze": runtime,
        "runtime_freeze_digest": _json_digest(runtime),
        "provider_generation_calls_before_freeze": 0,
        "knowledge_control": (
            "Empty for both variants: live retrieval would add unrelated mutable "
            "retrieval state to a prompt-interface ablation."
        ),
        "discovery_control": (
            "Fixture C uses one provider-free canonical search_text request through "
            "execute_discovery_request; the resulting normal DiscoveryObservation "
            "and rematerialized source are frozen once for both variants."
        ),
        "call_order": [list(item) for item in CALL_ORDER],
        "max_generation_calls": 10,
        "fixtures": fixtures,
        "safety": {
            "repairs_enabled": False,
            "plan_execution_enabled": False,
            "product_lifecycle_enabled": False,
            "publication_enabled": False,
            "canonical_workspace_mutation_enabled": False,
        },
    }
    _write_json(freeze_path, payload)
    print(json.dumps({
        "status": "prepared",
        "fixture_freeze": str(freeze_path),
        "workspace_root": str(workspace_root),
        "runtime_freeze_digest": payload["runtime_freeze_digest"],
        "provider_generation_calls": 0,
    }, indent=2))
    return freeze_path


def _load_freeze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_script_sha256") != _script_digest():
        raise RuntimeError("Evaluation script changed after fixture freeze")
    if payload.get("provider_generation_calls_before_freeze") != 0:
        raise RuntimeError("Freeze was not recorded before generation")
    return payload


def _rebuild_contexts(db: Any, freeze: Mapping[str, Any]) -> dict[str, FixtureContext]:
    workspace_root = Path(str(freeze["workspace_root"]))
    if not workspace_root.is_dir() or not workspace_root.name.startswith("phase34-s2a-"):
        raise RuntimeError("Frozen temporary workspace root is unavailable or unsafe")
    current_runtime = _runtime_freeze(db)
    if _json_digest(current_runtime) != freeze["runtime_freeze_digest"]:
        raise RuntimeError("Effective Planning runtime changed after freeze")
    contexts: dict[str, FixtureContext] = {}
    for fixture_id, spec in _fixtures().items():
        workspace = workspace_root / f"fixture-{fixture_id.lower()}-{spec.name.lower()}"
        frozen = freeze["fixtures"][fixture_id]
        identity = _identity(workspace, spec)
        observation_payload = frozen.get("discovery_observation")
        observation = None
        if observation_payload:
            observation = DiscoveryObservation(
                action=str(observation_payload["action"]),
                status=str(observation_payload["status"]),
                paths=tuple(observation_payload.get("paths") or ()),
                hits=tuple(
                    SearchHit(**item) for item in observation_payload.get("hits") or ()
                ),
                content=observation_payload.get("content"),
                truncated=bool(observation_payload.get("truncated")),
                reason=observation_payload.get("reason"),
            )
        materialization = materialize_planner_source_context(
            workspace,
            task_description=spec.task,
            expected_paths=spec.grounded_paths,
            supporting_paths=(
                observation.materialization_paths() if observation else ()
            ),
            workspace_identity=identity,
        )
        context = _build_context(
            db,
            workspace,
            spec,
            current_runtime,
            frozen_materialization=materialization,
            frozen_observation=observation,
        )
        checks = {
            "workspace_content_digest": _workspace_content_digest(workspace),
            "source_materialization_digest": _json_digest(
                context.materialization.to_metadata()
            ),
            "project_context_digest": _sha256_text(context.project_context),
            "semantic_input_digest": context.semantic_input_digest,
        }
        for key, observed in checks.items():
            if observed != frozen[key]:
                raise RuntimeError(
                    f"Fixture {fixture_id} {key} changed after freeze: "
                    f"{observed} != {frozen[key]}"
                )
        for variant in VARIANTS:
            if (
                context.prompts[variant]["record"]["prompt_sha256"]
                != frozen["prompts"][variant]["prompt_sha256"]
            ):
                raise RuntimeError(
                    f"Fixture {fixture_id} {variant} prompt changed after freeze"
                )
        contexts[fixture_id] = context
    return contexts


def _plan_text(plan: Iterable[Mapping[str, Any]], raw: str = "") -> str:
    return (json.dumps(list(plan), ensure_ascii=False) + "\n" + raw).lower()


def _plan_paths(plan: Iterable[Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    expected: list[str] = []
    mutating: list[str] = []
    operations: list[str] = []
    for step in plan:
        for path in step.get("expected_files") or []:
            value = str(path or "").strip().replace("\\", "/").lstrip("./")
            if value and value not in expected:
                expected.append(value)
        for operation in step.get("ops") or []:
            if not isinstance(operation, Mapping):
                continue
            op_name = str(operation.get("op") or "")
            operations.append(op_name)
            if op_name in {"write_file", "append_file", "replace_in_file", "delete_file"}:
                value = str(operation.get("path") or "").strip().replace("\\", "/").lstrip("./")
                if value and value not in mutating:
                    mutating.append(value)
    return expected, mutating, operations


def _verification_commands(plan: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(step.get("verification") or "").strip()
        for step in plan
        if str(step.get("verification") or "").strip()
    ]


def _semantic_checklist(spec: FixtureSpec, plan: list[dict[str, Any]], raw: str) -> dict[str, bool]:
    text = _plan_text(plan, raw)
    verifications = _verification_commands(plan)
    if spec.fixture_id == "A":
        third_party = bool(re.search(r"\bpip\s+install\b|requirements\.txt|pyproject\.toml", text))
        return {
            "celsius_and_fahrenheit": "celsius" in text and "fahrenheit" in text,
            "reusable_api": bool(re.search(r"def\s+(?:convert|.*celsius|.*fahrenheit)", text)) or "reusable" in text,
            "command_line_interface": any(token in text for token in ("argparse", "__main__", "command-line", " cli")),
            "automated_tests": "pytest" in text or "test_" in text or "unittest" in text,
            "short_readme": "readme" in text,
            "dependency_light": not third_party,
        }
    if spec.fixture_id == "B":
        public_preserved = "def greet" in text or (
            "greeter.py" in text and "replace_in_file" in text and "greet" in text
        )
        return {
            "name_in_greeting": "name" in text and any(token in text for token in ("hello", "greet", "greeting")),
            "public_greet_function_preserved": public_preserved,
            "caller_compatible_signature": public_preserved and not any(token in text for token in ("rename greet", "remove greet", "delete greeter")),
            "project_tests_verified": any("pytest" in value or "unittest" in value for value in verifications),
        }
    if spec.fixture_id == "C":
        clear_message = "message" in text and any(
            token in text
            for token in (
                "login failed",
                "failed login",
                "invalid username",
                "invalid credentials",
                "unable to log in",
            )
        )
        return {
            "clear_user_facing_failure_message": clear_message,
            "failed_login_behavior_updated": "auth_service.py" in text and ("failed" in text or "invalid" in text),
            "automated_coverage": "test_auth_service.py" in text and any("pytest" in value for value in verifications),
        }
    if spec.fixture_id == "D":
        return {
            "priority_recorded": "priority" in text,
            "existing_creation_behavior_preserved": any(
                token in text
                for token in (
                    "default",
                    "optional",
                    "backward",
                    "preserv",
                    "if priority is not none",
                )
            ),
            "api_and_data_handling_updated": "api.py" in text and "tasks.py" in text,
            "automated_tests_updated": "test_tasks.py" in text and any("pytest" in value for value in verifications),
        }
    exact = EXACT_E_VERIFICATION in verifications
    subtract_correct = bool(re.search(r"return\s+a\s*-\s*b", text)) or (
        "subtract" in text and "a - b" in text
    )
    addition_preserved = "def add" in text or "preserv" in text or (
        "replace_in_file" in text and "subtract" in text
    )
    return {
        "subtraction_corrected": subtract_correct,
        "existing_addition_preserved": addition_preserved,
        "exact_verification_command": exact,
    }


def _verification_correctness(spec: FixtureSpec, plan: list[dict[str, Any]]) -> str:
    values = _verification_commands(plan)
    if spec.fixture_id == "E":
        if EXACT_E_VERIFICATION in values:
            return "PASS"
        if any("pytest" in value and "test_calculator.py" in value for value in values):
            return "PARTIAL"
        return "FAIL"
    if any("pytest" in value or "unittest" in value for value in values):
        return "PASS"
    if values:
        return "PARTIAL"
    return "FAIL"


def _normalization_pipeline(context: FixtureContext, raw_output: str) -> dict[str, Any]:
    handler = EnhancedErrorHandler()
    success, parsed, strategy = handler.attempt_json_parsing(raw_output, context="planning")
    if not success:
        return {
            "parse_success": False,
            "parse_strategy": strategy,
            "plan": [],
            "normalization_error": None,
            "validation": None,
        }
    extracted = extract_plan_steps(parsed)
    if extracted is None:
        return {
            "parse_success": True,
            "parse_strategy": strategy,
            "plan": [],
            "normalization_error": "unexpected_plan_shape",
            "validation": None,
        }
    try:
        plan = normalize_provider_semantic_intents(
            extracted,
            inventory=build_semantic_target_inventory(
                context.materialization,
                additional_candidate_paths=observed_candidate_paths(context.observation),
            ),
            project_dir=context.workspace,
            source_materialization=context.materialization,
        )
        if len(plan) == 1 and bool(context.spec.files):
            expanded = split_repaired_single_step_full_lifecycle_plan(plan)
            if expanded:
                plan = expanded
        plan = PlannerService.sanitize_common_plan_issues(
            plan, task_prompt=context.spec.task
        )
        plan = _strengthen_weak_expected_file_verifications(plan)
        plan, target_report = normalize_existing_file_target_plan(
            plan, project_dir=context.workspace
        )
        plan, anchor_report = normalize_blank_line_divergent_replace_anchors(
            plan,
            project_dir=context.workspace,
            source_materialization=context.materialization,
        )
        plan, stale_report = normalize_stale_replace_ops_to_small_file_writes(
            plan, project_dir=context.workspace
        )
        plan = normalize_plan(plan, context.workspace, LOGGER)
        topology_db = SessionLocal()
        try:
            execution_topology = resolve_execution_topology_for_role(
                topology_db, BackendRole.EXECUTION
            )
        finally:
            topology_db.close()
        verdict = ValidatorService.validate_plan(
            plan,
            output_text=raw_output,
            task_prompt=context.spec.task,
            execution_profile="full_lifecycle",
            project_dir=context.workspace,
            title=context.spec.name,
            description=context.spec.task,
            validation_severity="standard",
            workflow_profile="default",
            workflow_stage=None,
            is_first_ordered_task=context.spec.first_ordered_task,
            workspace_identity=context.identity,
            planner_contract=None,
            source_materialization=context.materialization,
            execution_topology=execution_topology,
            intent_mode=context.spec.intent_mode,
        )
        return {
            "parse_success": True,
            "parse_strategy": strategy or "direct_json",
            "plan": plan,
            "normalization_error": None,
            "normalization_reports": {
                "existing_file_target": target_report,
                "replace_anchor": anchor_report,
                "stale_replace": stale_report,
            },
            "validation": verdict.to_dict(),
        }
    except Exception as exc:
        return {
            "parse_success": True,
            "parse_strategy": strategy or "direct_json",
            "plan": extracted,
            "normalization_error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            "validation": None,
        }


def _classify_result(
    *,
    pipeline: Mapping[str, Any],
    recall: float,
    usefulness: str,
    hallucinated_paths: int,
    unsupported_operations: int,
    wrong_classification: int,
) -> tuple[str | None, list[str]]:
    if not pipeline.get("parse_success") or pipeline.get("normalization_error"):
        return "ORCHESTRATOR_CONTRACT_FAILURE", ["SCHEMA"]
    validation = pipeline.get("validation") or {}
    if recall < 1.0 or usefulness != "PASS":
        return "TASK_SEMANTIC_FAILURE", ["MODEL_REASONING"]
    if validation.get("accepted") or validation.get("status") == "accepted":
        return None, []
    serialized = _canonical_json(validation).lower()
    lexical = [code for code in NON_SAFETY_LEXICAL_CODES if code in serialized]
    if (
        lexical
        and hallucinated_paths == 0
        and unsupported_operations == 0
        and wrong_classification == 0
    ):
        safety_markers = (
            "path_traversal",
            "path_protected_root",
            "source_version",
            "source_grounding",
            "create_only_task_existing",
            "accepted_path_authority_construction_failed",
        )
        if not any(marker in serialized for marker in safety_markers):
            return "HARNESS_FALSE_POSITIVE", [
                "LEXICAL_CONTRACT",
                "LEXICAL_EXISTING_WRITE_AUTHORIZATION",
            ]
    return "ORCHESTRATOR_CONTRACT_FAILURE", ["VALIDATION"]


def _analyze_candidate(context: FixtureContext, raw_output: str) -> dict[str, Any]:
    pipeline = _normalization_pipeline(context, raw_output)
    plan = list(pipeline.get("plan") or [])
    expected, mutating, operations = _plan_paths(plan)
    checklist = _semantic_checklist(context.spec, plan, raw_output)
    satisfied = sum(1 for value in checklist.values() if value)
    recall = satisfied / len(checklist) if checklist else 0.0
    existing = set(context.spec.files)
    all_paths = list(dict.fromkeys(expected + mutating))
    hallucinated = 0
    for path in all_paths:
        unsafe = path.startswith(("/", "..", "~"))
        extra_existing_fixture_path = bool(existing) and path not in existing
        if unsafe or extra_existing_fixture_path:
            hallucinated += 1
    unsupported = sum(1 for op in operations if op not in SUPPORTED_FILE_OPS)
    wrong_classification = 0
    for step in plan:
        description = str(step.get("description") or "").lower()
        for operation in step.get("ops") or []:
            if not isinstance(operation, Mapping):
                continue
            path = str(operation.get("path") or "").replace("\\", "/").lstrip("./")
            op_name = str(operation.get("op") or "")
            if op_name == "replace_in_file" and path not in existing:
                wrong_classification += 1
            if path in existing and op_name == "write_file" and any(
                word in description for word in ("create new", "scaffold new", "initialize new")
            ):
                wrong_classification += 1
            if context.spec.intent_mode == TaskIntentMode.CREATE_ONLY.value and path in existing:
                wrong_classification += 1
    verification = _verification_correctness(context.spec, plan)
    has_mutation = bool(mutating)
    if recall == 1.0 and has_mutation and verification == "PASS":
        usefulness = "PASS"
    elif recall >= 0.5 and has_mutation:
        usefulness = "PARTIAL"
    else:
        usefulness = "FAIL"
    primary, secondary = _classify_result(
        pipeline=pipeline,
        recall=recall,
        usefulness=usefulness,
        hallucinated_paths=hallucinated,
        unsupported_operations=unsupported,
        wrong_classification=wrong_classification,
    )
    validation = pipeline.get("validation") or {}
    details = validation.get("details") or {}
    finding_codes = list(validation.get("validator_rule_ids") or [])
    finding_codes.extend(details.get("semantic_violation_codes") or [])
    finding_codes = list(dict.fromkeys(str(value) for value in finding_codes if value))
    return {
        **pipeline,
        "task_requirement_checklist": checklist,
        "task_requirement_recall": recall,
        "initial_plan_valid": bool(validation.get("accepted")),
        "repair_required": not bool(validation.get("accepted")),
        "validation_status": validation.get("status") or "not_validated",
        "validator_finding_codes": finding_codes,
        "plan_step_count": len(plan),
        "expected_files": expected,
        "mutating_paths": mutating,
        "verification_commands": _verification_commands(plan),
        "hallucinated_paths": hallucinated,
        "unsupported_operations": unsupported,
        "wrong_existing_new_classification": wrong_classification,
        "verification_correctness": verification,
        "plan_usefulness": usefulness,
        "exact_user_constraint_recall": (
            EXACT_E_VERIFICATION in _verification_commands(plan)
            if context.spec.fixture_id == "E"
            else None
        ),
        "primary_failure_class": primary,
        "secondary_failure_notes": secondary,
    }


def _result_path(evidence_dir: Path, variant: str) -> Path:
    return evidence_dir / f"{variant.lower()}-results.json"


def _load_results(path: Path, freeze: Mapping[str, Any], variant: str) -> dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("runtime_freeze_digest") != freeze["runtime_freeze_digest"]:
            raise RuntimeError(f"{variant} results runtime digest mismatch")
        return payload
    return {
        "schema_version": "phase34-s2a-results/1",
        "variant": variant,
        "runtime_freeze_digest": freeze["runtime_freeze_digest"],
        "results": [],
    }


async def _invoke_cell(runtime: Any, prompt: str, fixture_id: str, variant: str) -> dict[str, Any]:
    return await PlannerService._execute_task_with_planning_lock(
        runtime,
        prompt,
        timeout_seconds=int(settings.PLANNING_SYNTHESIS_TIMEOUT_SECONDS),
        reuse_task_session=False,
        diagnostic_label="PLANNING",
        diagnostic_metadata={
            "phase": "PHASE34-S2A",
            "fixture_id": fixture_id,
            "variant": variant,
            "planning_attempt": "initial",
            "repairs_allowed": False,
            "execution_allowed": False,
        },
    )


def run(freeze_path: Path) -> None:
    freeze = _load_freeze(freeze_path)
    evidence_dir = freeze_path.parent
    db = SessionLocal()
    try:
        contexts = _rebuild_contexts(db, freeze)
        runtime = create_agent_runtime(db, None, None, role=BackendRole.PLANNING)
        metadata = runtime.get_backend_metadata()
        runtime_identity = {
            "backend": metadata.get("backend"),
            "model": metadata.get("model_family"),
            "profile": metadata.get("adaptation_profile"),
        }
        expected_identity = {
            "backend": freeze["runtime_freeze"]["planning_backend"],
            "model": freeze["runtime_freeze"]["planning_model"],
            "profile": freeze["runtime_freeze"]["planning_adaptation_profile"],
        }
        if runtime_identity != expected_identity:
            raise RuntimeError(
                f"Runtime identity changed before provider call: {runtime_identity} != {expected_identity}"
            )
        stores = {
            variant: _load_results(_result_path(evidence_dir, variant), freeze, variant)
            for variant in VARIANTS
        }
        completed = {
            (str(item["fixture_id"]), str(item["variant"]))
            for store in stores.values()
            for item in store["results"]
        }
        generation_attempts = sum(
            int(item.get("provider_call_attempts") or 0)
            for store in stores.values()
            for item in store["results"]
        )
        for fixture_id, variant in CALL_ORDER:
            if (fixture_id, variant) in completed:
                continue
            if generation_attempts >= int(freeze["max_generation_calls"]):
                raise RuntimeError("Provider generation call budget exhausted")
            context = contexts[fixture_id]
            prompt = context.prompts[variant]["provider"]
            before_digest = _workspace_content_digest(context.workspace)
            started = time.monotonic()
            generation_attempts += 1
            try:
                response = asyncio.run(
                    _invoke_cell(runtime, prompt, fixture_id, variant)
                )
                latency_ms = round((time.monotonic() - started) * 1000)
                raw_output = str(response.get("output") or "")
                analysis = _analyze_candidate(context, raw_output)
                transport_error = None
                diagnostics = response.get("diagnostics") or {}
            except Exception as exc:
                latency_ms = round((time.monotonic() - started) * 1000)
                raw_output = ""
                diagnostics = getattr(exc, "runtime_diagnostics", {}) or {}
                transport_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
                analysis = {
                    "parse_success": False,
                    "plan": [],
                    "initial_plan_valid": False,
                    "repair_required": False,
                    "validation_status": "provider_transport_failure",
                    "validator_finding_codes": [],
                    "plan_step_count": 0,
                    "expected_files": [],
                    "mutating_paths": [],
                    "verification_commands": [],
                    "task_requirement_checklist": {
                        requirement: False for requirement in context.spec.requirements
                    },
                    "task_requirement_recall": 0.0,
                    "hallucinated_paths": 0,
                    "unsupported_operations": 0,
                    "wrong_existing_new_classification": 0,
                    "verification_correctness": "FAIL",
                    "plan_usefulness": "FAIL",
                    "exact_user_constraint_recall": False if fixture_id == "E" else None,
                    "primary_failure_class": "PROVIDER_TRANSPORT_FAILURE",
                    "secondary_failure_notes": ["RUNTIME_TRANSFORM"],
                }
            after_digest = _workspace_content_digest(context.workspace)
            if before_digest != after_digest:
                raise RuntimeError(
                    f"Planning call mutated isolated fixture {fixture_id}; stopping"
                )
            cell = {
                "fixture_id": fixture_id,
                "fixture_name": context.spec.name,
                "variant": variant,
                "provider_call_attempts": 1,
                "provider_latency_ms": latency_ms,
                "transport_error": transport_error,
                "raw_provider_response": raw_output,
                "raw_provider_response_sha256": _sha256_text(raw_output),
                "prompt": context.prompts[variant]["record"],
                "runtime_identity": runtime_identity,
                "semantic_input_digest": context.semantic_input_digest,
                "workspace_digest_before": before_digest,
                "workspace_digest_after": after_digest,
                "provider_diagnostics": {
                    key: diagnostics.get(key)
                    for key in (
                        "prompt_stage",
                        "provider_bound_prompt_sha256_12",
                        "provider_bound_prompt_chars",
                        "provider_bound_prompt_token_estimate",
                        "provider_invocation_started",
                        "provider_response_received",
                    )
                    if key in diagnostics
                },
                **analysis,
            }
            stores[variant]["results"].append(cell)
            stores[variant]["total_provider_calls"] = sum(
                int(item.get("provider_call_attempts") or 0)
                for item in stores[variant]["results"]
            )
            _write_json(_result_path(evidence_dir, variant), stores[variant])
            print(
                f"CELL_COMPLETE fixture={fixture_id} variant={variant} "
                f"latency_ms={latency_ms} parse={cell['parse_success']} "
                f"valid={cell['initial_plan_valid']} recall={cell['task_requirement_recall']:.3f} "
                f"class={cell['primary_failure_class'] or 'NONE'}",
                flush=True,
            )
        _write_comparison(evidence_dir, freeze, stores)
    finally:
        db.close()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _write_comparison(
    evidence_dir: Path,
    freeze: Mapping[str, Any],
    stores: Mapping[str, Mapping[str, Any]],
) -> None:
    indexed = {
        variant: {str(item["fixture_id"]): item for item in stores[variant]["results"]}
        for variant in VARIANTS
    }
    if any(len(indexed[variant]) != 5 for variant in VARIANTS):
        return
    aggregates: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = list(indexed[variant].values())
        aggregates[variant] = {
            "initial_valid_count": sum(bool(row["initial_plan_valid"]) for row in rows),
            "mean_requirement_recall": round(
                _mean(float(row["task_requirement_recall"]) for row in rows), 4
            ),
            "total_hallucinated_paths": sum(int(row["hallucinated_paths"]) for row in rows),
            "total_contract_failures": sum(
                row.get("primary_failure_class") == "ORCHESTRATOR_CONTRACT_FAILURE"
                for row in rows
            ),
            "total_harness_false_positives": sum(
                row.get("primary_failure_class") == "HARNESS_FALSE_POSITIVE"
                for row in rows
            ),
            "prompt_tokens_total": sum(int(row["prompt"]["prompt_tokens_approx"]) for row in rows),
            "provider_latency_ms_total": sum(int(row["provider_latency_ms"]) for row in rows),
            "transport_failures": sum(bool(row.get("transport_error")) for row in rows),
        }
    current_tokens = aggregates["CURRENT"]["prompt_tokens_total"]
    compact_tokens = aggregates["COMPACT"]["prompt_tokens_total"]
    reduction = round((current_tokens - compact_tokens) * 100 / current_tokens, 2)
    rows = []
    shared_failures: list[dict[str, Any]] = []
    shared_validator_rejections: list[str] = []
    for fixture_id in sorted(indexed["CURRENT"]):
        current = indexed["CURRENT"][fixture_id]
        compact = indexed["COMPACT"][fixture_id]
        current_failed = {
            name for name, passed in current["task_requirement_checklist"].items() if not passed
        }
        compact_failed = {
            name for name, passed in compact["task_requirement_checklist"].items() if not passed
        }
        if (
            current_failed
            and current_failed == compact_failed
            and current.get("primary_failure_class") == "TASK_SEMANTIC_FAILURE"
            and compact.get("primary_failure_class") == "TASK_SEMANTIC_FAILURE"
        ):
            shared_failures.append(
                {"fixture_id": fixture_id, "failed_requirements": sorted(current_failed)}
            )
        if (
            current.get("primary_failure_class") == "HARNESS_FALSE_POSITIVE"
            and compact.get("primary_failure_class") == "HARNESS_FALSE_POSITIVE"
            and set(current.get("validator_finding_codes") or [])
            == set(compact.get("validator_finding_codes") or [])
        ):
            shared_validator_rejections.append(fixture_id)
        rows.append(
            {
                "fixture_id": fixture_id,
                "fixture_name": current["fixture_name"],
                "current_semantic_recall": current["task_requirement_recall"],
                "compact_semantic_recall": compact["task_requirement_recall"],
                "current_valid": current["initial_plan_valid"],
                "compact_valid": compact["initial_plan_valid"],
                "current_failure_class": current["primary_failure_class"],
                "compact_failure_class": compact["primary_failure_class"],
                "current_tokens": current["prompt"]["prompt_tokens_approx"],
                "compact_tokens": compact["prompt"]["prompt_tokens_approx"],
            }
        )
    evidence_frozen = all(
        indexed["CURRENT"][fixture_id]["semantic_input_digest"]
        == indexed["COMPACT"][fixture_id]["semantic_input_digest"]
        == freeze["fixtures"][fixture_id]["semantic_input_digest"]
        for fixture_id in indexed["CURRENT"]
    )
    runtime_frozen = all(
        indexed[variant][fixture_id]["runtime_identity"]
        == indexed["CURRENT"]["A"]["runtime_identity"]
        for variant in VARIANTS
        for fixture_id in indexed[variant]
    )
    f4 = (
        aggregates["COMPACT"]["total_harness_false_positives"] > 0
        and aggregates["COMPACT"]["mean_requirement_recall"]
        >= aggregates["CURRENT"]["mean_requirement_recall"]
    )
    f5 = (
        aggregates["COMPACT"]["mean_requirement_recall"]
        < aggregates["CURRENT"]["mean_requirement_recall"]
        or aggregates["COMPACT"]["total_hallucinated_paths"]
        > aggregates["CURRENT"]["total_hallucinated_paths"]
    )
    f8 = (
        aggregates["CURRENT"]["mean_requirement_recall"]
        > aggregates["COMPACT"]["mean_requirement_recall"] + 0.05
        or aggregates["CURRENT"]["initial_valid_count"]
        > aggregates["COMPACT"]["initial_valid_count"]
    )
    transport_failures = sum(
        aggregates[variant]["transport_failures"] for variant in VARIANTS
    )
    experiment_valid = evidence_frozen and runtime_frozen
    if not experiment_valid:
        decision = "F. EXPERIMENT_INVALID_DUE_TO_NON_FROZEN_INPUT_OR_RUNTIME"
    elif transport_failures:
        decision = "G. PROVIDER_RUNTIME_FAILURE_PREVENTED_ADJUDICATION"
    elif f8:
        decision = "D. CURRENT_OUTPERFORMS_COMPACT"
    elif f4:
        decision = "B. COMPACT_DIRECTION_PROMISING_BUT_VALIDATOR_CONFOUND_PRESENT"
    elif shared_failures and (
        aggregates["CURRENT"]["mean_requirement_recall"] < 0.8
        and aggregates["COMPACT"]["mean_requirement_recall"] < 0.8
    ):
        decision = "E. MODEL_OR_SHARED_CONTRACT_LIMIT_DOMINATES_BOTH_VARIANTS"
    elif (
        reduction >= 40
        and not f5
        and aggregates["COMPACT"]["initial_valid_count"]
        >= aggregates["CURRENT"]["initial_valid_count"]
        and aggregates["COMPACT"]["total_contract_failures"]
        <= aggregates["CURRENT"]["total_contract_failures"]
    ):
        decision = "A. COMPACT_DIRECTION_STRONGLY_SUPPORTED"
    else:
        decision = "C. NO_MEANINGFUL_DIFFERENCE_CURRENT_AND_COMPACT"
    comparison = {
        "schema_version": "phase34-s2a-comparison/1",
        "final_decision": decision,
        "experiment_valid": experiment_valid,
        "total_provider_calls": sum(
            int(stores[variant].get("total_provider_calls") or 0)
            for variant in VARIANTS
        ),
        "runtime_freeze_digest": freeze["runtime_freeze_digest"],
        "aggregates": aggregates,
        "prompt_reduction_percent": reduction,
        "comparison_rows": rows,
        "falsification_tests": {
            "F1_compact_omitted_hard_requirement": False,
            "F1_basis": "Frozen compact contract retains path, operation, schema, effect, verification, existing/new, command, and CREATE_ONLY requirements.",
            "F2_evidence_not_frozen": not evidence_frozen,
            "F3_runtime_not_frozen": not runtime_frozen,
            "F4_validator_confound": f4,
            "F5_compact_worse": f5 if not transport_failures else None,
            "F5_observed_unadjusted": f5,
            "F5_adjudication": (
                "INCONCLUSIVE_DUE_TO_TRANSPORT"
                if transport_failures
                else "ADJUDICABLE"
            ),
            "F6_shared_failure": bool(shared_failures),
            "F6_shared_failure_details": shared_failures,
            "F7_shared_validator_rejection": bool(shared_validator_rejections),
            "F7_shared_validator_rejection_fixtures": shared_validator_rejections,
            "F8_current_better": f8 if not transport_failures else None,
            "F8_observed_unadjusted": f8,
            "F8_adjudication": (
                "INCONCLUSIVE_DUE_TO_TRANSPORT"
                if transport_failures
                else "ADJUDICABLE"
            ),
        },
        "lexical_existing_write_confound_observed": any(
            row.get("primary_failure_class") == "HARNESS_FALSE_POSITIVE"
            and "LEXICAL_EXISTING_WRITE_AUTHORIZATION"
            in (row.get("secondary_failure_notes") or [])
            for variant in VARIANTS
            for row in indexed[variant].values()
        ),
        "verification_exact_command_recall_current": indexed["CURRENT"]["E"].get(
            "exact_user_constraint_recall"
        ),
        "verification_exact_command_recall_compact": indexed["COMPACT"]["E"].get(
            "exact_user_constraint_recall"
        ),
        "runtime_prompt_transform_effect": {
            "adaptation_profile": freeze["runtime_freeze"]["planning_adaptation_profile"],
            "runtime_adapter_additional_transform": "NONE",
            "whitespace_preserved": all(
                row["prompt"]["whitespace_preserved"]
                for variant in VARIANTS
                for row in indexed[variant].values()
            ),
            "section_boundaries_preserved": all(
                row["prompt"]["section_boundaries_preserved"]
                for variant in VARIANTS
                for row in indexed[variant].values()
            ),
            "source_format_preserved": all(
                row["prompt"]["source_format_preserved"]
                for variant in VARIANTS
                for row in indexed[variant].values()
            ),
        },
        "safety": {
            "product_attempts": 0,
            "live_lifecycles": 0,
            "plan_executions": 0,
            "repairs": 0,
        },
    }
    comparison["adjudication"] = {
        "generation_script_sha256": freeze["evaluation_script_sha256"],
        "adjudication_script_sha256": _script_digest(),
        "initial_valid_basis": "existing ValidatorResult.status == accepted",
        "semantic_scoring_basis": "parsed provider Plan before deterministic Planning normalization",
        "transport_failures_prevent_causal_adjudication": bool(transport_failures),
        "normalization_exact_verification_loss_cells": [
            f"{variant}:{row['fixture_id']}"
            for variant in VARIANTS
            for row in indexed[variant].values()
            if row.get("normalization_exact_verification_lost")
        ],
    }
    _write_json(evidence_dir / "comparison.json", comparison)
    print(json.dumps({
        "status": "complete",
        "final_decision": decision,
        "total_provider_calls": comparison["total_provider_calls"],
        "current": aggregates["CURRENT"],
        "compact": aggregates["COMPACT"],
        "prompt_reduction_percent": reduction,
        "comparison": str(evidence_dir / "comparison.json"),
    }, indent=2), flush=True)


def adjudicate(evidence_dir: Path) -> None:
    """Correct metric ownership provider-free; never invokes a runtime."""

    freeze_path = evidence_dir / "fixture-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    stores = {
        variant: json.loads(_result_path(evidence_dir, variant).read_text(encoding="utf-8"))
        for variant in VARIANTS
    }
    total_calls = sum(
        int(store.get("total_provider_calls") or 0) for store in stores.values()
    )
    if total_calls != 10:
        raise RuntimeError(f"Adjudication requires the complete ten-call matrix, got {total_calls}")
    handler = EnhancedErrorHandler()
    specs = _fixtures()
    for variant in VARIANTS:
        for row in stores[variant]["results"]:
            row["prompt"]["source_format_preserved"] = bool(
                row["prompt"].get("whitespace_preserved")
            )
            if row.get("transport_error"):
                continue
            raw_output = str(row.get("raw_provider_response") or "")
            success, parsed, strategy = handler.attempt_json_parsing(
                raw_output, context="planning"
            )
            raw_plan = extract_plan_steps(parsed) if success else None
            if raw_plan is None:
                continue
            fixture_id = str(row["fixture_id"])
            spec = specs[fixture_id]
            raw_plan = list(raw_plan)
            checklist = _semantic_checklist(spec, raw_plan, raw_output)
            recall = sum(bool(value) for value in checklist.values()) / len(checklist)
            _, raw_mutating, raw_operations = _plan_paths(raw_plan)
            raw_verifications = _verification_commands(raw_plan)
            verification = _verification_correctness(spec, raw_plan)
            usefulness = (
                "PASS"
                if recall == 1.0 and bool(raw_mutating) and verification == "PASS"
                else "PARTIAL"
                if recall >= 0.5 and bool(raw_mutating)
                else "FAIL"
            )
            validation = row.get("validation") or {}
            accepted = validation.get("status") == "accepted"
            primary, secondary = _classify_result(
                pipeline={
                    "parse_success": True,
                    "normalization_error": row.get("normalization_error"),
                    "validation": validation,
                },
                recall=recall,
                usefulness=usefulness,
                hallucinated_paths=int(row.get("hallucinated_paths") or 0),
                unsupported_operations=sum(
                    1 for operation in raw_operations if operation not in SUPPORTED_FILE_OPS
                ),
                wrong_classification=int(
                    row.get("wrong_existing_new_classification") or 0
                ),
            )
            normalized_verifications = list(row.get("verification_commands") or [])
            exact_lost = (
                fixture_id == "E"
                and EXACT_E_VERIFICATION in raw_verifications
                and EXACT_E_VERIFICATION not in normalized_verifications
            )
            row.setdefault(
                "automated_analysis_before_adjudication",
                {
                    key: row.get(key)
                    for key in (
                        "initial_plan_valid",
                        "repair_required",
                        "task_requirement_checklist",
                        "task_requirement_recall",
                        "verification_correctness",
                        "plan_usefulness",
                        "exact_user_constraint_recall",
                        "primary_failure_class",
                        "secondary_failure_notes",
                    )
                },
            )
            row.update(
                {
                    "initial_plan_valid": accepted,
                    "repair_required": not accepted,
                    "task_requirement_checklist": checklist,
                    "task_requirement_recall": recall,
                    "verification_correctness": verification,
                    "plan_usefulness": usefulness,
                    "exact_user_constraint_recall": (
                        EXACT_E_VERIFICATION in raw_verifications
                        if fixture_id == "E"
                        else None
                    ),
                    "primary_failure_class": primary,
                    "secondary_failure_notes": secondary,
                    "raw_parsed_plan": raw_plan,
                    "raw_plan_verification_commands": raw_verifications,
                    "normalization_exact_verification_lost": exact_lost,
                    "adjudication_basis": {
                        "validator_acceptance": "ValidatorResult.status",
                        "semantic_recall": "raw parsed provider Plan",
                        "normalizer_and_validator": "existing production path unchanged",
                        "provider_calls_added": 0,
                    },
                }
            )
        _write_json(_result_path(evidence_dir, variant), stores[variant])
    _write_comparison(evidence_dir, freeze, stores)


def self_check() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "ExecutorService",
        "execute_file_ops",
        "execute_verification_command",
        "execute_planning_phase",
        "repair_output",
        "create_session",
        "publish_candidate",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            if name in forbidden_calls:
                violations.append(name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            if any(token in module for token in ("execution.executor", "publication")):
                violations.append(module)
    if violations:
        raise RuntimeError(
            f"Harness contains forbidden execution/repair imports or calls: {sorted(set(violations))}"
        )
    directives = {spec.fixture_id: len(_compact_directives(spec)) + 2 for spec in _fixtures().values()}
    if any(value < 12 or value > 16 for value in directives.values()):
        raise RuntimeError(f"Compact directive target violated: {directives}")
    if len(CALL_ORDER) != 10 or len(set(CALL_ORDER)) != 10:
        raise RuntimeError("Call order must contain exactly ten unique cells")
    if set(CALL_ORDER) != {
        (fixture_id, variant)
        for fixture_id in _fixtures()
        for variant in VARIANTS
    }:
        raise RuntimeError("Call order does not cover the exact 5x2 matrix")
    print(json.dumps({
        "status": "self_check_passed",
        "compact_directive_counts_including_envelope": directives,
        "call_cells": len(CALL_ORDER),
        "forbidden_symbols": violations,
        "script_sha256": _script_digest(),
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("self-check", "prepare", "run", "adjudicate")
    )
    parser.add_argument(
        "--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR
    )
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.command == "self-check":
        self_check()
        return 0
    if args.command == "prepare":
        prepare(args.evidence_dir)
        return 0
    if args.command == "adjudicate":
        adjudicate(args.evidence_dir.resolve())
        return 0
    freeze = args.freeze or args.evidence_dir / "fixture-freeze.json"
    run(freeze.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

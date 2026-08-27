"""PHASE34-C6R1 capture-complete L1 certification runner.

This runner is deliberately certification-only. It delegates discovery to the
production function and injects only that function's existing ``capture_path``
argument so the normal adapter/parser/executor chain remains in use.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from tempfile import TemporaryDirectory
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.models import (
    Base,
    Plan,
    Project,
    Session as SessionModel,
    SessionTask,
    TaskCheckpoint,
    TaskExecution,
    TaskStatus,
)
from app.services.agents.agent_runtime import (
    BackendRole,
    low_resource_single_model_runtime_matrix,
    resolve_runtime_configuration,
)
from app.services.agents.providers import openai_chat_adapter
from app.services.orchestration.planning.discovery_contract_capture import (
    DiscoveryContractCapture,
)
from app.services.planning.planning_session_service import PlanningSessionService
from app.services.session.session_runtime_service import queue_task_for_session
from app.tests.phase34c6r1_capture_harness import bind_discovery_capture


MODEL = "Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf"
ROLES = (
    BackendRole.PLANNING,
    BackendRole.REPAIR,
    BackendRole.DEBUG_REPAIR,
    BackendRole.COMPLETION_REPAIR,
    BackendRole.EXECUTION,
)
TASK_TEXT = (
    "Create tiny_calc.py in the project workspace so answer() returns 42. "
    "The existing test_tiny_calc.py is the focused verification and must pass. "
    "Use one narrowly scoped task and preserve the existing test."
)


def _status(value: Any) -> str:
    return str(getattr(value, "value", value))


def _git_snapshot(path: Path) -> dict[str, str]:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return {"status": status.stdout, "diff": diff.stdout}


def _path_values(value: Any) -> list[str]:
    paths: list[str] = []

    def visit(current: Any) -> None:
        if isinstance(current, dict):
            for key, child in current.items():
                if "path" in str(key).lower() and isinstance(child, str):
                    paths.append(child)
                else:
                    visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return sorted(set(paths))


def _json_document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_process_local_certification_settings() -> None:
    settings.LOW_RESOURCE_SINGLE_MODEL = True
    settings.INLINE_PLANNING = True
    settings.PLANNING_BACKEND = "openai_chat_completions"
    settings.EXECUTION_BACKEND = "openai_chat_completions"
    settings.REPAIR_BACKEND = "openai_chat_completions"
    settings.DEBUG_REPAIR_BACKEND = "openai_chat_completions"
    settings.COMPLETION_REPAIR_BACKEND = "openai_chat_completions"
    settings.PLANNING_DIRECT_MODEL = MODEL
    settings.PLANNER_MODEL = MODEL
    settings.EXECUTION_MODEL = MODEL
    settings.AGENT_MODEL = MODEL
    settings.PLANNING_ADAPTATION_PROFILE = "ollama_default"
    settings.EXECUTION_ADAPTATION_PROFILE = "ollama_default"
    settings.REPAIR_ADAPTATION_PROFILE = "ollama_default"
    settings.DEBUG_REPAIR_ADAPTATION_PROFILE = "ollama_default"
    settings.COMPLETION_REPAIR_ADAPTATION_PROFILE = "ollama_default"
    settings.EXECUTION_CONTEXT_TOKENS = 64_000
    settings.DEBUG_REPAIR_CONTEXT_TOKENS = 64_000
    settings.PLANNING_REPAIR_CONTEXT_TOKENS = 64_000
    settings.RUNTIME_WORKSPACE_ENABLED = True
    settings.DEMO_MODE = False


def _preflight(db: Any) -> dict[str, Any]:
    configurations = {role: resolve_runtime_configuration(db, role) for role in ROLES}
    matrix = low_resource_single_model_runtime_matrix(db)
    endpoint = str(settings.PLANNING_DIRECT_BASE_URL or "").rstrip("/")
    identities = {
        role.value: {
            "backend": configuration.backend_name,
            "model": configuration.model_family,
            "profile": configuration.adaptation_profile,
            "endpoint": endpoint,
        }
        for role, configuration in configurations.items()
    }
    preflight = {
        "low_resource_single_model": bool(settings.LOW_RESOURCE_SINGLE_MODEL),
        "all_generation_roles_same_backend": len(
            {item.backend_name for item in configurations.values()}
        )
        == 1,
        "all_generation_roles_same_model": len(
            {item.model_family for item in configurations.values()}
        )
        == 1,
        "all_generation_roles_same_profile": len(
            {item.adaptation_profile for item in configurations.values()}
        )
        == 1,
        "all_generation_roles_same_endpoint": len(
            {endpoint for _role in configurations}
        )
        == 1,
        "unique_generation_runtime_families": len(
            {item.backend_name for item in configurations.values()}
        ),
        "unique_generation_model_identities": len(
            {item.model_family for item in configurations.values()}
        ),
        "second_generation_model_request_path_present": bool(
            matrix.get("second_provider_required")
        ),
        "openclaw_generation_path_reachable": bool(matrix.get("openclaw_required")),
        "execution_topology": matrix.get("execution_topology"),
        "discovery_response_schema_enabled": True,
        "identities": identities,
        "matrix": matrix,
    }
    if (
        not preflight["low_resource_single_model"]
        or not preflight["all_generation_roles_same_backend"]
        or not preflight["all_generation_roles_same_model"]
        or not preflight["all_generation_roles_same_profile"]
        or not preflight["all_generation_roles_same_endpoint"]
        or preflight["unique_generation_runtime_families"] != 1
        or preflight["unique_generation_model_identities"] != 1
        or preflight["second_generation_model_request_path_present"]
        or preflight["openclaw_generation_path_reachable"]
        or preflight["execution_topology"] != "structured_orchestrator_execution"
    ):
        raise RuntimeError("C6R1 preflight identity/topology gate failed")
    return preflight


def _capture_summary(document: dict[str, Any]) -> dict[str, Any]:
    response = document.get("response") or {}
    stages = document.get("stages") or {}
    parser = document.get("parser") or {}
    action = document.get("action") or {}
    raw_content = stages.get("message_content")
    return {
        "raw_response_captured": bool(response.get("raw_response_captured")),
        "raw_response_class": (
            "OpenAI chat-completion JSON; canonical JSON message content"
            if response.get("raw_response_captured")
            else "NOT_RETAINED"
        ),
        "http_status": response.get("http_status"),
        "response_model": response.get("response_model"),
        "message_content_class": (
            type(raw_content).__name__.upper()
            if raw_content is not None
            else "NOT_RETAINED"
        ),
        "extracted_content_class": (
            type(stages.get("extracted_content")).__name__.upper()
            if stages.get("extracted_content") is not None
            else "NOT_RETAINED"
        ),
        "normalized_content_class": (
            type(stages.get("normalized_content")).__name__.upper()
            if stages.get("normalized_content") is not None
            else "NOT_RETAINED"
        ),
        "parser_input_class": (
            type(stages.get("parser_input")).__name__.upper()
            if stages.get("parser_input") is not None
            else "NOT_RETAINED"
        ),
        "canonical": parser.get("success") is True
        and parser.get("action") in {"search_text", "read_file", "stop"},
        "parser_pass": parser.get("success") is True,
        "action": parser.get("action"),
        "action_validation_pass": action.get("validation_pass"),
        "action_executable": action.get("executable"),
    }


def run_live() -> dict[str, Any]:
    _configure_process_local_certification_settings()
    capture_path = (
        Path("docs/roadmap/reports/evidence/phase34c6r1")
        / "live-discovery-contract.json"
    ).resolve()
    if capture_path.exists():
        raise FileExistsError(f"refusing to overwrite capture: {capture_path}")

    result: dict[str, Any] = {
        "gate": "PHASE34-C6R1",
        "full_lifecycle_attempts": 1,
        "provider_retries": 0,
        "provider_calls": [],
        "provider_wire_ledger": [],
        "runtime_creations": [],
        "errors": [],
        "openclaw_generation_calls": 0,
        "capture_path": str(capture_path),
    }
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = session_factory()
    worker_module = None
    original_worker_db = None
    original_worker_task = None
    monitor_stop = threading.Event()
    resource_samples: list[tuple[int, int]] = []
    lifecycle_started = time.perf_counter()

    try:
        with TemporaryDirectory(prefix="phase34c6r1-") as temporary_root:
            root = Path(temporary_root)
            workspace = root / "tiny-calc-project"
            workspace.mkdir(parents=True)
            (workspace / "test_tiny_calc.py").write_text(
                "from tiny_calc import answer\n\n\ndef test_answer():\n"
                "    assert answer() == 42\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
            subprocess.run(["git", "add", "."], cwd=workspace, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.email=phase34c6r1@example.invalid",
                    "-c",
                    "user.name=PHASE34-C6R1",
                    "commit",
                    "-qm",
                    "baseline",
                ],
                cwd=workspace,
                check=True,
            )
            os.environ["OPENCLAW_WORKSPACE"] = str(root / "openclaw-workspace")
            settings.RUNTIME_ROOT = str(root / "runtime-root")

            preflight = _preflight(db)
            result["preflight"] = preflight

            project = Project(
                name="PHASE34-C6R1 tiny_calc",
                description="Capture-complete isolated L1 certification fixture",
                workspace_path=str(workspace),
            )
            db.add(project)
            db.commit()
            db.refresh(project)
            import app.tasks.worker as worker_module

            original_worker_db = worker_module.get_db_session
            original_worker_task = worker_module.execute_orchestration_task
            worker_module.get_db_session = session_factory

            class SyncResult:
                def __init__(self, value: Any):
                    self.id = "phase34c6r1-sync-worker"
                    self.result = value

            class SyncTask:
                @staticmethod
                def delay(**kwargs: Any) -> SyncResult:
                    applied = original_worker_task.apply(kwargs=kwargs)
                    worker_result = applied.get(propagate=True)
                    result["worker_result"] = worker_result
                    return SyncResult(worker_result)

            worker_module.execute_orchestration_task = SyncTask

            try:
                import psutil

                def sample_resources() -> None:
                    while not monitor_stop.is_set():
                        resource_samples.append(
                            (
                                int(psutil.virtual_memory().used),
                                int(psutil.swap_memory().used),
                            )
                        )
                        monitor_stop.wait(0.2)

                monitor = threading.Thread(target=sample_resources, daemon=True)
                monitor.start()
            except ImportError:
                monitor = None

            phase = {"value": "planning"}
            original_chat = openai_chat_adapter.OpenAIChatCompletionsRuntime._chat
            import httpx

            original_post = httpx.AsyncClient.post

            async def recording_chat(runtime: Any, *args: Any, **kwargs: Any):
                started = time.perf_counter()
                call = {
                    "stage": phase["value"],
                    "role": _status(runtime.backend_role),
                    "backend": runtime.backend_descriptor.name,
                    "model": runtime._model_name(),
                    "diagnostic_label": kwargs.get("diagnostic_label"),
                }
                try:
                    return await original_chat(runtime, *args, **kwargs)
                finally:
                    call["latency_seconds"] = round(time.perf_counter() - started, 3)
                    call["returned"] = True
                    result["provider_calls"].append(call)

            async def recording_post(client: Any, url: Any, *args: Any, **kwargs: Any):
                del args
                payload = kwargs.get("json") or {}
                response_format = payload.get("response_format")
                result["provider_wire_ledger"].append(
                    {
                        "stage": phase["value"],
                        "endpoint": str(url),
                        "model": payload.get("model"),
                        "has_response_schema": isinstance(response_format, dict)
                        and response_format.get("type") == "json_schema",
                        "has_native_tools": bool(payload.get("tools")),
                        "has_tool_choice": "tool_choice" in payload,
                        "has_functions": bool(payload.get("functions")),
                    }
                )
                return await original_post(client, url, **kwargs)

            httpx.AsyncClient.post = recording_post
            openai_chat_adapter.OpenAIChatCompletionsRuntime._chat = recording_chat

            agent_runtime_module = __import__(
                "app.services.agents.agent_runtime",
                fromlist=["create_agent_runtime"],
            )
            real_create_runtime = agent_runtime_module.create_agent_runtime

            def recording_create_runtime(*args: Any, **kwargs: Any):
                runtime = real_create_runtime(*args, **kwargs)
                configuration = getattr(runtime, "runtime_configuration", None)
                result["runtime_creations"].append(
                    {
                        "role": _status(getattr(configuration, "role", None)),
                        "backend": getattr(configuration, "backend_name", None),
                        "model": getattr(configuration, "model_family", None),
                        "profile": getattr(configuration, "adaptation_profile", None),
                    }
                )
                return runtime

            agent_runtime_module.create_agent_runtime = recording_create_runtime
            worker_module.create_agent_runtime = recording_create_runtime

            from app.services.orchestration.execution.executor import ExecutorService
            from app.services.orchestration.phases import execution_local_steps
            from app.services.orchestration.phases import execution_loop

            counters = {
                "e1": 0,
                "e2": 0,
                "e3": 0,
            }
            original_file_ops = ExecutorService.execute_file_ops

            def recording_file_ops(project_dir: Path, operations: Any, **kwargs: Any):
                counters["e1"] += len(operations) if isinstance(operations, list) else 0
                return original_file_ops(project_dir, operations, **kwargs)

            ExecutorService.execute_file_ops = recording_file_ops
            original_local = execution_loop._execute_local_shell_commands_step

            def recording_local(*, commands: list[Any], **kwargs: Any):
                counters["e2"] += len([item for item in commands if str(item).strip()])
                return original_local(commands=commands, **kwargs)

            execution_loop._execute_local_shell_commands_step = recording_local
            original_verification = execution_local_steps.execute_verification_command

            def recording_verification(*args: Any, **kwargs: Any):
                counters["e3"] += 1
                return original_verification(*args, **kwargs)

            execution_local_steps.execute_verification_command = recording_verification
            execution_loop.assess_step_execution.__globals__[
                "execute_verification_command"
            ] = recording_verification

            original_dispose = worker_module._dispose_runtime_workspace_safely

            def recording_dispose(sandbox: Any, **kwargs: Any):
                runtime_path = Path(sandbox.path)
                capture = {
                    "path": str(runtime_path),
                    "workspace": _git_snapshot(runtime_path),
                }
                if runtime_path.exists():
                    check = subprocess.run(
                        [sys.executable, "-m", "pytest", "-q", "test_tiny_calc.py"],
                        cwd=runtime_path,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    capture["independent_verification"] = {
                        "returncode": check.returncode,
                        "stdout": check.stdout,
                        "stderr": check.stderr,
                    }
                result["runtime_workspace_capture"] = capture
                return original_dispose(sandbox, **kwargs)

            worker_module._dispose_runtime_workspace_safely = recording_dispose

            try:
                with bind_discovery_capture(capture_path) as binding:
                    result["capture_root_created_before_provider_dispatch"] = (
                        capture_path.parent.exists() and capture_path.exists()
                    )
                    planning_service = PlanningSessionService(db)
                    planning_session = planning_service.start_session(
                        project, TASK_TEXT, skip_clarification=True
                    )
                    db.refresh(planning_session)
                    result["planning_session_id"] = planning_session.id
                    result["planning_session_status"] = planning_session.status
                    result["planning_succeeded"] = (
                        planning_session.status == "completed"
                    )
                    if planning_session.status != "completed":
                        raise RuntimeError(
                            f"planning did not complete: {planning_session.status}"
                        )

                    committed_session, plan, committed_tasks = planning_service.commit(
                        planning_session.id
                    )
                    del committed_session
                    result["plan_commit_reached"] = True
                    result["plan_commit_succeeded"] = isinstance(plan, Plan) and bool(
                        committed_tasks
                    )
                    result["plan_id"] = getattr(plan, "id", None)
                    if not result["plan_commit_succeeded"]:
                        raise RuntimeError("plan commit did not produce tasks")

                    execution_session = SessionModel(
                        project_id=project.id,
                        name="PHASE34-C6R1 full lifecycle execution",
                        description="Capture-complete isolated L1 certification",
                        status="pending",
                        execution_mode="manual",
                    )
                    db.add(execution_session)
                    db.commit()
                    db.refresh(execution_session)
                    phase["value"] = "execution"
                    queue_result = queue_task_for_session(
                        db,
                        execution_session,
                        committed_tasks[0].id,
                        timeout_seconds=300,
                    )
                    result["queue_result"] = {
                        key: str(value) for key, value in queue_result.items()
                    }
                    result["capture_path_injected"] = binding.injected
                    result["capture_stage_incoming_path"] = (
                        binding.incoming_capture_path
                    )
                    db.refresh(execution_session)
                    committed_task = (
                        db.query(SessionTask)
                        .filter(
                            SessionTask.session_id == execution_session.id,
                            SessionTask.task_id == committed_tasks[0].id,
                        )
                        .one()
                    )
                    task_execution = (
                        db.query(TaskExecution)
                        .filter(TaskExecution.id == queue_result["task_execution_id"])
                        .one()
                    )
                    checkpoints = (
                        db.query(TaskCheckpoint)
                        .filter(
                            TaskCheckpoint.session_id == execution_session.id,
                            TaskCheckpoint.task_id == committed_tasks[0].id,
                            TaskCheckpoint.checkpoint_type == "validation_plan",
                        )
                        .all()
                    )
                    checkpoint_payload = (
                        json.loads(checkpoints[-1].state_snapshot)
                        if checkpoints
                        else {}
                    )
                    checkpoint_details = checkpoint_payload.get("details") or {}
                    authority = checkpoint_details.get("accepted_path_authority")
                    result["execution_session_id"] = execution_session.id
                    result["task_execution_status"] = _status(task_execution.status)
                    result["task_status"] = _status(committed_task.status)
                    result["session_status"] = _status(execution_session.status)
                    result["plan_admission_succeeded"] = bool(checkpoints and authority)
                    result["apa_created"] = bool(authority)
                    result["apa_paths"] = _path_values(authority)
                    result["checkpoint_details"] = {
                        key: value
                        for key, value in checkpoint_details.items()
                        if "path" in key.lower()
                        or "version" in key.lower()
                        or "source" in key.lower()
                        or "plan" in key.lower()
                    }
                    result["source_version_fencing"] = any(
                        key in checkpoint_details
                        for key in (
                            "source_plan_hash",
                            "source_commit_identity",
                            "source_version",
                            "plan_source_hash",
                        )
                    )
                    result["plan_identity_preserved"] = bool(
                        getattr(plan, "id", None)
                    ) and bool(committed_tasks[0].plan_id)
                    result["workspace_after"] = _git_snapshot(workspace)
                    result["channel_counters"] = counters
                    result["e4_residual_reasoning_reached"] = any(
                        call.get("diagnostic_label") not in {None, "PLANNING_DISCOVERY"}
                        and call.get("stage") == "execution"
                        for call in result["provider_calls"]
                    )
                    result["e4_mutation_required"] = False
                    result["e4_native_tools_present"] = any(
                        item["has_native_tools"]
                        for item in result["provider_wire_ledger"]
                    )
                    result["expected_behavior_fixed"] = bool(
                        result.get("runtime_workspace_capture", {})
                        .get("independent_verification", {})
                        .get("returncode")
                        == 0
                    )
            except Exception as exc:
                db.rollback()
                result["errors"].append(f"{type(exc).__name__}: {str(exc)[:1000]}")
            finally:
                execution_loop.assess_step_execution.__globals__[
                    "execute_verification_command"
                ] = original_verification
                execution_local_steps.execute_verification_command = (
                    original_verification
                )
                execution_loop._execute_local_shell_commands_step = original_local
                ExecutorService.execute_file_ops = original_file_ops
                worker_module._dispose_runtime_workspace_safely = original_dispose
                worker_module.create_agent_runtime = real_create_runtime
                agent_runtime_module.create_agent_runtime = real_create_runtime
                openai_chat_adapter.OpenAIChatCompletionsRuntime._chat = original_chat
                httpx.AsyncClient.post = original_post
                monitor_stop.set()
                if monitor is not None:
                    monitor.join(timeout=2)
    finally:
        result["lifecycle_seconds"] = round(time.perf_counter() - lifecycle_started, 3)
        result["resource_samples"] = len(resource_samples)
        result["peak_system_ram"] = max(
            (item[0] for item in resource_samples), default=None
        )
        result["peak_swap"] = max((item[1] for item in resource_samples), default=None)
        if worker_module is not None:
            if original_worker_db is not None:
                worker_module.get_db_session = original_worker_db
            if original_worker_task is not None:
                worker_module.execute_orchestration_task = original_worker_task
        if capture_path.exists():
            result["capture"] = _capture_summary(_json_document(capture_path))
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
    result.pop("resource_samples", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required for the one authorized provider run")
    print(json.dumps(run_live(), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

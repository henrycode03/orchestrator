#!/usr/bin/env python3
"""Run Phase 18I Machine A Candidate Recovery limited validation.

Evidence-only harness. It creates controlled Orchestrator session/task rows,
executes the existing Recovery Registry Candidate Planning path with Machine A
standard runtime inputs, retains event journals, and restores feature-flag
settings after each controlled session.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import shutil
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from app.config import Settings, settings
from app.database import SessionLocal
from app.models import (
    PlanningArtifact,
    PlanningMessage,
    PlanningSession,
    Project,
    Session as SessionModel,
    SessionTask,
    Task,
    TaskStatus,
)
from app.services.orchestration.recovery.execution_recovery_evidence import (
    ExecutionRecoveryEvidence,
)
from app.services.orchestration.recovery.recovery_context import RecoveryContext
from app.services.orchestration.recovery.recovery_outcome import RecoveryOutcome
from app.services.orchestration.recovery.recovery_strategy_registry import (
    RecoveryStrategyRegistry,
)
from app.services.orchestration.validation.validator import ValidatorService
from app.services.planning.candidate_recovery import (
    CandidateRecoveryRequest,
    execute_single_sibling_candidate_recovery,
    planning_failure_signature,
)


WORKSPACE_ROOT = Path("/root/.openclaw/workspace/vault/projects")
WORKSPACE_SLUG_18I = "phase18i-machine-a-limited-validation"
WORKSPACE_SLUG_18J = "phase18j-machine-a-expanded-validation"
PROJECT_NAME_18I = "Phase 18I Machine A Limited Validation"
PROJECT_NAME_18J = "Phase 18J Machine A Expanded Validation"
EVIDENCE_ARCHIVE_18I = Path(
    "docs/roadmap/reports/evidence/phase18i-machine-a-limited-validation"
)
EVIDENCE_ARCHIVE_18J = Path(
    "docs/roadmap/reports/evidence/phase18j-machine-a-expanded-validation"
)
SUCCESS_STATUSES = {"accepted", "warning"}


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    source_variant: str
    sibling_variant: str
    chain_length: int = 1
    artifact_name: str = "phase18i"


@dataclass(frozen=True)
class SessionEvidence:
    name: str
    session_id: int
    task_id: int
    journal_path: Path
    validator_rule_ids: tuple[str, ...]
    recovery_trigger: str
    recovery_decision: str
    recovery_latency_ms: int
    rescue_success: bool
    selected_candidate: str
    selected_candidate_still_containing_rule: bool
    failure_signature: str
    machine_profile: str
    runtime_profile: str
    validator_statuses: dict[str, str]
    source_rules_by_candidate: dict[str, tuple[str, ...]]
    outcome_status: str
    rollback_verified: bool


SCENARIOS = (
    Scenario(
        slug="missing-verification-rescue-1",
        title="Missing verification rescued by valid sibling",
        source_variant="missing_verification",
        sibling_variant="valid",
    ),
    Scenario(
        slug="missing-verification-rescue-2",
        title="Repeated missing verification rescued by valid sibling",
        source_variant="missing_verification",
        sibling_variant="valid_alt",
    ),
    Scenario(
        slug="weak-verification-rescue",
        title="Weak verification rescued by valid sibling",
        source_variant="weak_verification",
        sibling_variant="valid",
    ),
    Scenario(
        slug="missing-verification-exhausted",
        title="Missing verification repeated by sibling",
        source_variant="missing_verification",
        sibling_variant="missing_verification",
    ),
    Scenario(
        slug="weak-verification-exhausted",
        title="Weak verification repeated by sibling",
        source_variant="weak_verification",
        sibling_variant="weak_verification",
    ),
    Scenario(
        slug="mixed-rule-exhausted",
        title="Missing verification followed by weak sibling",
        source_variant="missing_verification",
        sibling_variant="weak_verification",
    ),
)

SCENARIOS_18J = (
    Scenario(
        slug="api-module-missing-verification-rescue",
        title="API module missing verification rescued",
        source_variant="missing_verification",
        sibling_variant="valid",
        chain_length=2,
        artifact_name="phase18j_api",
    ),
    Scenario(
        slug="api-module-weak-verification-rescue",
        title="API module weak verification rescued",
        source_variant="weak_verification",
        sibling_variant="valid_alt",
        chain_length=3,
        artifact_name="phase18j_api",
    ),
    Scenario(
        slug="worker-module-missing-verification-exhausted",
        title="Worker module missing verification exhausted",
        source_variant="missing_verification",
        sibling_variant="missing_verification",
        chain_length=2,
        artifact_name="phase18j_worker",
    ),
    Scenario(
        slug="worker-module-weak-verification-exhausted",
        title="Worker module weak verification exhausted",
        source_variant="weak_verification",
        sibling_variant="weak_verification",
        chain_length=3,
        artifact_name="phase18j_worker",
    ),
    Scenario(
        slug="config-module-mixed-exhausted",
        title="Config module mixed-rule exhausted",
        source_variant="missing_verification",
        sibling_variant="weak_verification",
        chain_length=3,
        artifact_name="phase18j_config",
    ),
    Scenario(
        slug="cli-module-missing-verification-rescue",
        title="CLI module missing verification rescued",
        source_variant="missing_verification",
        sibling_variant="valid_alt",
        chain_length=4,
        artifact_name="phase18j_cli",
    ),
    Scenario(
        slug="schema-module-weak-verification-rescue",
        title="Schema module weak verification rescued",
        source_variant="weak_verification",
        sibling_variant="valid",
        chain_length=4,
        artifact_name="phase18j_schema",
    ),
    Scenario(
        slug="docs-helper-mixed-exhausted",
        title="Docs helper mixed-rule exhausted",
        source_variant="missing_verification",
        sibling_variant="weak_verification",
        chain_length=4,
        artifact_name="phase18j_docs",
    ),
    Scenario(
        slug="report-module-missing-verification-rescue",
        title="Report module missing verification rescued",
        source_variant="missing_verification",
        sibling_variant="valid",
        chain_length=5,
        artifact_name="phase18j_report",
    ),
    Scenario(
        slug="report-module-weak-verification-exhausted",
        title="Report module weak verification exhausted",
        source_variant="weak_verification",
        sibling_variant="weak_verification",
        chain_length=5,
        artifact_name="phase18j_report",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="orchestrator.db")
    parser.add_argument("--workspace-root", default=str(WORKSPACE_ROOT))
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--campaign", choices=("18i", "18j"), default="18i")
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Repeat the scenario matrix. Defaults: 1 for 18i, 5 for 18j.",
    )
    args = parser.parse_args()

    campaign = campaign_config(args.campaign)
    cycles = args.cycles if args.cycles is not None else campaign["default_cycles"]
    db_path = Path(args.db)
    workspace_root = Path(args.workspace_root)
    workspace = workspace_root / str(campaign["workspace_slug"])
    archive_dir = Path(args.archive_dir or str(campaign["archive_dir"]))
    prepare_directory(workspace / ".agent" / "events")
    prepare_directory(archive_dir)

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%d%H%M%S")
    before_migrations = migration_snapshot(db_path)
    repo_defaults = Settings(_env_file=None)
    evidence: list[SessionEvidence] = []

    db = SessionLocal()
    try:
        project = get_or_create_project(
            db,
            workspace,
            project_name=str(campaign["project_name"]),
            phase_label=str(campaign["phase_label"]),
            workspace_slug=str(campaign["workspace_slug"]),
        )
        for cycle_index, scenario in iter_campaign_scenarios(
            tuple(campaign["scenarios"]), cycles
        ):
            session, task = create_session_bundle(
                db,
                project,
                scenario=scenario,
                run_id=run_id,
                phase_label=str(campaign["phase_label"]),
                cycle_index=cycle_index,
            )
            db.commit()
            outcome = execute_controlled_session(
                workspace=workspace,
                session_id=session.id,
                task_id=task.id,
                scenario=scenario,
                phase_label=str(campaign["phase_label"]),
            )
            journal_path = (
                workspace
                / ".agent"
                / "events"
                / f"session_{session.id}_task_{task.id}.jsonl"
            )
            evidence.append(
                summarize_session(
                    scenario=scenario,
                    session_id=session.id,
                    task_id=task.id,
                    journal_path=journal_path,
                    outcome=outcome,
                    repo_defaults=repo_defaults,
                    cycle_index=cycle_index,
                )
            )
    finally:
        db.close()

    after_migrations = migration_snapshot(db_path)
    archived_files = archive_journals(
        [item.journal_path for item in evidence], archive_dir
    )
    write_json_summary(
        archive_dir / f"{str(campaign['phase_key'])}-summary-{run_id}.json",
        evidence=evidence,
        archived_files=archived_files,
        before_migrations=before_migrations,
        after_migrations=after_migrations,
        repo_defaults=repo_defaults,
        started_at=started_at,
        campaign_name=str(campaign["phase_label"]),
        cycles=cycles,
    )
    print(
        render_markdown(
            evidence=evidence,
            archived_files=archived_files,
            before_migrations=before_migrations,
            after_migrations=after_migrations,
            repo_defaults=repo_defaults,
            started_at=started_at,
            campaign_name=str(campaign["phase_label"]),
            cycles=cycles,
        )
    )


def campaign_config(name: str) -> dict[str, Any]:
    if name == "18j":
        return {
            "phase_label": "Phase 18J",
            "workspace_slug": WORKSPACE_SLUG_18J,
            "project_name": PROJECT_NAME_18J,
            "archive_dir": EVIDENCE_ARCHIVE_18J,
            "scenarios": SCENARIOS_18J,
            "default_cycles": 5,
            "phase_key": "phase18j",
        }
    return {
        "phase_label": "Phase 18I",
        "workspace_slug": WORKSPACE_SLUG_18I,
        "project_name": PROJECT_NAME_18I,
        "archive_dir": EVIDENCE_ARCHIVE_18I,
        "scenarios": SCENARIOS,
        "default_cycles": 1,
        "phase_key": "phase18i",
    }


def iter_campaign_scenarios(
    scenarios: tuple[Scenario, ...],
    cycles: int,
) -> Iterable[tuple[int, Scenario]]:
    for cycle_index in range(1, cycles + 1):
        for scenario in scenarios:
            yield cycle_index, scenario


def get_or_create_project(
    db,
    workspace: Path,
    *,
    project_name: str,
    phase_label: str,
    workspace_slug: str,
) -> Project:
    project = db.query(Project).filter(Project.name == project_name).first()
    if project is None:
        project = Project(
            name=project_name,
            description=f"{phase_label} Machine A Candidate Recovery validation project",
            workspace_path=workspace_slug,
        )
        db.add(project)
        db.flush()
    else:
        project.workspace_path = workspace_slug
    return project


def create_session_bundle(
    db,
    project: Project,
    *,
    scenario: Scenario,
    run_id: str,
    phase_label: str,
    cycle_index: int,
) -> tuple[SessionModel, Task]:
    task = Task(
        project_id=project.id,
        title=f"{phase_label} {scenario.title} cycle {cycle_index} {run_id}",
        description=(
            f"Controlled Machine A validation scenario: {scenario.title}; "
            f"cycle {cycle_index}"
        ),
        status=TaskStatus.DONE,
        execution_profile="full_lifecycle",
        task_subfolder=f"{phase_label.lower().replace(' ', '')}-{scenario.slug}-{cycle_index}-{run_id}",
    )
    db.add(task)
    db.flush()

    session = SessionModel(
        project_id=project.id,
        name=f"{phase_label} {scenario.title} cycle {cycle_index} {run_id}",
        description="Machine A standard-runtime Candidate Recovery validation",
        status="completed",
        execution_mode="automatic",
        default_execution_profile="full_lifecycle",
        is_active=False,
        instance_id=str(uuid.uuid4()),
    )
    db.add(session)
    db.flush()
    db.add(
        SessionTask(
            session_id=session.id,
            task_id=task.id,
            status=TaskStatus.DONE,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
    )
    planning_session = PlanningSession(
        project_id=project.id,
        title=f"Phase 18I Planning {scenario.title} {run_id}",
        prompt=f"Controlled validation for {scenario.title}.",
        status="completed",
        source_brain="local",
        completed_at=datetime.now(UTC),
    )
    db.add(planning_session)
    db.flush()
    db.add(
        PlanningMessage(
            planning_session_id=planning_session.id,
            role="assistant",
            content=(
                f"Evidence-only planning session for {scenario.title}, "
                f"cycle {cycle_index}."
            ),
            metadata_json={
                "phase": phase_label.replace("Phase ", ""),
                "session_id": session.id,
                "task_id": task.id,
                "machine_profile": "machine-a",
                "runtime_profile": "standard",
                "cycle": cycle_index,
            },
        )
    )
    db.add(
        PlanningArtifact(
            planning_session_id=planning_session.id,
            artifact_type="planner_markdown",
            filename="planner.md",
            content=(
                "## Task List\n"
                f"- [ ] TASK_START: {scenario.title} | Evidence only | "
                f"order=1 | P1 | effort=small | profile=full_lifecycle | "
                f"chain_length={scenario.chain_length}.\n"
            ),
            version=1,
            is_latest=True,
        )
    )
    db.flush()
    return session, task


def execute_controlled_session(
    *,
    workspace: Path,
    session_id: int,
    task_id: int,
    scenario: Scenario,
    phase_label: str,
) -> RecoveryOutcome:
    original_recovery_enabled = settings.CANDIDATE_RECOVERY_ENABLED
    original_slot_merge_enabled = settings.CANDIDATE_SLOT_MERGE_ENABLED
    original_runtime_profile = settings.RUNTIME_PROFILE
    try:
        settings.CANDIDATE_RECOVERY_ENABLED = True
        settings.CANDIDATE_SLOT_MERGE_ENABLED = False
        settings.RUNTIME_PROFILE = "standard"
        source_plan = plan_for(scenario.source_variant, scenario=scenario)
        source_verdict = validate_plan(workspace, source_plan)
        signature = planning_failure_signature(tuple(source_verdict.reasons or ()))
        state = SimpleNamespace()

        def execute_candidate():
            sibling_plan = plan_for(scenario.sibling_variant, scenario=scenario)
            return execute_single_sibling_candidate_recovery(
                CandidateRecoveryRequest(
                    project_dir=workspace,
                    session_id=session_id,
                    task_id=task_id,
                    original_plan=source_plan,
                    original_output_text=json.dumps(source_plan),
                    original_verdict=source_verdict,
                    runtime_profile="standard",
                    parent_event_id=None,
                    generate_sibling=lambda: (
                        sibling_plan,
                        json.dumps(sibling_plan),
                    ),
                    validate_candidate=lambda plan, _text: validate_plan(
                        workspace,
                        plan,
                    ),
                )
            )

        evidence = ExecutionRecoveryEvidence(
            task_title=f"{phase_label} {scenario.title}",
            task_description="Controlled Machine A planning validation failure",
            failed_command="planning_validation",
            exit_code=None,
            stdout_excerpt="",
            stderr_excerpt="planning validation failed",
            traceback_excerpt="",
            validator_rejection_reason=signature,
            failure_class="planning_validation_failed",
        )
        context = RecoveryContext(
            project_dir=workspace,
            session_id=session_id,
            task_id=task_id,
            scope="planning",
            evidence=evidence,
            orchestration_state=state,
            runtime_profile="standard",
            recovery_metadata={
                "planning_failure_signature": signature,
                "candidate_executor": execute_candidate,
                "candidate_operator": "",
            },
        )
        return RecoveryStrategyRegistry.execute_candidate_planning(context=context)
    finally:
        settings.CANDIDATE_RECOVERY_ENABLED = original_recovery_enabled
        settings.CANDIDATE_SLOT_MERGE_ENABLED = original_slot_merge_enabled
        settings.RUNTIME_PROFILE = original_runtime_profile


def summarize_session(
    *,
    scenario: Scenario,
    session_id: int,
    task_id: int,
    journal_path: Path,
    outcome: RecoveryOutcome,
    repo_defaults: Settings,
    cycle_index: int = 1,
) -> SessionEvidence:
    events = read_events(journal_path)
    validated = [
        event
        for event in events
        if event.get("event_type") == "plan_candidate_validated"
    ]
    rules_by_candidate: dict[str, tuple[str, ...]] = {}
    statuses: dict[str, str] = {}
    all_rules: list[str] = []
    for event in validated:
        details = details_of(event)
        candidate_id = str(details.get("candidate_id") or "")
        rules = tuple(str(rule) for rule in details.get("validator_rule_ids") or ())
        if candidate_id:
            rules_by_candidate[candidate_id] = rules
            statuses[candidate_id] = str(details.get("validator_status") or "")
        all_rules.extend(rules)

    selected_candidate = selected_candidate_id(events)
    selected_rules = set(rules_by_candidate.get(selected_candidate, ()))
    source_rules = set(rules_by_candidate.get("candidate-original", ()))
    decision_event = next(
        (
            event
            for event in events
            if event.get("event_type") == "recovery_decision_routed"
        ),
        {},
    )
    selected_status = statuses.get(selected_candidate, "")
    rescue_success = bool(outcome.succeeded and selected_status in SUCCESS_STATUSES)
    candidate_outcome = outcome.strategy_result.get("candidate_outcome", {})
    if not isinstance(candidate_outcome, dict):
        candidate_outcome = {}
    selected_candidate_details = candidate_outcome.get("selected_candidate") or {}
    if not isinstance(selected_candidate_details, dict):
        selected_candidate_details = {}
    return SessionEvidence(
        name=f"{scenario.slug}-cycle-{cycle_index}",
        session_id=session_id,
        task_id=task_id,
        journal_path=journal_path,
        validator_rule_ids=tuple(sorted(set(all_rules))),
        recovery_trigger="planning_validation_failed",
        recovery_decision=str(details_of(decision_event).get("strategy") or ""),
        recovery_latency_ms=int(outcome.duration_ms),
        rescue_success=rescue_success,
        selected_candidate=selected_candidate,
        selected_candidate_still_containing_rule=bool(source_rules & selected_rules),
        failure_signature=str(
            selected_candidate_details.get("planning_failure_signature")
            or details_of(decision_event).get("signature_hash")
            or ""
        ),
        machine_profile="machine-a",
        runtime_profile="standard",
        validator_statuses=statuses,
        source_rules_by_candidate=rules_by_candidate,
        outcome_status=str(outcome.strategy_result.get("status") or ""),
        rollback_verified=(
            settings.CANDIDATE_RECOVERY_ENABLED
            == repo_defaults.CANDIDATE_RECOVERY_ENABLED
            and settings.CANDIDATE_SLOT_MERGE_ENABLED
            == repo_defaults.CANDIDATE_SLOT_MERGE_ENABLED
        ),
    )


def render_markdown(
    *,
    evidence: list[SessionEvidence],
    archived_files: list[Path],
    before_migrations: tuple[str, ...],
    after_migrations: tuple[str, ...],
    repo_defaults: Settings,
    started_at: datetime,
    campaign_name: str = "Phase 18I",
    cycles: int = 1,
) -> str:
    sessions = len(evidence)
    triggered = [
        item for item in evidence if item.recovery_decision == "candidate_planning"
    ]
    rescued = [item for item in evidence if item.rescue_success]
    exhausted = [item for item in evidence if exhausted_outcome(item)]
    failures = [item for item in evidence if not item.rescue_success]
    false_rescues = [
        item
        for item in rescued
        if item.validator_statuses.get(item.selected_candidate) not in SUCCESS_STATUSES
    ]
    questionable = [
        item for item in rescued if item.selected_candidate_still_containing_rule
    ]
    latencies = [item.recovery_latency_ms for item in evidence]
    accepted = [
        item
        for item in evidence
        if item.validator_statuses.get(item.selected_candidate) in SUCCESS_STATUSES
    ]
    validator_distribution: Counter[str] = Counter()
    rule_frequency: Counter[str] = Counter()
    failure_signatures: Counter[str] = Counter()
    selected_candidates: Counter[str] = Counter()
    rescue_paths: Counter[str] = Counter()
    exhausted_paths: Counter[str] = Counter()
    failed_rescue_rules: Counter[str] = Counter()
    rollback_events = sum(1 for item in evidence if item.rollback_verified)
    for item in evidence:
        validator_distribution.update(item.validator_statuses.values())
        rule_frequency.update(item.validator_rule_ids)
        selected_candidates.update([item.selected_candidate or "none"])
        rule_key = ",".join(item.validator_rule_ids) or "none"
        if item.rescue_success:
            rescue_paths[rule_key] += 1
        if exhausted_outcome(item):
            exhausted_paths[rule_key] += 1
            failed_rescue_rules.update(item.validator_rule_ids)
        if not item.rescue_success:
            failure_signatures[item.failure_signature] += 1

    lines = [
        f"# {campaign_name} Machine A Candidate Recovery Validation Evidence",
        "",
        f"Started: {started_at.isoformat()}",
        f"Sessions executed: {sessions}",
        f"Scenario cycles: {cycles}",
        f"Recovery trigger rate: {rate(len(triggered), sessions):.3f}",
        f"Recovery trigger 95% CI: {format_ci(len(triggered), sessions)}",
        f"Rescue rate: {rate(len(rescued), len(triggered)):.3f}",
        f"Rescue rate 95% CI: {format_ci(len(rescued), len(triggered))}",
        f"Accepted selected-candidate rate: {rate(len(accepted), sessions):.3f}",
        f"Accepted selected-candidate 95% CI: {format_ci(len(accepted), sessions)}",
        f"Exhausted rate: {rate(len(exhausted), len(triggered)):.3f}",
        f"Exhausted rate 95% CI: {format_ci(len(exhausted), len(triggered))}",
        f"Success rate: {rate(len(rescued), sessions):.3f}",
        f"False rescue count: {len(false_rescues)}",
        f"Questionable rescue count: {len(questionable)}",
        f"Rollback events: {rollback_events}/{sessions}",
        "",
        "## Latency Distribution",
        f"- count: {len(latencies)}",
        f"- min_ms: {min(latencies) if latencies else 0}",
        f"- median_ms: {median_int(latencies) if latencies else 0}",
        f"- p95_ms: {percentile_int(latencies, 0.95) if latencies else 0}",
        f"- max_ms: {max(latencies) if latencies else 0}",
        "",
        "## Validator Distribution",
    ]
    append_counter(lines, validator_distribution)
    lines.extend(["", "## Rule Frequencies"])
    append_counter(lines, rule_frequency)
    lines.extend(["", "## Rescue Paths"])
    append_counter(lines, rescue_paths)
    lines.extend(["", "## Exhausted Paths"])
    append_counter(lines, exhausted_paths)
    lines.extend(["", "## Rules Repeatedly Failing Rescue"])
    append_counter(lines, failed_rescue_rules)
    lines.extend(["", "## Selected Candidate Persistence"])
    append_counter(lines, selected_candidates)
    lines.append(
        f"- selected_candidate_still_containing_source_rule: {len(questionable)}"
    )
    lines.extend(["", "## Repeated Failures"])
    repeated = Counter(
        {
            signature: count
            for signature, count in failure_signatures.items()
            if count > 1 and signature
        }
    )
    append_counter(lines, repeated)
    lines.extend(["", "## Session Records"])
    for item in evidence:
        lines.append(
            "- "
            f"{item.name}: session_id={item.session_id}, task_id={item.task_id}, "
            f"rules={list(item.validator_rule_ids)}, trigger={item.recovery_trigger}, "
            f"decision={item.recovery_decision}, latency_ms={item.recovery_latency_ms}, "
            f"rescue_success={item.rescue_success}, selected={item.selected_candidate or 'none'}, "
            "selected_candidate_still_containing_rule="
            f"{item.selected_candidate_still_containing_rule}, "
            f"failure_signature={item.failure_signature}, "
            f"machine_profile={item.machine_profile}, runtime_profile={item.runtime_profile}, "
            f"journal={item.journal_path}"
        )
    lines.extend(
        [
            "",
            "## Rollback And Drift Checks",
            f"- repository_default_CANDIDATE_RECOVERY_ENABLED: {repo_defaults.CANDIDATE_RECOVERY_ENABLED}",
            f"- repository_default_CANDIDATE_SLOT_MERGE_ENABLED: {repo_defaults.CANDIDATE_SLOT_MERGE_ENABLED}",
            f"- runtime_flags_restored_after_each_session: {rollback_events == sessions}",
            f"- database_schema_changes: {before_migrations != after_migrations}",
            "- migration_file_changes: checked by `git diff --check` and git diff review",
            "",
            "## Archived Evidence",
        ]
    )
    for path in archived_files:
        lines.append(f"- {path}")
    return "\n".join(lines)


def write_json_summary(
    path: Path,
    *,
    evidence: list[SessionEvidence],
    archived_files: list[Path],
    before_migrations: tuple[str, ...],
    after_migrations: tuple[str, ...],
    repo_defaults: Settings,
    started_at: datetime,
    campaign_name: str = "Phase 18I",
    cycles: int = 1,
) -> None:
    payload = {
        "campaign": campaign_name,
        "started_at": started_at.isoformat(),
        "cycles": cycles,
        "sessions": [
            item.__dict__ | {"journal_path": str(item.journal_path)}
            for item in evidence
        ],
        "archived_files": [str(path) for path in archived_files],
        "before_migrations": list(before_migrations),
        "after_migrations": list(after_migrations),
        "repository_defaults": {
            "CANDIDATE_RECOVERY_ENABLED": repo_defaults.CANDIDATE_RECOVERY_ENABLED,
            "CANDIDATE_SLOT_MERGE_ENABLED": repo_defaults.CANDIDATE_SLOT_MERGE_ENABLED,
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    chmod_shared(path)


def plan_for(variant: str, *, scenario: Scenario | None = None) -> list[dict[str, Any]]:
    scenario = scenario or Scenario(
        slug="phase18i-default",
        title="Phase 18I default",
        source_variant=variant,
        sibling_variant=variant,
    )
    module_name = scenario.artifact_name
    chain_length = max(1, int(scenario.chain_length))
    content = f"def {module_name}_value():\n" f"    return {18 + chain_length}\n"
    verification = f"python -m py_compile src/{module_name}.py"
    if variant == "missing_verification":
        verification = ""
    elif variant == "weak_verification":
        verification = f"test -f src/{module_name}.py"
    elif variant == "valid_alt":
        content = f"def {module_name}_value():\n" f"    return {19 + chain_length}\n"
    plan: list[dict[str, Any]] = []
    for step_number in range(1, chain_length + 1):
        path = f"src/{module_name}_{step_number}.py"
        step_content = (
            content
            if step_number == chain_length
            else f"def {module_name}_step_{step_number}():\n    return {step_number}\n"
        )
        step_verification = f"python -m py_compile {path}"
        if step_number == chain_length:
            step_verification = verification
        plan.append(
            {
                "step_number": step_number,
                "description": f"Implement controlled module step {step_number}",
                "commands": ["mkdir -p src"],
                "verification": step_verification,
                "rollback": f"rm -f {path}",
                "expected_files": [path],
                "ops": [
                    {
                        "op": "write_file",
                        "path": path,
                        "content": step_content,
                    }
                ],
            }
        )
    return plan


def validate_plan(workspace: Path, plan: list[dict[str, Any]]) -> Any:
    return ValidatorService.validate_plan(
        plan,
        output_text="",
        task_prompt="Implement a small Python module",
        execution_profile="full_lifecycle",
        project_dir=workspace,
    )


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def selected_candidate_id(events: Iterable[dict[str, Any]]) -> str:
    for event in reversed(list(events)):
        if event.get("event_type") != "plan_candidate_selected":
            continue
        details = details_of(event)
        return str(
            details.get("candidate_id") or details.get("selected_candidate_id") or ""
        )
    return ""


def details_of(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details") or {}
    return details if isinstance(details, dict) else {}


def migration_snapshot(db_path: Path) -> tuple[str, ...]:
    if not db_path.exists():
        return ()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select name from sqlite_master where type in ('table', 'index', 'trigger') order by name"
        ).fetchall()
    finally:
        con.close()
    return tuple(str(row[0]) for row in rows)


def archive_journals(journal_paths: Iterable[Path], archive_dir: Path) -> list[Path]:
    archived: list[Path] = []
    prepare_directory(archive_dir)
    for source in journal_paths:
        if not source.exists():
            continue
        target = archive_dir / source.name
        shutil.copy2(source, target)
        chmod_shared(target)
        archived.append(target)
    return archived


def prepare_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    chmod_shared(path)
    parent = path.parent
    workspace_slugs = {WORKSPACE_SLUG_18I, WORKSPACE_SLUG_18J}
    while parent != parent.parent and parent.name in {".agent", *workspace_slugs}:
        chmod_shared(parent)
        parent = parent.parent


def chmod_shared(path: Path) -> None:
    try:
        os.chmod(path, 0o777 if path.is_dir() else 0o666)
    except OSError:
        pass


def append_counter(lines: list[str], counter: Counter[str]) -> None:
    if not counter:
        lines.append("- none observed")
        return
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: {count}")


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def exhausted_outcome(item: SessionEvidence) -> bool:
    return (
        item.recovery_decision == "candidate_planning"
        and not item.rescue_success
        and not item.selected_candidate
        and item.outcome_status in {"exhausted", "failed"}
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denominator = 1 + z**2 / total
    centre = p_hat + z**2 / (2 * total)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
    return ((centre - margin) / denominator, (centre + margin) / denominator)


def format_ci(successes: int, total: int) -> str:
    low, high = wilson_interval(successes, total)
    return f"{low:.3f}-{high:.3f} (n={total})"


def median_int(values: list[int]) -> int:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return int((ordered[midpoint - 1] + ordered[midpoint]) / 2)


def percentile_int(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * percentile) - 1),
    )
    return ordered[index]


if __name__ == "__main__":
    main()

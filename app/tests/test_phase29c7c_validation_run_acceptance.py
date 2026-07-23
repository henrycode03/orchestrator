"""Focused Phase 29C-7C validation-run and acceptance boundary tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import pytest
from sqlalchemy import create_engine, inspect

from app.db_migrations import (
    MIGRATIONS,
    _migration_040_execution_task_validation_runs_acceptance,
    run_schema_migrations,
)
from app.models import (
    Base,
    ExecutionPlan,
    ExecutionTaskAcceptanceDecision,
    ExecutionTaskValidationRun,
    ExecutionTaskValidationSpecification,
    ExecutionTaskTransition,
)
from app.services.execution.candidate_evidence import (
    CandidateEvidenceError,
    CandidatePredicateResult,
    DeterministicValidatorRegistry,
)
from app.services.execution.validation_run import (
    FinalizeExecutionTaskValidationCommand,
    StartExecutionTaskValidationCommand,
    ValidationRunError,
    ValidationRunService,
    evaluate_pass_policy,
)
from app.services.execution.validation_contract import ValidationContractService
from app.services.planning.operator_review import canonical_json_hash
from app.services.planning.structured_task_plan import Task

from test_phase29c6b_runtime_evidence import _owned, _record_command, _start_command
from test_phase29c7b_evidence_validator import _contract as primitive_contract
from test_phase29c7b_evidence_validator import _structured_runtime


ENVIRONMENT_HASH = "a" * 64


def _plan_contract_hash(db_session, plan_id: int) -> str:
    tasks = (
        db_session.query(
            __import__("app.models", fromlist=["ExecutionTask"]).ExecutionTask
        )
        .filter_by(execution_plan_id=plan_id)
        .order_by(
            __import__(
                "app.models", fromlist=["ExecutionTask"]
            ).ExecutionTask.plan_task_id
        )
        .all()
    )
    specs = {
        item.execution_task_id: item
        for item in db_session.query(ExecutionTaskValidationSpecification)
        .filter_by(execution_plan_id=plan_id)
        .all()
    }
    return canonical_json_hash(
        [
            {
                "plan_task_id": task.plan_task_id,
                "contract_status": specs[task.id].contract_status,
                "specification_hash": specs[task.id].canonical_specification_hash,
            }
            for task in tasks
        ]
    )


def _rebind_contract(db_session, task, specification, contract):
    authored = Task(**task.task_spec)
    authored = replace(
        authored,
        work_items=tuple(
            replace(item, validation_contract=contract) for item in authored.work_items
        ),
    )
    projection = ValidationContractService.projection_for_task(authored)
    structured = projection.canonical_payload["structured_contract"]
    specification.contract_status = projection.contract_status
    specification.schema_version = projection.canonical_payload["schema_version"]
    specification.original_done_when = list(projection.original_done_when)
    specification.structured_contract = structured
    specification.pass_policy = structured["pass_policy"]
    specification.review_requirement = structured["review_requirement"]
    specification.environment_identity = structured["environment"]
    specification.validator_set_identity = structured["environment"]["validator_set_id"]
    specification.canonical_payload = projection.canonical_payload
    specification.canonical_specification_hash = projection.canonical_hash
    task.task_spec = authored.to_dict()
    task.done_when = [item.done_when for item in authored.work_items]
    task.validation_contract_status = "structured_executable"
    task.validation_contract_id = specification.id
    plan = db_session.get(ExecutionPlan, task.execution_plan_id)
    plan.validation_contract_set_hash = _plan_contract_hash(db_session, plan.id)
    db_session.flush()
    return contract


def _validation_command(task, outcome, specification, *, key="validation-start-1"):
    return StartExecutionTaskValidationCommand(
        execution_plan_id=task.execution_plan_id,
        execution_task_id=task.id,
        execution_task_attempt_id=outcome.execution_task_attempt_id,
        candidate_outcome_id=outcome.id,
        validation_specification_id=specification.id,
        validation_specification_hash=specification.canonical_specification_hash,
        expected_task_state="awaiting_validation",
        expected_task_state_version=task.state_version,
        validator_set_id="deterministic_readonly",
        validator_set_version="1",
        environment_configuration_hash=ENVIRONMENT_HASH,
        validation_idempotency_key=key,
    )


def _prepared_runtime(
    db_session, *, predicate_id="output_reference_exists", review="none"
):
    supported_fixture_predicate = predicate_id
    if predicate_id in {"json_schema_matches", "required_fields_present"}:
        supported_fixture_predicate = "output_reference_exists"
    task, _created, outcome, specification = _structured_runtime(
        db_session, predicate_id=supported_fixture_predicate
    )
    contract = primitive_contract(supported_fixture_predicate)
    if predicate_id != supported_fixture_predicate:
        predicate = contract.predicates[0]
        parameters = (
            {"schema_evidence_key": "primary_output"}
            if predicate_id == "json_schema_matches"
            else {"fields": ["result.id"]}
        )
        contract = replace(
            contract,
            predicates=(
                replace(predicate, predicate_id=predicate_id, parameters=parameters),
            ),
        )
    if review != "none":
        contract = replace(contract, review_requirement=review)
    _rebind_contract(db_session, task, specification, contract)
    db_session.commit()
    return task, outcome, specification


def test_start_is_committed_without_running_primitives_and_replays(db_session):
    task, outcome, specification = _prepared_runtime(db_session)
    service = ValidationRunService(db_session)
    command = _validation_command(task, outcome, specification)
    first = service.start_validation_run(command)
    assert first.run.run_status == "pending"
    assert db_session.query(ExecutionTaskValidationRun).count() == 1
    assert db_session.query(ExecutionTaskValidationRun).first().run_status == "pending"
    db_session.commit()
    replay = service.start_validation_run(command)
    assert replay.replayed is True
    assert replay.run.id == first.run.id
    assert replay.run.started_at == first.run.started_at


def test_start_rejects_legacy_and_not_required_contracts(db_session):
    task, outcome, specification = _prepared_runtime(db_session)
    service = ValidationRunService(db_session)
    task.validation_contract_status = "legacy_unstructured"
    with pytest.raises(ValidationRunError) as legacy:
        service.start_validation_run(_validation_command(task, outcome, specification))
    assert legacy.value.code == "validation_contract_unavailable"
    task.validation_contract_status = "validation_not_required"
    with pytest.raises(ValidationRunError) as not_required:
        service.start_validation_run(_validation_command(task, outcome, specification))
    assert not_required.value.code == "validation_not_required"
    assert db_session.query(ExecutionTaskValidationRun).count() == 0


def test_start_same_key_conflict_and_different_key_cannot_create_second_run(db_session):
    task, outcome, specification = _prepared_runtime(db_session)
    service = ValidationRunService(db_session)
    command = _validation_command(task, outcome, specification)
    service.start_validation_run(command)
    db_session.commit()
    with pytest.raises(ValidationRunError) as same_key:
        service.start_validation_run(
            replace(
                command,
                expected_task_state_version=command.expected_task_state_version + 1,
            )
        )
    assert same_key.value.code == "validation_run_idempotency_conflict"
    with pytest.raises(ValidationRunError) as different_key:
        service.start_validation_run(
            replace(command, validation_idempotency_key="validation-start-2")
        )
    assert different_key.value.code == "validation_run_already_exists"


def test_complete_metadata_contract_is_accepted_and_releases_dependency_only_once(
    db_session,
):
    task, outcome, specification = _prepared_runtime(db_session)
    service = ValidationRunService(db_session)
    command = _validation_command(task, outcome, specification)
    result = service.execute_validation_run(command)
    db_session.refresh(task)
    decision = db_session.query(ExecutionTaskAcceptanceDecision).one()
    assert result.run.run_status == "accepted"
    assert decision.decision_status == "accepted"
    assert task.status == "succeeded"
    assert task.state_version == decision.resulting_task_state_version
    assert decision.decision_reason == "validation_accepted"
    assert (
        db_session.query(ExecutionTaskTransition)
        .filter_by(to_state="succeeded")
        .count()
        == 1
    )
    replay = service.execute_validation_run(command)
    assert replay.run.id == result.run.id
    assert db_session.query(ExecutionTaskAcceptanceDecision).count() == 1
    assert (
        db_session.query(ExecutionTaskTransition)
        .filter_by(to_state="succeeded")
        .count()
        == 1
    )


class _FailingReferenceValidator:
    validator_id = "output_reference_exists"
    validator_version = 1

    def validate(self, predicate, evidence, context):
        return CandidatePredicateResult(
            "failed", False, "forced_candidate_failure", {}, {}, {}
        )


def test_authoritative_failure_is_rejected_without_terminal_failed_or_recovery(
    db_session,
):
    task, outcome, specification = _prepared_runtime(db_session)
    registry = DeterministicValidatorRegistry(configuration_hash=ENVIRONMENT_HASH)
    registry.register(
        predicate_id="output_reference_exists",
        predicate_version=1,
        validator_id="output_reference_exists",
        validator_version=1,
        validator=_FailingReferenceValidator(),
    )
    service = ValidationRunService(db_session)
    result = service.execute_validation_run(
        _validation_command(task, outcome, specification), registry=registry
    )
    db_session.refresh(task)
    decision = db_session.query(ExecutionTaskAcceptanceDecision).one()
    assert result.run.run_status == "rejected"
    assert decision.decision_status == "rejected"
    assert task.status == "awaiting_recovery"
    assert task.status != "failed"
    assert (
        not db_session.query(ExecutionTaskTransition)
        .filter_by(to_state="failed")
        .count()
    )


@pytest.mark.parametrize(
    "predicate_id,expected_reason",
    [
        ("json_schema_matches", "required_predicate_unsupported"),
        ("artifact_exists", "required_predicate_unsupported"),
        ("required_fields_present", "primitive_integrity_failure"),
    ],
)
def test_unsupported_or_content_contract_is_blocked_and_lifecycle_neutral(
    db_session, predicate_id, expected_reason
):
    task, outcome, specification = _prepared_runtime(
        db_session, predicate_id=predicate_id
    )
    service = ValidationRunService(db_session)
    result = service.execute_validation_run(
        _validation_command(task, outcome, specification)
    )
    db_session.refresh(task)
    decision = db_session.query(ExecutionTaskAcceptanceDecision).one()
    assert result.run.run_status == "blocked"
    assert decision.decision_status == "blocked"
    assert task.status == "awaiting_validation"
    assert task.state_version == result.run.task_state_version_at_start
    assert expected_reason in (result.run.bounded_detail or result.run.bounded_reason)
    assert (
        not db_session.query(ExecutionTaskTransition)
        .filter_by(to_state="succeeded")
        .count()
    )


def test_operator_review_requirement_is_review_required_not_auto_accepted(db_session):
    task, outcome, specification = _prepared_runtime(
        db_session, review="operator_required"
    )
    service = ValidationRunService(db_session)
    result = service.execute_validation_run(
        _validation_command(task, outcome, specification)
    )
    db_session.refresh(task)
    decision = db_session.query(ExecutionTaskAcceptanceDecision).one()
    assert result.run.run_status == "review_required"
    assert decision.decision_status == "review_required"
    assert decision.review_result["reason"] == "review_authority_missing"
    assert task.status == "awaiting_validation"


class _ExplodingResolver:
    def resolve(self, command):
        raise CandidateEvidenceError(
            "validation_resolver_failed", "bounded resolver failure"
        )


def test_unexpected_resolver_failure_is_validation_error_not_rejection(db_session):
    task, outcome, specification = _prepared_runtime(db_session)
    service = ValidationRunService(db_session, resolver=_ExplodingResolver())
    result = service.execute_validation_run(
        _validation_command(task, outcome, specification)
    )
    db_session.refresh(task)
    decision = db_session.query(ExecutionTaskAcceptanceDecision).one()
    assert result.run.run_status == "validation_error"
    assert decision.decision_status == "validation_error"
    assert task.status == "awaiting_validation"
    assert (
        not db_session.query(ExecutionTaskTransition)
        .filter_by(to_state="succeeded")
        .count()
    )


def test_pass_policy_distinguishes_failure_from_unavailable_authority():
    contract = primitive_contract("output_reference_exists")
    predicate = contract.predicates[0]
    failed = SimpleNamespace(result_status="failed", passed=False)
    unsupported = SimpleNamespace(result_status="unsupported", passed=False)
    passed = SimpleNamespace(result_status="passed", passed=True)
    assert (
        evaluate_pass_policy(contract, {(predicate.predicate_id, 1): failed}).status
        == "failed"
    )
    assert (
        evaluate_pass_policy(
            contract, {(predicate.predicate_id, 1): unsupported}
        ).status
        == "blocked"
    )
    assert (
        evaluate_pass_policy(contract, {(predicate.predicate_id, 1): passed}).status
        == "passed"
    )


def test_integrity_detects_aggregate_tampering_and_inspection_is_read_only(db_session):
    task, outcome, specification = _prepared_runtime(db_session)
    service = ValidationRunService(db_session)
    result = service.execute_validation_run(
        _validation_command(task, outcome, specification)
    )
    run = db_session.get(ExecutionTaskValidationRun, result.run.id)
    run.aggregate_evidence_hash = "f" * 64
    db_session.commit()
    integrity = service.verify_validation_run_integrity(run.id)
    assert not integrity.verified
    assert "validation_run_aggregate_evidence_hash_mismatch" in integrity.issues
    projection = service.inspect_execution_task_validation(task.id)
    assert projection.validation_state == "decision_lifecycle_mismatch"
    assert db_session.query(ExecutionTaskValidationRun).count() == 1


def test_migration_040_is_replay_safe_and_fresh_schema_compatible(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'validation-acceptance.db'}")
    Base.metadata.create_all(engine)
    _migration_040_execution_task_validation_runs_acceptance(engine)
    _migration_040_execution_task_validation_runs_acceptance(engine)
    inspector = inspect(engine)
    assert inspector.has_table("execution_task_validation_runs")
    assert inspector.has_table("execution_task_acceptance_decisions")
    assert "uq_execution_task_validation_run_candidate_spec_generation" in {
        item["name"]
        for item in inspector.get_unique_constraints("execution_task_validation_runs")
    }
    assert "uq_execution_task_acceptance_candidate_spec" in {
        item["name"]
        for item in inspector.get_unique_constraints(
            "execution_task_acceptance_decisions"
        )
    }
    engine.dispose()


def test_migration_040_replays_from_the_phase_29c7b_schema_without_fabrication(
    tmp_path,
):
    engine = create_engine(f"sqlite:///{tmp_path / 'phase29c7b-to-c7c.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE execution_task_acceptance_decisions")
        connection.exec_driver_sql("DROP TABLE execution_task_validation_runs")
    run_schema_migrations(engine, MIGRATIONS[:-1])
    run_schema_migrations(engine)
    run_schema_migrations(engine)
    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT COUNT(*) FROM execution_task_validation_runs"
            ).scalar_one()
            == 0
        )
        assert (
            connection.exec_driver_sql(
                "SELECT COUNT(*) FROM execution_task_acceptance_decisions"
            ).scalar_one()
            == 0
        )
        assert (
            connection.exec_driver_sql(
                "SELECT COUNT(*) FROM execution_task_resolved_validation_evidence"
            ).scalar_one()
            == 0
        )
    engine.dispose()

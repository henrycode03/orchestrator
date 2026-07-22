"""Focused tests for Phase 29B-3 Execution Release Contract Hardening."""

from __future__ import annotations

from dataclasses import replace as _replace

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_current_active_user, get_current_user
from app.models import (
    ExecutionCommitCommand,
    ExecutionDependencyEdge,
    ExecutionGroup,
    ExecutionGroupMember,
    ExecutionPlan,
    ExecutionTask,
    PlanningCommitManifest,
    PlanningCompletionManifest,
    Project,
    User,
)
from app.services.execution.execution_plan_commit_service import (
    ExecutionPlanCommitError,
    ExecutionPlanCommitService,
)
from app.services.planning.execution_commit import (
    ExecutionCommitError,
    ExecutionCommitRequest,
    PlanningExecutionCommitService,
)
from app.services.planning.protocol_persistence import (
    PlanningProtocolPersistenceService,
)
from app.services.planning.structured_task_plan import Dependency, StructuredTaskPlan

from app.tests.test_phase29b1_execution_plan_commit_service import (
    _build_accepted_commit_authority,
)
from app.tests.test_phase29b2_planning_execution_commit import (
    _build_approved_session,
    _request_for,
)


# ---------------------------------------------------------------------------
# 1. Release identity: future multi-generation compatibility
# ---------------------------------------------------------------------------


def test_different_accepted_checkpoint_is_a_separate_historical_release(db_session):
    """A commit manifest bound to a *different* accepted checkpoint must not
    block release of the current checkpoint's authority."""

    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    unrelated = PlanningCommitManifest(
        planning_session_id=session.id,
        protocol_version="v2",
        session_generation_id=session.generation_id,
        commit_identity="unrelated-commit-identity",
        task_provenance={
            "schema": "planning_execution_commit.v1",
            "structured_task_plan_checkpoint_id": promotion_checkpoint.id + 999,
            "structured_task_plan_hash": "9" * 64,
        },
    )
    db_session.add(unrelated)
    db_session.commit()

    service = PlanningExecutionCommitService(db_session)
    result = service.commit(session.id, _request_for(promotion_checkpoint, session))
    assert result.boundary_state == "released"

    manifests = (
        db_session.query(PlanningCommitManifest)
        .filter(PlanningCommitManifest.planning_session_id == session.id)
        .all()
    )
    assert len(manifests) == 2


def test_unrelated_session_manifest_does_not_interfere(db_session):
    project_a, session_a, plan_a, review_a, approval_a, checkpoint_a = (
        _build_approved_session(db_session, idempotency_key="a")
    )
    project_b, session_b, plan_b, review_b, approval_b, checkpoint_b = (
        _build_approved_session(db_session, idempotency_key="b")
    )
    service = PlanningExecutionCommitService(db_session)
    result_a = service.commit(
        session_a.id,
        _request_for(checkpoint_a, session_a, idempotency_key="commit-a"),
    )
    result_b = service.commit(
        session_b.id,
        _request_for(checkpoint_b, session_b, idempotency_key="commit-b"),
    )
    assert result_a.boundary_state == "released"
    assert result_b.boundary_state == "released"
    assert result_a.planning_commit_manifest_id != result_b.planning_commit_manifest_id


# ---------------------------------------------------------------------------
# 2. Idempotency-key command binding
# ---------------------------------------------------------------------------


def test_same_key_different_authority_is_idempotency_conflict(db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    service = PlanningExecutionCommitService(db_session)
    request = _request_for(promotion_checkpoint, session, idempotency_key="shared-key")
    service.commit(session.id, request)

    conflicting = ExecutionCommitRequest(
        idempotency_key="shared-key",
        operator_subject=request.operator_subject,
        structured_task_plan_checkpoint_id=promotion_checkpoint.id,
        structured_task_plan_hash=promotion_checkpoint.content_hash,
        expected_session_generation_id=session.generation_id,
        expected_review_id="a-different-review-id",
    )
    with pytest.raises(ExecutionCommitError) as exc_info:
        service.commit(session.id, conflicting)
    assert exc_info.value.code == "idempotency_key_conflict"


def test_command_binding_persists_and_advances_on_retry(db_session, monkeypatch):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    from app.services.execution import execution_plan_commit_service as ecs_module

    original_map = dict(ecs_module.DEPENDENCY_RUNTIME_CLASS_MAP)
    monkeypatch.setattr(ecs_module, "DEPENDENCY_RUNTIME_CLASS_MAP", {})

    service = PlanningExecutionCommitService(db_session)
    request = _request_for(promotion_checkpoint, session, idempotency_key="retry-key")
    first = service.commit(session.id, request)
    assert first.boundary_state == "released_execution_pending"

    command = (
        db_session.query(ExecutionCommitCommand)
        .filter(ExecutionCommitCommand.idempotency_key == "retry-key")
        .one()
    )
    assert command.boundary_state == "released_execution_pending"
    assert command.execution_plan_id is None
    assert command.planning_commit_manifest_id == first.planning_commit_manifest_id

    monkeypatch.setattr(ecs_module, "DEPENDENCY_RUNTIME_CLASS_MAP", original_map)
    second = service.commit(session.id, request)
    assert second.boundary_state == "released"

    db_session.refresh(command)
    assert command.boundary_state == "released"
    assert command.execution_plan_id == second.execution_plan_id


def test_different_key_same_authority_replays_via_manifest_identity(db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    service = PlanningExecutionCommitService(db_session)
    first = service.commit(
        session.id, _request_for(promotion_checkpoint, session, idempotency_key="k1")
    )
    second = service.commit(
        session.id, _request_for(promotion_checkpoint, session, idempotency_key="k2")
    )
    assert second.idempotent_replay is True
    assert second.planning_commit_manifest_id == first.planning_commit_manifest_id
    assert second.execution_plan_id == first.execution_plan_id
    assert (
        db_session.query(ExecutionCommitCommand).count() == 2
    )  # each key gets its own command binding


# ---------------------------------------------------------------------------
# 4. Partial-success response fields
# ---------------------------------------------------------------------------


def test_partial_success_never_leaks_raw_exception_text(db_session, monkeypatch):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    from app.services.execution import execution_plan_commit_service as ecs_module

    monkeypatch.setattr(ecs_module, "DEPENDENCY_RUNTIME_CLASS_MAP", {})
    service = PlanningExecutionCommitService(db_session)
    result = service.commit(session.id, _request_for(promotion_checkpoint, session))

    assert result.boundary_state == "released_execution_pending"
    assert result.integrity_status == "execution_materialization_pending"
    assert result.execution_error_code == "execution_materialization_failed"
    assert result.retryable is True
    assert result.execution_plan_id is None

    from app.schemas.planning_execution_commit import ExecutionCommitResponse

    payload = {
        "planning_session_id": result.planning_session_id,
        "session_generation_id": result.session_generation_id,
        "structured_task_plan_checkpoint_id": result.structured_task_plan_checkpoint_id,
        "structured_task_plan_hash": result.structured_task_plan_hash,
        "review_id": result.review_id,
        "approval_event_id": result.approval_event_id,
        "completion_manifest_id": result.completion_manifest_id,
        "completion_manifest_hash": result.completion_manifest_hash,
        "planning_commit_manifest_id": result.planning_commit_manifest_id,
        "commit_identity": result.commit_identity,
        "boundary_state": result.boundary_state,
        "idempotent_replay": result.idempotent_replay,
        "integrity_status": result.integrity_status,
        "execution_plan_id": result.execution_plan_id,
        "execution_plan_generation": result.execution_plan_generation,
        "execution_plan_status": result.execution_plan_status,
        "task_count": result.task_count,
        "dependency_edge_count": result.dependency_edge_count,
        "group_count": result.group_count,
        "group_membership_count": result.group_membership_count,
        "retryable": result.retryable,
        "execution_error_code": result.execution_error_code,
    }
    response = ExecutionCommitResponse(**payload)
    dumped = response.model_dump_json()
    assert "unknown Structured Task Plan dependency type" not in dumped
    assert "execution_failure_reason" not in dumped


# ---------------------------------------------------------------------------
# 5. Processing-lease correctness
# ---------------------------------------------------------------------------


def test_already_held_lease_returns_completion_manifest_pending(db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    session.processing_token = "someone-elses-token"
    db_session.commit()

    service = PlanningExecutionCommitService(db_session)
    with pytest.raises(ExecutionCommitError) as exc_info:
        service.commit(session.id, _request_for(promotion_checkpoint, session))
    assert exc_info.value.code == "completion_manifest_pending"

    db_session.refresh(session)
    assert session.processing_token == "someone-elses-token"


def test_rollback_after_commit_manifest_conflict_leaves_no_stale_token(db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    tampered = PlanningCommitManifest(
        planning_session_id=session.id,
        protocol_version="v2",
        session_generation_id=session.generation_id,
        commit_identity="tampered-identity",
        task_provenance={
            "schema": "planning_execution_commit.v1",
            "structured_task_plan_checkpoint_id": promotion_checkpoint.id,
            "structured_task_plan_hash": "1" * 64,
        },
    )
    db_session.add(tampered)
    db_session.commit()

    service = PlanningExecutionCommitService(db_session)
    with pytest.raises(ExecutionCommitError) as exc_info:
        service.commit(session.id, _request_for(promotion_checkpoint, session))
    assert exc_info.value.code == "commit_manifest_conflict"

    db_session.refresh(session)
    assert session.processing_token is None
    assert session.processing_started_at is None


def test_successful_commit_clears_processing_token(db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    service = PlanningExecutionCommitService(db_session)
    service.commit(session.id, _request_for(promotion_checkpoint, session))
    db_session.refresh(session)
    assert session.processing_token is None
    assert session.processing_started_at is None


def test_replay_does_not_reacquire_lease_when_completion_manifest_exists(db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    service = PlanningExecutionCommitService(db_session)
    service.commit(
        session.id,
        _request_for(promotion_checkpoint, session, idempotency_key="first"),
    )
    db_session.refresh(session)
    assert session.processing_token is None

    session.processing_token = "held-by-someone-else"
    db_session.commit()
    replay = service.commit(
        session.id,
        _request_for(promotion_checkpoint, session, idempotency_key="second"),
    )
    assert replay.idempotent_replay is True
    db_session.refresh(session)
    assert session.processing_token == "held-by-someone-else"


# ---------------------------------------------------------------------------
# 6. Graph defensive tests
# ---------------------------------------------------------------------------


def _tampered_plan(plan: StructuredTaskPlan, **overrides) -> StructuredTaskPlan:
    return _replace(plan, **overrides)


def _patch_accepted_plan(
    monkeypatch, session, commit_manifest, checkpoint, tampered_plan
):
    """Bypass ``_resolve_authority``'s own consistency checks (session
    generation, completion-manifest binding, checkpoint hash) so a
    deliberately-invalid *graph* (self edge, duplicate edge, dangling
    reference) reaches the graph-materialization code under test without
    weakening any public Structured Task Plan construction/validation."""

    def _fake_resolve(self, planning_commit_manifest_id):
        return session, commit_manifest, tampered_plan, checkpoint

    monkeypatch.setattr(ExecutionPlanCommitService, "_resolve_authority", _fake_resolve)


def test_self_dependency_edge_fails_closed(db_session, monkeypatch):
    _, session, plan, checkpoint, _, commit_manifest = _build_accepted_commit_authority(
        db_session
    )
    task_id = plan.tasks[0].id
    tampered = _tampered_plan(
        plan,
        dependencies=(
            Dependency(
                "self-dep",
                task_id,
                task_id,
                "hard_completion",
                "self edge",
            ),
        ),
    )
    _patch_accepted_plan(monkeypatch, session, commit_manifest, checkpoint, tampered)
    service = ExecutionPlanCommitService(db_session)
    with pytest.raises(ExecutionPlanCommitError, match="self-edge"):
        service.commit(commit_manifest.id)
    db_session.rollback()
    assert db_session.query(ExecutionPlan).count() == 0


def test_duplicate_dependency_edge_fails_closed(db_session, monkeypatch):
    _, session, plan, checkpoint, _, commit_manifest = _build_accepted_commit_authority(
        db_session
    )
    prerequisite_id, dependent_id = plan.tasks[0].id, plan.tasks[1].id
    tampered = _tampered_plan(
        plan,
        dependencies=(
            Dependency(
                "dup-1",
                prerequisite_id,
                dependent_id,
                "hard_completion",
                "first",
            ),
            Dependency(
                "dup-2",
                prerequisite_id,
                dependent_id,
                "hard_completion",
                "second",
            ),
        ),
    )
    _patch_accepted_plan(monkeypatch, session, commit_manifest, checkpoint, tampered)
    service = ExecutionPlanCommitService(db_session)
    with pytest.raises(ExecutionPlanCommitError, match="duplicate dependency edge"):
        service.commit(commit_manifest.id)
    db_session.rollback()
    assert db_session.query(ExecutionPlan).count() == 0


def test_unresolved_dependency_endpoint_fails_closed(db_session, monkeypatch):
    _, session, plan, checkpoint, _, commit_manifest = _build_accepted_commit_authority(
        db_session
    )
    tampered = _tampered_plan(
        plan,
        dependencies=(
            Dependency(
                "dangling",
                plan.tasks[0].id,
                "no-such-task",
                "hard_completion",
                "dangling endpoint",
            ),
        ),
    )
    _patch_accepted_plan(monkeypatch, session, commit_manifest, checkpoint, tampered)
    service = ExecutionPlanCommitService(db_session)
    with pytest.raises(ExecutionPlanCommitError, match="does not resolve"):
        service.commit(commit_manifest.id)
    db_session.rollback()
    assert db_session.query(ExecutionPlan).count() == 0


def test_unresolved_group_member_fails_closed(db_session, monkeypatch):
    from app.services.planning.structured_task_plan import (
        ExecutionGroup as PlanExecutionGroup,
    )

    _, session, plan, checkpoint, _, commit_manifest = _build_accepted_commit_authority(
        db_session
    )
    tampered = _tampered_plan(
        plan,
        execution_groups=(
            PlanExecutionGroup(
                id="orphan-group",
                kind="sequential",
                order=1,
                task_ids=("no-such-task",),
                parallel_limit=1,
                skip_policy="not_skippable",
            ),
        ),
    )
    _patch_accepted_plan(monkeypatch, session, commit_manifest, checkpoint, tampered)
    service = ExecutionPlanCommitService(db_session)
    with pytest.raises(ExecutionPlanCommitError, match="does not resolve"):
        service.commit(commit_manifest.id)
    db_session.rollback()
    assert db_session.query(ExecutionPlan).count() == 0


def test_commit_failure_rolls_back_all_five_graph_tables(db_session, monkeypatch):
    _, session, plan, checkpoint, _, commit_manifest = _build_accepted_commit_authority(
        db_session
    )
    from app.services.planning.structured_task_plan import (
        ExecutionGroup as PlanExecutionGroup,
    )

    # A valid graph up through groups, but a dangling group member forces a
    # failure only after tasks/edges/groups would already have been staged.
    tampered = _tampered_plan(
        plan,
        execution_groups=(
            PlanExecutionGroup(
                id="orphan-group",
                kind="sequential",
                order=1,
                task_ids=("no-such-task",),
                parallel_limit=1,
                skip_policy="not_skippable",
            ),
        ),
    )
    _patch_accepted_plan(monkeypatch, session, commit_manifest, checkpoint, tampered)
    service = ExecutionPlanCommitService(db_session)
    with pytest.raises(ExecutionPlanCommitError):
        service.commit(commit_manifest.id)
    db_session.rollback()

    assert db_session.query(ExecutionPlan).count() == 0
    assert db_session.query(ExecutionTask).count() == 0
    assert db_session.query(ExecutionDependencyEdge).count() == 0
    assert db_session.query(ExecutionGroup).count() == 0
    assert db_session.query(ExecutionGroupMember).count() == 0


# ---------------------------------------------------------------------------
# 7. Retention and deletion
# ---------------------------------------------------------------------------


def test_deleting_execution_plan_cascades_only_to_graph_children(db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    service = PlanningExecutionCommitService(db_session)
    result = service.commit(session.id, _request_for(promotion_checkpoint, session))
    execution_plan = db_session.get(ExecutionPlan, result.execution_plan_id)
    manifest_id = result.planning_commit_manifest_id
    completion_manifest_id = result.completion_manifest_id

    db_session.delete(execution_plan)
    db_session.commit()

    assert db_session.query(ExecutionPlan).count() == 0
    assert db_session.query(ExecutionTask).count() == 0
    assert db_session.query(ExecutionDependencyEdge).count() == 0
    assert db_session.query(ExecutionGroup).count() == 0
    assert db_session.query(ExecutionGroupMember).count() == 0

    assert db_session.get(PlanningCommitManifest, manifest_id) is not None
    assert (
        db_session.get(PlanningCompletionManifest, completion_manifest_id) is not None
    )


def test_soft_deleting_project_does_not_delete_release_authority(db_session):
    from datetime import datetime, timezone

    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    service = PlanningExecutionCommitService(db_session)
    result = service.commit(session.id, _request_for(promotion_checkpoint, session))

    project.deleted_at = datetime.now(timezone.utc)
    db_session.commit()

    assert db_session.get(ExecutionPlan, result.execution_plan_id) is not None
    assert (
        db_session.get(PlanningCommitManifest, result.planning_commit_manifest_id)
        is not None
    )


# ---------------------------------------------------------------------------
# 3. Public API contract
# ---------------------------------------------------------------------------


def _client_as(api_app, *, user_id: int, email: str) -> TestClient:
    user = User(id=user_id, email=email, hashed_password="not-used", is_active=True)
    api_app.dependency_overrides[get_current_user] = lambda: user
    api_app.dependency_overrides[get_current_active_user] = lambda: user
    return TestClient(api_app)


def _payload_for(promotion_checkpoint, session, *, idempotency_key="api-commit-1"):
    return {
        "idempotency_key": idempotency_key,
        "structured_task_plan_checkpoint_id": promotion_checkpoint.id,
        "structured_task_plan_hash": promotion_checkpoint.content_hash,
        "expected_session_generation_id": session.generation_id,
    }


def test_unauthenticated_execution_commit_is_rejected(api_client):
    response = api_client.post(
        "/api/v1/planning/sessions/1/execution-commit",
        json={
            "idempotency_key": "x",
            "structured_task_plan_checkpoint_id": 1,
            "structured_task_plan_hash": "0" * 64,
            "expected_session_generation_id": "g",
        },
    )
    assert response.status_code == 401


def test_project_owner_succeeds_and_response_schema_is_correct(api_app, db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    db_session.commit()

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    response = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit",
        json=_payload_for(promotion_checkpoint, session),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["boundary_state"] == "released"
    assert body["execution_plan_id"] is not None
    assert body["idempotent_replay"] is False
    assert "execution_failure_reason" not in body


def test_unrelated_user_is_forbidden(api_app, db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    db_session.commit()

    client = _client_as(api_app, user_id=2, email="someone-else@example.test")
    response = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit",
        json=_payload_for(promotion_checkpoint, session),
    )
    assert response.status_code == 403


def test_administrator_succeeds_for_unowned_project(api_app, db_session, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_EMAILS", "admin@example.test")
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    db_session.commit()

    client = _client_as(api_app, user_id=99, email="admin@example.test")
    response = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit",
        json=_payload_for(promotion_checkpoint, session),
    )
    assert response.status_code == 200, response.text


def test_missing_session_is_404(api_app, db_session):
    client = _client_as(api_app, user_id=1, email="owner@example.test")
    response = client.post(
        "/api/v1/planning/sessions/999999/execution-commit",
        json={
            "idempotency_key": "x",
            "structured_task_plan_checkpoint_id": 1,
            "structured_task_plan_hash": "0" * 64,
            "expected_session_generation_id": "g",
        },
    )
    assert response.status_code == 404


def test_soft_deleted_project_is_404(api_app, db_session):
    from datetime import datetime, timezone

    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    project.deleted_at = datetime.now(timezone.utc)
    db_session.commit()

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    response = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit",
        json=_payload_for(promotion_checkpoint, session),
    )
    assert response.status_code == 404


def test_protocol_v1_session_is_422(api_app, db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    session.protocol_version = "v1"
    db_session.commit()

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    response = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit",
        json=_payload_for(promotion_checkpoint, session),
    )
    assert response.status_code == 404  # v1 sessions are filtered out of context lookup


def test_wrong_generation_is_412(api_app, db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    db_session.commit()

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    payload = _payload_for(promotion_checkpoint, session)
    payload["expected_session_generation_id"] = "wrong-generation"
    response = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit", json=payload
    )
    assert response.status_code == 412
    assert response.json()["detail"]["code"] == "authority_stale"


def test_pending_review_is_422_via_api(api_app, db_session):
    from app.tests.test_phase29b2_planning_execution_commit import (
        _build_pending_review_session,
    )

    project, session, plan, brief_checkpoint, candidate, review = (
        _build_pending_review_session(db_session)
    )
    project.user_id = 1
    db_session.commit()

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    response = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit",
        json=_payload_for(candidate, session),
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "task_plan_not_approved"


def test_idempotent_replay_via_api_returns_same_execution_plan(api_app, db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    db_session.commit()

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    payload = _payload_for(promotion_checkpoint, session)
    first = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit", json=payload
    )
    assert first.status_code == 200
    second = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit", json=payload
    )
    assert second.status_code == 200
    assert second.json()["idempotent_replay"] is True
    assert second.json()["execution_plan_id"] == first.json()["execution_plan_id"]


def test_idempotency_key_conflict_is_409_via_api(api_app, db_session):
    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    db_session.commit()

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    payload = _payload_for(promotion_checkpoint, session)
    first = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit", json=payload
    )
    assert first.status_code == 200

    conflicting = dict(payload)
    conflicting["expected_review_id"] = "not-the-real-review-id"
    second = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit", json=conflicting
    )
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "idempotency_key_conflict"


def test_partial_success_response_is_202_and_redacted(api_app, db_session, monkeypatch):
    from app.services.execution import execution_plan_commit_service as ecs_module

    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    db_session.commit()
    monkeypatch.setattr(ecs_module, "DEPENDENCY_RUNTIME_CLASS_MAP", {})

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    response = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit",
        json=_payload_for(promotion_checkpoint, session),
    )
    assert response.status_code == 202
    body = response.json()
    assert body["boundary_state"] == "released_execution_pending"
    assert body["integrity_status"] == "execution_materialization_pending"
    assert body["execution_error_code"] == "execution_materialization_failed"
    assert body["retryable"] is True
    assert body["execution_plan_id"] is None
    assert "execution_failure_reason" not in body
    assert "unknown Structured Task Plan dependency type" not in response.text


def test_rate_limiting_is_invoked_for_execution_commit(
    api_app, db_session, monkeypatch
):
    from app import config as cfg
    from app.services.auth.rate_limit import clear_auth_rate_limits

    clear_auth_rate_limits()
    monkeypatch.setattr(cfg.settings, "API_RATE_LIMIT_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(cfg.settings, "API_RATE_LIMIT_WINDOW_SECONDS", 60)

    project, session, plan, review_id, approval, promotion_checkpoint = (
        _build_approved_session(db_session)
    )
    project.user_id = 1
    db_session.commit()

    client = _client_as(api_app, user_id=1, email="owner@example.test")
    payload = _payload_for(promotion_checkpoint, session)
    first = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit", json=payload
    )
    assert first.status_code == 200

    second_payload = dict(payload)
    second_payload["idempotency_key"] = "api-commit-2"
    second = client.post(
        f"/api/v1/planning/sessions/{session.id}/execution-commit",
        json=second_payload,
    )
    assert second.status_code == 429
    clear_auth_rate_limits()

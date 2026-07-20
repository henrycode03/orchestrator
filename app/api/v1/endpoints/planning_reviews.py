"""Authenticated Protocol v2 operator-review API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_active_user
from app.models import PlanningCheckpoint, PlanningSession, Project, User
from app.schemas.planning_review import (
    AcknowledgeReviewRequest,
    AmendReviewRequest,
    ApproveReviewRequest,
    CancelReviewRequest,
    RegenerateReviewRequest,
    RejectReviewRequest,
    ReviewActionResponse,
    ReviewCandidateBindingRequest,
    ReviewDetailResponse,
    ReviewEventResponse,
    ReviewListResponse,
    ReviewSummaryResponse,
)
from app.services.auth.authorization import is_admin_user
from app.services.auth.rate_limit import enforce_api_rate_limit
from app.services.planning.operator_review import (
    ReviewConflict,
    ReviewDecisionRequest,
    ReviewDomainError,
    ReviewIntegrityError,
    ReviewOperationError,
    canonical_json_bytes,
    canonical_json_hash,
)
from app.services.planning.operator_review_persistence import (
    OperatorReviewPersistenceService,
    ReviewReadModel,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/planning")

_HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)

_ERROR_STATUS = {
    "candidate_not_approvable": _HTTP_422,
    "candidate_stale": status.HTTP_412_PRECONDITION_FAILED,
    "review_already_decided": status.HTTP_409_CONFLICT,
    "idempotency_key_conflict": status.HTTP_409_CONFLICT,
    "newer_candidate_exists": status.HTTP_409_CONFLICT,
    "lineage_mismatch": status.HTTP_409_CONFLICT,
    "validation_mismatch": _HTTP_422,
    "review_head_stale": status.HTTP_412_PRECONDITION_FAILED,
    "stale_review_head": status.HTTP_412_PRECONDITION_FAILED,
    "integrity_failure": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "promotion_conflict": status.HTTP_409_CONFLICT,
    "review_not_found": status.HTTP_404_NOT_FOUND,
    "review_forbidden": status.HTTP_403_FORBIDDEN,
}


def _audit(
    action: str,
    *,
    session: PlanningSession,
    project: Project,
    current_user: User,
    review_id: str | None = None,
    candidate_checkpoint_id: int | None = None,
    candidate_hash: str | None = None,
    event_id: str | None = None,
    result: str,
    error_code: str | None = None,
) -> None:
    logger.info(
        "operator_review_audit",
        extra={
            "audit_event": "operator_review_" + action,
            "session_id": session.id,
            "project_id": project.id,
            "review_id": review_id,
            "candidate_checkpoint_id": candidate_checkpoint_id,
            "candidate_hash_prefix": (candidate_hash or "")[:12] or None,
            "event_id": event_id,
            "operator_subject": current_user.email,
            "decision": action,
            "result": result,
            "error_code": error_code,
        },
    )


def _problem(code: str, message: str | None = None) -> HTTPException:
    public_code = "review_head_stale" if code == "stale_review_head" else code
    safe_message = {
        "integrity_failure": "Review integrity could not be verified",
        "review_not_found": "Review not found",
        "review_forbidden": "Review action is not authorized",
    }.get(public_code, message or "Review action could not be completed")
    return HTTPException(
        status_code=_ERROR_STATUS.get(public_code, _HTTP_422),
        detail={"code": public_code, "message": safe_message},
    )


def _raise_domain_error(db: Session, exc: Exception) -> None:
    db.rollback()
    if isinstance(exc, ReviewIntegrityError):
        raise _problem("integrity_failure") from exc
    if isinstance(exc, ReviewOperationError):
        conflict = exc.conflict
        raise _problem(conflict.code, conflict.message) from exc
    if isinstance(exc, ReviewDomainError):
        raise _problem("candidate_not_approvable", str(exc)) from exc
    raise _problem("integrity_failure") from exc


def _review_context(
    db: Session, session_id: int, current_user: User
) -> tuple[PlanningSession, Project]:
    session = (
        db.query(PlanningSession)
        .join(Project, Project.id == PlanningSession.project_id)
        .filter(
            PlanningSession.id == int(session_id),
            Project.deleted_at.is_(None),
        )
        .one_or_none()
    )
    if session is None or session.protocol_version != "v2":
        raise _problem("review_not_found")
    project = db.get(Project, session.project_id)
    if project is None or project.deleted_at is not None:
        raise _problem("review_not_found")
    if not is_admin_user(current_user) and project.user_id != current_user.id:
        _audit(
            "authorization_denial",
            session=session,
            project=project,
            current_user=current_user,
            result="denied",
            error_code="review_forbidden",
        )
        raise _problem("review_forbidden")
    return session, project


def _actor(current_user: User, project: Project):
    from app.services.planning.operator_review import ReviewActor

    admin = is_admin_user(current_user)
    role = "administrator" if admin else "project_owner"
    basis = "administrator" if admin else "project_owner"
    return ReviewActor(
        subject=current_user.email,
        role=role,
        authority_basis=basis,
        actor_kind="human",
        authorized=True,
    )


def _binding_payload(binding: ReviewCandidateBindingRequest) -> dict[str, Any]:
    return binding.model_dump(mode="json")


def _decision_request(
    payload: Any,
    *,
    comment: str | None = None,
    reason: str | None = None,
    guidance: str | None = None,
    amendment_id: str | None = None,
    amendment_hash: str | None = None,
) -> ReviewDecisionRequest:
    return ReviewDecisionRequest(
        idempotency_key=payload.idempotency_key,
        candidate_binding=_binding_payload(payload.binding),
        comment=comment,
        reason=reason,
        expected_head_sequence=payload.review_head_sequence,
        expected_head_token=payload.review_head_token,
        guidance=guidance,
        amendment_id=amendment_id,
        amendment_hash=amendment_hash,
    )


def _artifact_identity(checkpoint: Any) -> dict[str, Any] | None:
    if checkpoint is None or checkpoint.status != "accepted":
        return None
    return {
        "artifact_authority": "accepted",
        "checkpoint_id": checkpoint.id,
        "stage_name": checkpoint.stage_name,
        "stage_version": checkpoint.checkpoint_version,
        "content_hash": checkpoint.content_hash,
    }


def _lifecycle_artifact(
    lifecycle: dict[str, Any], *, prefix: str, stage_name: str
) -> dict[str, Any] | None:
    checkpoint_id = lifecycle.get(prefix + "_checkpoint_id")
    content_hash = lifecycle.get(prefix + "_checkpoint_hash")
    if checkpoint_id is None or content_hash is None:
        return None
    return {
        "artifact_authority": "accepted",
        "checkpoint_id": checkpoint_id,
        "stage_name": stage_name,
        "content_hash": content_hash,
    }


def _summary(model: ReviewReadModel) -> ReviewSummaryResponse:
    projection = model.projection
    binding = projection.candidate_binding
    return ReviewSummaryResponse(
        review_id=projection.review_id,
        stage_name=binding.stage_name,
        stage_version=binding.stage_version,
        candidate_checkpoint_id=binding.candidate_checkpoint_id,
        candidate_content_hash=binding.candidate_content_hash,
        validation_hash=binding.validation_hash,
        review_state=projection.state,
        review_required_reasons=projection.review_required_reasons,
        current_event_sequence=projection.current_sequence,
        review_head_token=projection.review_head_token,
        allowed_decisions=projection.allowed_decisions,
        current_accepted_artifact=_artifact_identity(model.accepted_checkpoint),
        terminal_decision=projection.terminal_decision,
        promotion_checkpoint_id=projection.accepted_promotion_checkpoint_id,
        created_at=model.created_at,
        updated_at=model.updated_at,
        stale=projection.stale or projection.superseded,
        integrity_status=(
            "integrity_failure"
            if projection.integrity_error
            else ("stale" if projection.stale or projection.superseded else "valid")
        ),
    )


def _detail(model: ReviewReadModel) -> ReviewDetailResponse:
    projection = model.projection
    binding = projection.candidate_binding
    history = tuple(
        ReviewEventResponse(
            event_id=event.event_id,
            event_type=event.event_type,
            event_sequence=event.event_sequence,
            decision=event.decision_text,
            operator_subject=event.actor.subject if event.actor.is_human else None,
            created_at=event.created_at,
        )
        for event in model.events
    )
    summary = _summary(model)
    return ReviewDetailResponse(
        **summary.model_dump(),
        candidate_binding=binding.to_dict(),
        candidate_content=model.candidate_content,
        validation_evidence=model.validation.to_dict(),
        lineage={
            "input_manifest_id": binding.input_manifest_id,
            "input_manifest_hash": binding.input_manifest_hash,
            "accepted_brief_checkpoint_id": binding.accepted_brief_checkpoint_id,
            "accepted_brief_hash": binding.accepted_brief_hash,
            "predecessors": [item.to_dict() for item in binding.predecessors],
            "session_generation_id": binding.session_generation_id,
            "stage_generation_id": binding.stage_generation_id,
            "stage_configuration_fingerprint": binding.stage_configuration_fingerprint,
        },
        structural_diff=model.structural_diff,
        event_history=history,
        rejection_reason=projection.rejection_reason,
        cancellation_reason=projection.cancellation_reason,
        command_identity=projection.command_identity,
        amendment_id=projection.amendment_id,
        amendment_hash=projection.amendment_hash,
        completion_impact={
            "blocks_required_stage": projection.state
            in {"pending", "stale", "superseded", "rejected", "cancelled"},
            "accepted_authority_created": projection.state == "approved",
        },
    )


def _commit(db: Session) -> None:
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise _problem("integrity_failure") from exc


def _validate_amendment_base(
    service: OperatorReviewPersistenceService,
    payload: AmendReviewRequest,
    session_id: int,
) -> None:
    checkpoint = (
        service.db.query(PlanningCheckpoint)
        .filter(
            PlanningCheckpoint.id == payload.base_checkpoint_id,
            PlanningCheckpoint.planning_session_id == int(session_id),
        )
        .one_or_none()
    )
    if checkpoint is None or checkpoint.content_hash != payload.base_checkpoint_hash:
        raise ReviewOperationError(
            ReviewConflict(
                "lineage_mismatch",
                "amendment base checkpoint binding does not match persistence",
            )
        )
    expected_stage = {
        "planning_brief": "planning_brief",
        "brief_record": "planning_brief",
        "structured_task_plan": "structured_task_plan",
        "task_record": "structured_task_plan",
    }[payload.target_kind]
    if checkpoint.stage_name != expected_stage:
        raise ReviewOperationError(
            ReviewConflict(
                "lineage_mismatch",
                "amendment target kind does not match base checkpoint",
            )
        )


def _action_response(
    db: Session,
    session: PlanningSession,
    service: OperatorReviewPersistenceService,
    result: Any,
    *,
    regeneration: dict[str, Any] | None = None,
    amendment: dict[str, Any] | None = None,
) -> ReviewActionResponse:
    model = service.get_review_read_model(session.id, result.review_id)
    lifecycle = service.build_lifecycle_projection(session.id)
    promotion = result.promotion
    return ReviewActionResponse(
        review_id=result.review_id,
        event_id=result.event_id,
        decision=result.event_type,
        review_state=model.projection.state,
        candidate_checkpoint_id=model.projection.candidate_binding.candidate_checkpoint_id,
        candidate_content_hash=model.projection.candidate_binding.candidate_content_hash,
        promotion_checkpoint_id=promotion.checkpoint_id if promotion else None,
        promotion_content_hash=promotion.content_hash if promotion else None,
        promotion_reason=("operator_approve_unchanged" if promotion else None),
        current_accepted_artifact=_artifact_identity(model.accepted_checkpoint),
        current_accepted_brief=_lifecycle_artifact(
            lifecycle,
            prefix="accepted_brief",
            stage_name="planning_brief",
        ),
        current_accepted_task_plan=_lifecycle_artifact(
            lifecycle,
            prefix="accepted_task_plan",
            stage_name="structured_task_plan",
        ),
        planning_lifecycle_state=str(lifecycle.get("review_state", "generating")),
        completion_reevaluation=(
            {
                "requested": bool(result.completion_reevaluation_requested),
                "state": lifecycle.get("planning_completion_state"),
                "pending": True,
            }
            if result.completion_reevaluation_requested
            else None
        ),
        regeneration=regeneration,
        amendment=amendment,
        idempotent_replay=bool(result.replayed),
    )


@router.get(
    "/sessions/{session_id}/reviews",
    response_model=ReviewListResponse,
    summary="List Protocol v2 operator reviews",
)
def list_reviews(
    session_id: int,
    stage: str | None = Query(default=None, max_length=100),
    state: str | None = Query(default=None, max_length=40),
    candidate_checkpoint_id: int | None = Query(default=None, gt=0),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    session, project = _review_context(db, session_id, current_user)
    service = OperatorReviewPersistenceService(db)
    try:
        models, next_cursor = service.list_review_read_models(
            session.id,
            stage_name=stage,
            state=state,
            candidate_checkpoint_id=candidate_checkpoint_id,
            limit=limit,
            cursor=cursor,
        )
        _audit(
            "read",
            session=session,
            project=project,
            current_user=current_user,
            result="success",
        )
        return ReviewListResponse(
            items=tuple(_summary(model) for model in models),
            next_cursor=str(next_cursor) if next_cursor is not None else None,
        )
    except Exception as exc:
        _raise_domain_error(db, exc)


@router.get(
    "/sessions/{session_id}/reviews/{review_id}",
    response_model=ReviewDetailResponse,
    summary="Inspect one exact Protocol v2 review candidate",
)
def get_review(
    session_id: int,
    review_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    session, project = _review_context(db, session_id, current_user)
    service = OperatorReviewPersistenceService(db)
    try:
        model = service.get_review_read_model(session.id, review_id)
        _audit(
            "read",
            session=session,
            project=project,
            current_user=current_user,
            review_id=review_id,
            candidate_checkpoint_id=model.projection.candidate_binding.candidate_checkpoint_id,
            candidate_hash=model.projection.candidate_binding.candidate_content_hash,
            result="success",
        )
        return _detail(model)
    except Exception as exc:
        _raise_domain_error(db, exc)


def _write_context(
    request: Request,
    session_id: int,
    db: Session,
    current_user: User,
    *,
    action: str,
) -> tuple[PlanningSession, Project, OperatorReviewPersistenceService, Any]:
    session, project = _review_context(db, session_id, current_user)
    enforce_api_rate_limit(
        request,
        action,
        current_user=current_user,
        scope_id=str(project.id),
    )
    return (
        session,
        project,
        OperatorReviewPersistenceService(db),
        _actor(current_user, project),
    )


@router.post(
    "/sessions/{session_id}/reviews/{review_id}/approve",
    response_model=ReviewActionResponse,
)
def approve_review(
    session_id: int,
    review_id: str,
    payload: ApproveReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    session, project, service, actor = _write_context(
        request, session_id, db, current_user, action="review_decision"
    )
    try:
        result = service.approve_review_unchanged(
            review_id,
            actor,
            request=_decision_request(payload, comment=payload.comment),
        )
        _commit(db)
        response = _action_response(db, session, service, result)
        _audit(
            "approval",
            session=session,
            project=project,
            current_user=current_user,
            review_id=review_id,
            candidate_checkpoint_id=response.candidate_checkpoint_id,
            candidate_hash=response.candidate_content_hash,
            event_id=response.event_id,
            result="replayed" if response.idempotent_replay else "success",
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        _audit(
            (
                "stale_conflict"
                if isinstance(exc, ReviewOperationError)
                else "integrity_failure"
            ),
            session=session,
            project=project,
            current_user=current_user,
            review_id=review_id,
            result="failed",
            error_code=getattr(getattr(exc, "conflict", None), "code", None),
        )
        _raise_domain_error(db, exc)


def _terminal_action(
    request: Request,
    session_id: int,
    review_id: str,
    payload: Any,
    db: Session,
    current_user: User,
    *,
    action: str,
    method: str,
    reason_field: str,
) -> ReviewActionResponse:
    session, project, service, actor = _write_context(
        request, session_id, db, current_user, action="review_decision"
    )
    try:
        decision = _decision_request(
            payload,
            reason=getattr(payload, reason_field, None),
            comment=getattr(payload, "comment", None),
        )
        result = getattr(service, method)(review_id, actor, request=decision)
        _commit(db)
        response = _action_response(db, session, service, result)
        _audit(
            action,
            session=session,
            project=project,
            current_user=current_user,
            review_id=review_id,
            candidate_checkpoint_id=response.candidate_checkpoint_id,
            candidate_hash=response.candidate_content_hash,
            event_id=response.event_id,
            result="replayed" if response.idempotent_replay else "success",
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        _raise_domain_error(db, exc)


@router.post(
    "/sessions/{session_id}/reviews/{review_id}/reject",
    response_model=ReviewActionResponse,
)
def reject_review(
    session_id: int,
    review_id: str,
    payload: RejectReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return _terminal_action(
        request,
        session_id,
        review_id,
        payload,
        db,
        current_user,
        action="rejection",
        method="reject_review",
        reason_field="reason",
    )


@router.post(
    "/sessions/{session_id}/reviews/{review_id}/cancel",
    response_model=ReviewActionResponse,
)
def cancel_review(
    session_id: int,
    review_id: str,
    payload: CancelReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return _terminal_action(
        request,
        session_id,
        review_id,
        payload,
        db,
        current_user,
        action="cancellation",
        method="cancel_review",
        reason_field="reason",
    )


@router.post(
    "/sessions/{session_id}/reviews/{review_id}/acknowledge",
    response_model=ReviewActionResponse,
)
def acknowledge_review(
    session_id: int,
    review_id: str,
    payload: AcknowledgeReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return _terminal_action(
        request,
        session_id,
        review_id,
        payload,
        db,
        current_user,
        action="acknowledgment",
        method="acknowledge_review",
        reason_field="_unused",
    )


@router.post(
    "/sessions/{session_id}/reviews/{review_id}/regenerate",
    response_model=ReviewActionResponse,
)
def request_regeneration(
    session_id: int,
    review_id: str,
    payload: RegenerateReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    session, project, service, actor = _write_context(
        request, session_id, db, current_user, action="review_request"
    )
    try:
        result = service.request_regeneration(
            review_id,
            actor,
            request=_decision_request(
                payload, reason=payload.reason, guidance=payload.guidance
            ),
        )
        _commit(db)
        return _action_response(
            db,
            session,
            service,
            result,
            regeneration={
                "requested": True,
                "event_id": result.event_id,
                "command_identity": service.get_review_read_model(
                    session.id, review_id
                ).projection.command_identity,
                "provider_invoked": False,
                "generation_started": False,
                "candidate_created": False,
                "dispatch_state": "pending_future",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_domain_error(db, exc)


@router.post(
    "/sessions/{session_id}/reviews/{review_id}/amend",
    response_model=ReviewActionResponse,
)
def request_amendment(
    session_id: int,
    review_id: str,
    payload: AmendReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    session, project, service, actor = _write_context(
        request, session_id, db, current_user, action="review_request"
    )
    try:
        _validate_amendment_base(service, payload, session.id)
        metadata = {
            "target_kind": payload.target_kind,
            "base_checkpoint_id": payload.base_checkpoint_id,
            "base_checkpoint_hash": payload.base_checkpoint_hash,
            "requested_change_kinds": list(payload.requested_change_kinds),
            "target_record_references": list(payload.target_record_references),
            "instruction": payload.instruction,
            "regeneration_guidance": payload.regeneration_guidance,
            "reason": payload.reason,
        }
        amendment_hash = canonical_json_hash(metadata)
        amendment_id = "amendment-" + amendment_hash[:32]
        guidance = canonical_json_bytes(metadata).decode("utf-8")
        decision = _decision_request(
            payload,
            reason=payload.reason,
            guidance=guidance,
            amendment_id=amendment_id,
            amendment_hash=amendment_hash,
        )
        result = service.request_amendment(
            review_id,
            actor,
            request=decision,
            guidance=guidance,
            amendment_id=amendment_id,
            amendment_hash=amendment_hash,
        )
        _commit(db)
        return _action_response(
            db,
            session,
            service,
            result,
            amendment={
                "event_id": result.event_id,
                "amendment_id": amendment_id,
                "amendment_hash": amendment_hash,
                "base_artifact": {
                    "checkpoint_id": payload.base_checkpoint_id,
                    "content_hash": payload.base_checkpoint_hash,
                },
                "intended_invalidation_effects": [
                    "candidate_superseded_when_future_generation_is recorded"
                ],
                "provider_invoked": False,
                "artifact_amended": False,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_domain_error(db, exc)


__all__ = ["router"]

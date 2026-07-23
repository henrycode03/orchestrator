"""Phase 29D-1 Controlled Apply authorization boundary.

An acceptance decision proves a candidate satisfied its released validation
contract; it does not, by itself, grant permission to mutate a workspace.
This module evaluates ``controlled_apply_policy/1`` — a single deterministic
policy — against one immutable ``ExecutionTaskChangeSet`` and persists an
immutable authorization decision.  It never edits files, invokes Git, runs
commands, dispatches an apply worker, or transitions task lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    ExecutionPlan,
    ExecutionTask,
    ExecutionTaskAcceptanceDecision,
    ExecutionTaskApplyAuthorization,
    ExecutionTaskAttempt,
    ExecutionTaskChangeSet,
    Project,
)
from app.services.execution.candidate_content import (
    CandidateContentStore,
    LocalContentAddressedStore,
)
from app.services.execution.changeset import (
    validate_changeset_path,
    ChangeSetError,
    verify_change_set_integrity,
)
from app.services.planning.operator_review import canonical_json_hash


APPLY_AUTHORIZATION_SCHEMA_VERSION = "execution-task-apply-authorization/1.0"
APPLY_POLICY_ID = "controlled_apply_policy"
APPLY_POLICY_VERSION = 1
AUTHORIZATION_STATUSES = frozenset({"authorized", "blocked", "denied"})

# No independently verified workspace/base-state identity authority exists in
# the current architecture (``Project.workspace_path`` is an unverified
# config string; there is no repository-identity or snapshot model).  Per the
# Phase 29D-1 base-state requirement, this policy version therefore never
# treats a ChangeSet's caller-declared base state as trustworthy and always
# blocks before reaching an ``authorized`` outcome.
BASE_STATE_AUTHORITY_UNAVAILABLE_REASON = "blocked_missing_base_state_authority"
OPERATOR_APPROVAL_REQUIRED_REASON = "blocked_operator_approval_required"


class ApplyAuthorizationError(RuntimeError):
    """Bounded Controlled Apply authorization failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ApplyPolicyDecision:
    status: str
    reason: str
    detail: str | None = None


def evaluate_apply_policy(
    db: Session,
    change_set: ExecutionTaskChangeSet,
    *,
    store: CandidateContentStore | None = None,
) -> ApplyPolicyDecision:
    """Deterministically classify one ChangeSet under ``controlled_apply_policy/1``."""

    plan = db.get(ExecutionPlan, change_set.execution_plan_id)
    if plan is None:
        return ApplyPolicyDecision("blocked", "missing_plan_authority")
    if plan.status != "active" or plan.superseded_by_execution_plan_id is not None:
        return ApplyPolicyDecision("denied", "superseded_plan")

    task = db.get(ExecutionTask, change_set.execution_task_id)
    attempt = db.get(ExecutionTaskAttempt, change_set.execution_task_attempt_id)
    if task is None or attempt is None:
        return ApplyPolicyDecision("blocked", "missing_task_authority")
    if (
        task.execution_plan_id != plan.id
        or attempt.execution_plan_id != plan.id
        or attempt.execution_task_id != task.id
        or attempt.attempt_generation != change_set.attempt_generation
    ):
        return ApplyPolicyDecision("denied", "authority_linkage_mismatch")

    acceptance = db.get(
        ExecutionTaskAcceptanceDecision, change_set.acceptance_decision_id
    )
    if acceptance is None:
        return ApplyPolicyDecision("blocked", "missing_acceptance_authority")
    if (
        acceptance.decision_status != "accepted"
        or acceptance.canonical_decision_hash != change_set.acceptance_decision_hash
    ):
        return ApplyPolicyDecision("denied", "candidate_not_accepted")
    if (
        canonical_json_hash(acceptance.canonical_decision_payload)
        != acceptance.canonical_decision_hash
    ):
        return ApplyPolicyDecision("denied", "acceptance_integrity_failure")

    reader = store or LocalContentAddressedStore()
    changeset_integrity = verify_change_set_integrity(db, change_set.id, store=reader)
    if not changeset_integrity.verified:
        return ApplyPolicyDecision(
            "denied",
            "changeset_integrity_failure",
            ",".join(changeset_integrity.issues) or None,
        )

    try:
        operations = change_set.canonical_changeset_payload.get("operations", [])
        seen_paths: set[str] = set()
        for operation in operations:
            canonical_path = validate_changeset_path(operation.get("path"))
            if canonical_path in seen_paths:
                return ApplyPolicyDecision("denied", "unsafe_operation_duplicate_path")
            seen_paths.add(canonical_path)
    except ChangeSetError as exc:
        return ApplyPolicyDecision("denied", "unsafe_operation_path", exc.code)

    target_project = db.get(Project, change_set.target_project_id)
    if target_project is None:
        return ApplyPolicyDecision("blocked", "missing_target_identity")

    existing_conflicting = (
        db.query(ExecutionTaskApplyAuthorization)
        .filter(
            ExecutionTaskApplyAuthorization.change_set_id == change_set.id,
            ExecutionTaskApplyAuthorization.apply_policy_id == APPLY_POLICY_ID,
            ExecutionTaskApplyAuthorization.apply_policy_version
            == APPLY_POLICY_VERSION,
        )
        .count()
    )
    if existing_conflicting > 0:
        return ApplyPolicyDecision("denied", "conflicting_authorization_exists")

    # Base-state trust is the final architectural gate: no independently
    # verifiable workspace/base-state identity exists yet, so a ChangeSet
    # that passes every prior check still cannot be authorized to apply.
    return ApplyPolicyDecision("blocked", BASE_STATE_AUTHORITY_UNAVAILABLE_REASON)


@dataclass(frozen=True)
class AuthorizeApplyCommand:
    change_set_id: int
    authorization_idempotency_key: str
    decision_actor_type: str = "operator"
    decision_actor_id: str = "system"


@dataclass(frozen=True)
class ApplyAuthorizationResult:
    authorization: ExecutionTaskApplyAuthorization
    replayed: bool = False


def _input_payload(
    command: AuthorizeApplyCommand, change_set: ExecutionTaskChangeSet
) -> dict[str, Any]:
    return {
        "schema_version": APPLY_AUTHORIZATION_SCHEMA_VERSION,
        "change_set_id": int(change_set.id),
        "change_set_hash": change_set.changeset_sha256,
        "acceptance_decision_id": int(change_set.acceptance_decision_id),
        "acceptance_decision_hash": change_set.acceptance_decision_hash,
        "apply_policy_id": APPLY_POLICY_ID,
        "apply_policy_version": APPLY_POLICY_VERSION,
        "authorization_idempotency_key": command.authorization_idempotency_key,
        "decision_actor_type": command.decision_actor_type,
        "decision_actor_id": command.decision_actor_id,
    }


class ApplyAuthorizationService:
    """Evaluate and persist one immutable Controlled Apply authorization."""

    def __init__(
        self,
        db: Session,
        *,
        store: CandidateContentStore | None = None,
        now: Any = None,
    ):
        self.db = db
        self.store = store or LocalContentAddressedStore()
        self._now = now or (lambda: datetime.now(timezone.utc))

    def authorize(self, command: AuthorizeApplyCommand) -> ApplyAuthorizationResult:
        change_set = self.db.get(ExecutionTaskChangeSet, int(command.change_set_id))
        if change_set is None:
            raise ApplyAuthorizationError(
                "apply_authorization_changeset_missing", "ChangeSet does not exist"
            )

        input_payload = _input_payload(command, change_set)
        input_hash = canonical_json_hash(input_payload)

        existing = (
            self.db.query(ExecutionTaskApplyAuthorization)
            .filter(
                ExecutionTaskApplyAuthorization.authorization_idempotency_key
                == command.authorization_idempotency_key
            )
            .one_or_none()
        )
        if existing is not None:
            if existing.canonical_input_hash != input_hash:
                raise ApplyAuthorizationError(
                    "apply_authorization_idempotency_conflict",
                    "authorization key is bound to a different request",
                )
            return ApplyAuthorizationResult(existing, replayed=True)

        conflicting = (
            self.db.query(ExecutionTaskApplyAuthorization)
            .filter(
                ExecutionTaskApplyAuthorization.change_set_id == change_set.id,
                ExecutionTaskApplyAuthorization.apply_policy_id == APPLY_POLICY_ID,
                ExecutionTaskApplyAuthorization.apply_policy_version
                == APPLY_POLICY_VERSION,
            )
            .one_or_none()
        )
        if conflicting is not None:
            raise ApplyAuthorizationError(
                "apply_authorization_conflict",
                "this ChangeSet already has an authorization under this policy",
            )

        decision = evaluate_apply_policy(self.db, change_set, store=self.store)
        now = self._now()
        decision_payload = {
            "schema_version": APPLY_AUTHORIZATION_SCHEMA_VERSION,
            "execution_plan_id": change_set.execution_plan_id,
            "execution_task_id": change_set.execution_task_id,
            "execution_task_attempt_id": change_set.execution_task_attempt_id,
            "attempt_generation": change_set.attempt_generation,
            "change_set_id": change_set.id,
            "change_set_hash": change_set.changeset_sha256,
            "acceptance_decision_id": change_set.acceptance_decision_id,
            "acceptance_decision_hash": change_set.acceptance_decision_hash,
            "target_project_id": change_set.target_project_id,
            "target_workspace_identity": change_set.target_workspace_identity,
            "base_state_hash": change_set.base_state_hash,
            "apply_policy_id": APPLY_POLICY_ID,
            "apply_policy_version": APPLY_POLICY_VERSION,
            "authorization_status": decision.status,
            "decision_reason": decision.reason,
            "bounded_detail": decision.detail,
        }
        decision_hash = canonical_json_hash(decision_payload)
        command_hash = canonical_json_hash(input_payload)

        row = ExecutionTaskApplyAuthorization(
            execution_plan_id=change_set.execution_plan_id,
            execution_task_id=change_set.execution_task_id,
            execution_task_attempt_id=change_set.execution_task_attempt_id,
            attempt_generation=change_set.attempt_generation,
            change_set_id=change_set.id,
            change_set_hash=change_set.changeset_sha256,
            acceptance_decision_id=change_set.acceptance_decision_id,
            acceptance_decision_hash=change_set.acceptance_decision_hash,
            target_project_id=change_set.target_project_id,
            target_workspace_identity=change_set.target_workspace_identity,
            base_state_hash=change_set.base_state_hash,
            apply_policy_id=APPLY_POLICY_ID,
            apply_policy_version=APPLY_POLICY_VERSION,
            authorization_status=decision.status,
            decision_reason=decision.reason,
            bounded_detail=decision.detail,
            canonical_input_payload=input_payload,
            canonical_input_hash=input_hash,
            canonical_decision_payload=decision_payload,
            canonical_decision_hash=decision_hash,
            authorization_idempotency_key=command.authorization_idempotency_key,
            deterministic_authorization_command_id=(
                f"apply-authorization-command:{command_hash}"
            ),
            decision_actor_type=command.decision_actor_type,
            decision_actor_id=command.decision_actor_id,
            decided_at=now,
        )
        try:
            with self.db.begin_nested():
                self.db.add(row)
                self.db.flush()
        except IntegrityError as exc:
            replay = (
                self.db.query(ExecutionTaskApplyAuthorization)
                .filter(
                    ExecutionTaskApplyAuthorization.authorization_idempotency_key
                    == command.authorization_idempotency_key
                )
                .one_or_none()
            )
            if replay is not None and replay.canonical_input_hash == input_hash:
                return ApplyAuthorizationResult(replay, replayed=True)
            raise ApplyAuthorizationError(
                "apply_authorization_insert_conflict",
                "authorization conflicts with canonical authority",
            ) from exc
        return ApplyAuthorizationResult(row)


@dataclass(frozen=True)
class ApplyAuthorizationIntegrityResult:
    execution_plan_id: int | None
    execution_task_id: int | None
    verified: bool
    issues: tuple[str, ...] = ()


def verify_apply_authorization_integrity(
    db: Session,
    authorization_id: int,
    *,
    store: CandidateContentStore | None = None,
) -> ApplyAuthorizationIntegrityResult:
    """Read-only re-verification of one persisted authorization."""

    row = db.get(ExecutionTaskApplyAuthorization, int(authorization_id))
    if row is None:
        return ApplyAuthorizationIntegrityResult(
            None, None, False, ("apply_authorization_missing",)
        )
    issues: list[str] = []
    change_set = db.get(ExecutionTaskChangeSet, row.change_set_id)
    if change_set is None:
        issues.append("apply_authorization_changeset_missing")
    else:
        if change_set.changeset_sha256 != row.change_set_hash:
            issues.append("apply_authorization_changeset_hash_mismatch")
        changeset_integrity = verify_change_set_integrity(
            db, change_set.id, store=store
        )
        if not changeset_integrity.verified:
            issues.append("apply_authorization_changeset_integrity_failure")
    if canonical_json_hash(row.canonical_input_payload) != row.canonical_input_hash:
        issues.append("apply_authorization_input_hash_mismatch")
    if (
        canonical_json_hash(row.canonical_decision_payload)
        != row.canonical_decision_hash
    ):
        issues.append("apply_authorization_decision_hash_mismatch")
    if (
        row.canonical_decision_payload.get("authorization_status")
        != row.authorization_status
    ):
        issues.append("apply_authorization_status_tampered")
    duplicate = (
        db.query(ExecutionTaskApplyAuthorization)
        .filter(
            ExecutionTaskApplyAuthorization.change_set_id == row.change_set_id,
            ExecutionTaskApplyAuthorization.apply_policy_id == row.apply_policy_id,
            ExecutionTaskApplyAuthorization.apply_policy_version
            == row.apply_policy_version,
        )
        .count()
    )
    if duplicate != 1:
        issues.append("apply_authorization_duplicate_active")
    return ApplyAuthorizationIntegrityResult(
        row.execution_plan_id,
        row.execution_task_id,
        not issues,
        tuple(sorted(set(issues))),
    )


__all__ = [
    "APPLY_AUTHORIZATION_SCHEMA_VERSION",
    "APPLY_POLICY_ID",
    "APPLY_POLICY_VERSION",
    "AUTHORIZATION_STATUSES",
    "BASE_STATE_AUTHORITY_UNAVAILABLE_REASON",
    "OPERATOR_APPROVAL_REQUIRED_REASON",
    "ApplyAuthorizationError",
    "ApplyAuthorizationIntegrityResult",
    "ApplyAuthorizationResult",
    "ApplyAuthorizationService",
    "ApplyPolicyDecision",
    "AuthorizeApplyCommand",
    "evaluate_apply_policy",
    "verify_apply_authorization_integrity",
]

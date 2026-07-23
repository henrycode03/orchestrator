"""Phase 29D-1 immutable ChangeSet authority.

A ChangeSet is a proposed mutation intent only.  It is created exclusively
from one explicitly structured, byte-backed ``ExecutionTaskCandidateContent``
row bearing the exact ``application/vnd.orchestrator.changeset+json`` media
type, bound to exactly one accepted ``ExecutionTaskAcceptanceDecision``.  This
module never edits files, invokes Git, executes commands, or dispatches an
apply worker; it only authorizes the *description* of a mutation, never the
mutation itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    ExecutionPlan,
    ExecutionTask,
    ExecutionTaskAcceptanceDecision,
    ExecutionTaskAttempt,
    ExecutionTaskAttemptOutcome,
    ExecutionTaskCandidateContent,
    ExecutionTaskChangeSet,
    ExecutionTaskChangeSetOperation,
    ExecutionEvidence,
    Project,
)
from app.services.execution.candidate_content import (
    CandidateContentError,
    CandidateContentStore,
    LocalContentAddressedStore,
    verify_candidate_content_integrity,
)
from app.services.execution.execution_evidence import (
    ExecutionEvidenceError,
    parse_execution_evidence_reference,
    resolve_execution_evidence_reference,
)
from app.services.planning.operator_review import canonical_json_hash


CHANGESET_SCHEMA_VERSION = "execution-task-change-set/1.0"
CHANGESET_FORMAT = "orchestrator-changeset/1"
CHANGESET_MEDIA_TYPE = "application/vnd.orchestrator.changeset+json"
SUPPORTED_OPERATIONS = frozenset({"create_file", "replace_file", "delete_file"})
MAX_CHANGESET_BYTES = 1_048_576
MAX_OPERATIONS = 200
MAX_PATH_LENGTH = 1024
MAX_PATH_SEGMENTS = 64
MAX_SEGMENT_LENGTH = 255
PROTECTED_PATH_SEGMENTS = frozenset({".git", ".orchestrator", ".claude"})
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_CONTENT_REF_RE = re.compile(r"^candidate-content://([1-9][0-9]{0,18})$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ALLOWED_BASE_STATE_KEYS = frozenset(
    {"project_id", "workspace_identity", "repository_head", "clean"}
)
_ALLOWED_OPERATION_KEYS = {
    "create_file": frozenset({"operation", "path", "content_reference"}),
    "replace_file": frozenset(
        {"operation", "path", "expected_previous_sha256", "content_reference"}
    ),
    "delete_file": frozenset({"operation", "path", "expected_previous_sha256"}),
}


class ChangeSetError(RuntimeError):
    """Bounded ChangeSet authority failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def validate_changeset_path(value: Any) -> str:
    """Canonicalize and validate one bounded relative operation path.

    Never accesses the filesystem; only validates the canonical operation
    intent expressed by the structured ChangeSet payload.
    """

    if not isinstance(value, str) or value == "":
        raise ChangeSetError(
            "changeset_path_invalid", "path must be a non-empty string"
        )
    if len(value) > MAX_PATH_LENGTH or _CONTROL_RE.search(value):
        raise ChangeSetError("changeset_path_invalid", "path is malformed")
    if "\\" in value:
        raise ChangeSetError("changeset_path_invalid", "path must use '/' separators")
    if "://" in value:
        raise ChangeSetError("changeset_path_invalid", "path must not be a URL/URI")
    if value.startswith("/"):
        raise ChangeSetError("changeset_path_invalid", "path must be relative")
    if value.startswith("~"):
        raise ChangeSetError("changeset_path_invalid", "home expansion is not allowed")
    if re.match(r"^[A-Za-z]:", value):
        raise ChangeSetError("changeset_path_invalid", "drive letters are not allowed")
    segments = value.split("/")
    if len(segments) > MAX_PATH_SEGMENTS:
        raise ChangeSetError("changeset_path_invalid", "path has too many segments")
    normalized: list[str] = []
    for segment in segments:
        if segment == "":
            raise ChangeSetError("changeset_path_invalid", "empty path segment")
        if segment in (".", ".."):
            raise ChangeSetError(
                "changeset_path_invalid", "traversal segments are not allowed"
            )
        if len(segment) > MAX_SEGMENT_LENGTH:
            raise ChangeSetError("changeset_path_invalid", "path segment too long")
        normalized.append(segment)
    canonical = "/".join(normalized)
    if normalized[0].lower() in PROTECTED_PATH_SEGMENTS:
        raise ChangeSetError("changeset_path_protected", "path is protected")
    return canonical


def _parse_content_reference(
    value: Any,
) -> tuple[str, int, str]:
    """Return (scheme, identifier, normalized reference) or raise."""

    if not isinstance(value, str) or value == "":
        raise ChangeSetError(
            "changeset_content_reference_invalid", "content reference is required"
        )
    match = _CANDIDATE_CONTENT_REF_RE.fullmatch(value)
    if match is not None:
        identifier = int(match.group(1))
        return "candidate-content", identifier, f"candidate-content://{identifier}"
    try:
        parsed = parse_execution_evidence_reference(value)
    except ExecutionEvidenceError as exc:
        raise ChangeSetError(
            "changeset_content_reference_unsupported",
            "content reference scheme is unsupported",
        ) from exc
    if parsed.scheme != "execution-evidence":
        raise ChangeSetError(
            "changeset_content_reference_unsupported",
            "only exact-id content references are supported",
        )
    return "execution-evidence", int(parsed.identifier), parsed.normalized


@dataclass(frozen=True)
class ResolvedOperation:
    operation: str
    canonical_path: str
    expected_previous_sha256: str | None
    content_reference: str | None
    content_reference_scheme: str | None
    content_reference_id: int | None
    content_sha256: str | None
    content_media_type: str | None
    content_byte_length: int | None


def _parse_json_strict(content: bytes) -> Any:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ChangeSetError(
            "changeset_content_not_utf8", "ChangeSet bytes are not valid UTF-8"
        ) from exc

    def duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ChangeSetError(
                    "changeset_content_invalid",
                    "duplicate JSON object keys are not allowed",
                )
            result[key] = item
        return result

    try:
        return json.loads(text, object_pairs_hook=duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ChangeSetError(
            "changeset_content_invalid", "ChangeSet bytes are not valid JSON"
        ) from exc


def _require_exact_keys(value: Any, allowed: frozenset[str], field: str) -> dict:
    if not isinstance(value, dict):
        raise ChangeSetError("changeset_content_invalid", f"{field} must be an object")
    extra = set(value.keys()) - allowed
    if extra:
        raise ChangeSetError(
            "changeset_content_invalid", f"{field} contains unsupported fields"
        )
    return value


def _validate_base_state(value: Any) -> dict[str, Any]:
    payload = _require_exact_keys(value, _ALLOWED_BASE_STATE_KEYS, "base_state")
    project_id = payload.get("project_id")
    if (
        not isinstance(project_id, int)
        or isinstance(project_id, bool)
        or project_id <= 0
    ):
        raise ChangeSetError(
            "changeset_base_state_invalid", "base_state.project_id is invalid"
        )
    workspace_identity = payload.get("workspace_identity")
    if workspace_identity is not None and (
        not isinstance(workspace_identity, str)
        or workspace_identity == ""
        or len(workspace_identity) > 255
        or _CONTROL_RE.search(workspace_identity)
    ):
        raise ChangeSetError(
            "changeset_base_state_invalid",
            "base_state.workspace_identity is invalid",
        )
    repository_head = payload.get("repository_head")
    if repository_head is not None and (
        not isinstance(repository_head, str)
        or repository_head == ""
        or len(repository_head) > 128
        or _CONTROL_RE.search(repository_head)
    ):
        raise ChangeSetError(
            "changeset_base_state_invalid", "base_state.repository_head is invalid"
        )
    clean = payload.get("clean")
    if clean is not None and not isinstance(clean, bool):
        raise ChangeSetError(
            "changeset_base_state_invalid", "base_state.clean must be a boolean"
        )
    return {
        "project_id": project_id,
        "workspace_identity": workspace_identity,
        "repository_head": repository_head,
        "clean": clean,
    }


def _resolve_operation(
    db: Session,
    raw: Any,
    *,
    execution_plan_id: int,
    execution_task_id: int,
    execution_task_attempt_id: int,
    store: CandidateContentStore,
) -> ResolvedOperation:
    if not isinstance(raw, dict) or "operation" not in raw:
        raise ChangeSetError(
            "changeset_operation_invalid", "operation must be an object"
        )
    kind = raw.get("operation")
    if kind not in SUPPORTED_OPERATIONS:
        raise ChangeSetError(
            "changeset_operation_unsupported", "operation type is not supported"
        )
    allowed_keys = _ALLOWED_OPERATION_KEYS[kind]
    _require_exact_keys(raw, allowed_keys, "operation")
    canonical_path = validate_changeset_path(raw.get("path"))

    expected_previous_sha256: str | None = None
    if kind in ("replace_file", "delete_file"):
        declared = raw.get("expected_previous_sha256")
        if not isinstance(declared, str) or not _HASH_RE.fullmatch(declared):
            raise ChangeSetError(
                "changeset_operation_invalid",
                "expected_previous_sha256 is required and must be a sha256 hash",
            )
        expected_previous_sha256 = declared

    content_reference: str | None = None
    content_reference_scheme: str | None = None
    content_reference_id: int | None = None
    content_sha256: str | None = None
    content_media_type: str | None = None
    content_byte_length: int | None = None

    if kind in ("create_file", "replace_file"):
        scheme, identifier, normalized = _parse_content_reference(
            raw.get("content_reference")
        )
        content_reference = normalized
        content_reference_scheme = scheme
        content_reference_id = identifier
        if scheme == "candidate-content":
            row = db.get(ExecutionTaskCandidateContent, identifier)
            if row is None:
                raise ChangeSetError(
                    "changeset_content_reference_missing",
                    "referenced candidate content does not exist",
                )
            if (
                row.execution_plan_id != execution_plan_id
                or row.execution_task_id != execution_task_id
                or row.execution_task_attempt_id != execution_task_attempt_id
            ):
                raise ChangeSetError(
                    "changeset_content_reference_authority_mismatch",
                    "content reference is not bound to this ChangeSet authority",
                )
            integrity = verify_candidate_content_integrity(db, row.id, store=store)
            if not integrity.verified:
                raise ChangeSetError(
                    "changeset_content_reference_integrity_failure",
                    "referenced candidate content failed integrity verification",
                )
            content_sha256 = row.content_sha256
            content_media_type = row.media_type
            content_byte_length = row.byte_length
        else:
            resolution = resolve_execution_evidence_reference(
                db, normalized, store=store
            )
            if not resolution.verified:
                raise ChangeSetError(
                    "changeset_content_reference_integrity_failure",
                    "referenced execution evidence failed integrity verification",
                )
            evidence_row = db.get(ExecutionEvidence, identifier)
            if (
                evidence_row is None
                or evidence_row.execution_plan_id != execution_plan_id
                or evidence_row.execution_task_id != execution_task_id
                or evidence_row.execution_task_attempt_id != execution_task_attempt_id
            ):
                raise ChangeSetError(
                    "changeset_content_reference_authority_mismatch",
                    "content reference is not bound to this ChangeSet authority",
                )
            content_sha256 = resolution.content_sha256
            content_media_type = resolution.media_type
            content_byte_length = resolution.byte_length

    return ResolvedOperation(
        operation=kind,
        canonical_path=canonical_path,
        expected_previous_sha256=expected_previous_sha256,
        content_reference=content_reference,
        content_reference_scheme=content_reference_scheme,
        content_reference_id=content_reference_id,
        content_sha256=content_sha256,
        content_media_type=content_media_type,
        content_byte_length=content_byte_length,
    )


@dataclass(frozen=True)
class IngestChangeSetCommand:
    execution_plan_id: int
    execution_task_id: int
    execution_task_attempt_id: int
    attempt_generation: int
    candidate_outcome_id: int
    acceptance_decision_id: int
    source_candidate_content_id: int
    ingestion_idempotency_key: str
    creation_actor_type: str = "operator"
    creation_actor_id: str = "system"


@dataclass(frozen=True)
class ChangeSetIngestionResult:
    change_set: ExecutionTaskChangeSet
    replayed: bool = False


def _ingestion_payload(
    command: IngestChangeSetCommand,
    *,
    canonical_changeset_payload: dict[str, Any],
    changeset_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHANGESET_SCHEMA_VERSION,
        "execution_plan_id": int(command.execution_plan_id),
        "execution_task_id": int(command.execution_task_id),
        "execution_task_attempt_id": int(command.execution_task_attempt_id),
        "attempt_generation": int(command.attempt_generation),
        "candidate_outcome_id": int(command.candidate_outcome_id),
        "acceptance_decision_id": int(command.acceptance_decision_id),
        "source_candidate_content_id": int(command.source_candidate_content_id),
        "changeset_sha256": changeset_sha256,
        "ingestion_idempotency_key": command.ingestion_idempotency_key,
        "creation_actor_type": command.creation_actor_type,
        "creation_actor_id": command.creation_actor_id,
    }


class ChangeSetIngestionService:
    """Parse and persist one immutable ChangeSet from an accepted candidate."""

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

    def ingest(self, command: IngestChangeSetCommand) -> ChangeSetIngestionResult:
        (
            plan,
            task,
            attempt,
            outcome,
            acceptance,
            content,
        ) = self._authorize(command)

        try:
            content_bytes = self.store.read(content.storage_key)
        except CandidateContentError as exc:
            raise ChangeSetError(
                "changeset_source_content_unreadable", exc.message
            ) from exc
        if (
            len(content_bytes) != content.byte_length
            or hashlib.sha256(content_bytes).hexdigest() != content.content_sha256
        ):
            raise ChangeSetError(
                "changeset_source_content_integrity_failure",
                "source candidate content bytes failed independent verification",
            )

        parsed = _parse_json_strict(content_bytes)
        payload = _require_exact_keys(
            parsed, frozenset({"format", "base_state", "operations"}), "changeset"
        )
        if payload.get("format") != CHANGESET_FORMAT:
            raise ChangeSetError(
                "changeset_format_unsupported",
                "ChangeSet format is missing or unsupported",
            )
        base_state = _validate_base_state(payload.get("base_state"))
        if base_state["project_id"] != plan.project_id:
            raise ChangeSetError(
                "changeset_base_state_project_mismatch",
                "base_state.project_id does not match the task's project",
            )
        target_project = self.db.get(Project, base_state["project_id"])
        if target_project is None:
            raise ChangeSetError(
                "changeset_target_project_missing", "target project does not exist"
            )

        operations_raw = payload.get("operations")
        if not isinstance(operations_raw, list) or not operations_raw:
            raise ChangeSetError(
                "changeset_operations_invalid",
                "operations must be a non-empty bounded list",
            )
        if len(operations_raw) > MAX_OPERATIONS:
            raise ChangeSetError("changeset_operations_invalid", "too many operations")

        resolved_operations: list[ResolvedOperation] = []
        seen_paths: set[str] = set()
        for raw_operation in operations_raw:
            resolved = _resolve_operation(
                self.db,
                raw_operation,
                execution_plan_id=plan.id,
                execution_task_id=task.id,
                execution_task_attempt_id=attempt.id,
                store=self.store,
            )
            if resolved.canonical_path in seen_paths:
                raise ChangeSetError(
                    "changeset_operation_duplicate_path",
                    "duplicate canonical path across operations",
                )
            seen_paths.add(resolved.canonical_path)
            resolved_operations.append(resolved)

        canonical_operations = [
            {
                "operation": item.operation,
                "path": item.canonical_path,
                "expected_previous_sha256": item.expected_previous_sha256,
                "content_reference": item.content_reference,
            }
            for item in resolved_operations
        ]
        canonical_changeset_payload = {
            "format": CHANGESET_FORMAT,
            "base_state": base_state,
            "operations": canonical_operations,
        }
        changeset_sha256 = canonical_json_hash(canonical_changeset_payload)
        base_state_hash = canonical_json_hash(base_state)

        ingestion_payload = _ingestion_payload(
            command,
            canonical_changeset_payload=canonical_changeset_payload,
            changeset_sha256=changeset_sha256,
        )
        command_hash = canonical_json_hash(ingestion_payload)

        existing = (
            self.db.query(ExecutionTaskChangeSet)
            .filter(
                ExecutionTaskChangeSet.ingestion_idempotency_key
                == command.ingestion_idempotency_key
            )
            .one_or_none()
        )
        if existing is not None:
            if existing.canonical_ingestion_command_hash != command_hash:
                raise ChangeSetError(
                    "changeset_idempotency_conflict",
                    "ingestion key is bound to a different ChangeSet",
                )
            return ChangeSetIngestionResult(existing, replayed=True)

        now = self._now()
        metadata_payload = {
            "schema_version": CHANGESET_SCHEMA_VERSION,
            "execution_plan_id": plan.id,
            "execution_task_id": task.id,
            "execution_task_attempt_id": attempt.id,
            "attempt_generation": attempt.attempt_generation,
            "candidate_outcome_id": outcome.id,
            "source_candidate_content_id": content.id,
            "source_candidate_content_sha256": content.content_sha256,
            "acceptance_decision_id": acceptance.id,
            "acceptance_decision_hash": acceptance.canonical_decision_hash,
            "changeset_format": CHANGESET_FORMAT,
            "media_type": CHANGESET_MEDIA_TYPE,
            "target_project_id": target_project.id,
            "operation_count": len(resolved_operations),
            "changeset_sha256": changeset_sha256,
            "base_state_hash": base_state_hash,
        }
        row = ExecutionTaskChangeSet(
            execution_plan_id=plan.id,
            execution_task_id=task.id,
            execution_task_attempt_id=attempt.id,
            attempt_generation=attempt.attempt_generation,
            candidate_outcome_id=outcome.id,
            source_candidate_content_id=content.id,
            source_candidate_content_sha256=content.content_sha256,
            acceptance_decision_id=acceptance.id,
            acceptance_decision_hash=acceptance.canonical_decision_hash,
            changeset_format=CHANGESET_FORMAT,
            media_type=CHANGESET_MEDIA_TYPE,
            target_project_id=target_project.id,
            target_workspace_identity=base_state["workspace_identity"],
            base_state_payload=base_state,
            base_state_hash=base_state_hash,
            operation_count=len(resolved_operations),
            canonical_changeset_payload=canonical_changeset_payload,
            changeset_sha256=changeset_sha256,
            canonical_metadata_payload=metadata_payload,
            canonical_metadata_hash=canonical_json_hash(metadata_payload),
            ingestion_idempotency_key=command.ingestion_idempotency_key,
            canonical_ingestion_command_payload=ingestion_payload,
            canonical_ingestion_command_hash=command_hash,
            creation_actor_type=command.creation_actor_type,
            creation_actor_id=command.creation_actor_id,
            created_at=now,
        )
        try:
            with self.db.begin_nested():
                self.db.add(row)
                self.db.flush()
                for index, item in enumerate(resolved_operations):
                    self.db.add(
                        ExecutionTaskChangeSetOperation(
                            change_set_id=row.id,
                            operation_index=index,
                            operation=item.operation,
                            canonical_path=item.canonical_path,
                            expected_previous_sha256=item.expected_previous_sha256,
                            content_reference=item.content_reference,
                            content_reference_scheme=item.content_reference_scheme,
                            content_reference_id=item.content_reference_id,
                            content_sha256=item.content_sha256,
                            content_media_type=item.content_media_type,
                            content_byte_length=item.content_byte_length,
                            created_at=now,
                        )
                    )
                self.db.flush()
        except IntegrityError as exc:
            replay = (
                self.db.query(ExecutionTaskChangeSet)
                .filter(
                    ExecutionTaskChangeSet.ingestion_idempotency_key
                    == command.ingestion_idempotency_key
                )
                .one_or_none()
            )
            if (
                replay is not None
                and replay.canonical_ingestion_command_hash == command_hash
            ):
                return ChangeSetIngestionResult(replay, replayed=True)
            raise ChangeSetError(
                "changeset_insert_conflict",
                "ChangeSet metadata conflicts with canonical authority",
            ) from exc
        return ChangeSetIngestionResult(row)

    def _authorize(self, command: IngestChangeSetCommand) -> tuple[
        ExecutionPlan,
        ExecutionTask,
        ExecutionTaskAttempt,
        ExecutionTaskAttemptOutcome,
        ExecutionTaskAcceptanceDecision,
        ExecutionTaskCandidateContent,
    ]:
        plan = self.db.get(ExecutionPlan, int(command.execution_plan_id))
        task = self.db.get(ExecutionTask, int(command.execution_task_id))
        attempt = self.db.get(
            ExecutionTaskAttempt, int(command.execution_task_attempt_id)
        )
        outcome = self.db.get(
            ExecutionTaskAttemptOutcome, int(command.candidate_outcome_id)
        )
        acceptance = self.db.get(
            ExecutionTaskAcceptanceDecision, int(command.acceptance_decision_id)
        )
        content = self.db.get(
            ExecutionTaskCandidateContent, int(command.source_candidate_content_id)
        )
        if any(
            item is None for item in (plan, task, attempt, outcome, acceptance, content)
        ):
            raise ChangeSetError(
                "changeset_authority_missing", "ChangeSet authority is incomplete"
            )
        assert plan is not None
        assert task is not None
        assert attempt is not None
        assert outcome is not None
        assert acceptance is not None
        assert content is not None
        if plan.status != "active" or plan.superseded_by_execution_plan_id is not None:
            raise ChangeSetError(
                "changeset_authority_invalid", "execution plan is not active"
            )
        if (
            task.execution_plan_id != plan.id
            or attempt.execution_plan_id != plan.id
            or attempt.execution_task_id != task.id
            or attempt.attempt_generation != int(command.attempt_generation)
            or outcome.execution_plan_id != plan.id
            or outcome.execution_task_id != task.id
            or outcome.execution_task_attempt_id != attempt.id
        ):
            raise ChangeSetError(
                "changeset_authority_invalid",
                "ChangeSet authority linkage is inconsistent",
            )
        if (
            acceptance.execution_plan_id != plan.id
            or acceptance.execution_task_id != task.id
            or acceptance.execution_task_attempt_id != attempt.id
            or acceptance.candidate_outcome_id != outcome.id
        ):
            raise ChangeSetError(
                "changeset_authority_invalid",
                "acceptance decision is not bound to this attempt/outcome",
            )
        if acceptance.decision_status != "accepted":
            raise ChangeSetError(
                "changeset_candidate_not_accepted",
                "candidate outcome was not accepted",
            )
        if (
            canonical_json_hash(acceptance.canonical_decision_payload)
            != acceptance.canonical_decision_hash
        ):
            raise ChangeSetError(
                "changeset_acceptance_integrity_failure",
                "acceptance decision failed tamper verification",
            )
        if (
            content.execution_plan_id != plan.id
            or content.execution_task_id != task.id
            or content.execution_task_attempt_id != attempt.id
            or content.candidate_outcome_id != outcome.id
        ):
            raise ChangeSetError(
                "changeset_source_content_not_accepted_candidate",
                "source content is not the accepted candidate's own content",
            )
        if content.media_type != CHANGESET_MEDIA_TYPE:
            raise ChangeSetError(
                "changeset_media_type_unsupported",
                "candidate content does not use the required ChangeSet media type",
            )
        content_integrity = verify_candidate_content_integrity(
            self.db, content.id, store=self.store
        )
        if not content_integrity.verified:
            raise ChangeSetError(
                "changeset_source_content_integrity_failure",
                "source candidate content failed integrity verification",
            )
        return plan, task, attempt, outcome, acceptance, content


@dataclass(frozen=True)
class ChangeSetIntegrityResult:
    execution_plan_id: int | None
    execution_task_id: int | None
    verified: bool
    issues: tuple[str, ...] = ()


def verify_change_set_integrity(
    db: Session, change_set_id: int, *, store: CandidateContentStore | None = None
) -> ChangeSetIntegrityResult:
    """Read-only re-verification of one persisted ChangeSet authority."""

    row = db.get(ExecutionTaskChangeSet, int(change_set_id))
    if row is None:
        return ChangeSetIntegrityResult(None, None, False, ("changeset_missing",))
    issues: list[str] = []
    reader = store or LocalContentAddressedStore()

    content = db.get(ExecutionTaskCandidateContent, row.source_candidate_content_id)
    acceptance = db.get(ExecutionTaskAcceptanceDecision, row.acceptance_decision_id)
    if content is None or acceptance is None:
        issues.append("changeset_authority_missing")
    else:
        if content.content_sha256 != row.source_candidate_content_sha256:
            issues.append("changeset_source_content_hash_mismatch")
        if acceptance.canonical_decision_hash != row.acceptance_decision_hash:
            issues.append("changeset_acceptance_hash_mismatch")
        content_integrity = verify_candidate_content_integrity(
            db, content.id, store=reader
        )
        if not content_integrity.verified:
            issues.append("changeset_source_content_integrity_failure")

    if canonical_json_hash(row.canonical_changeset_payload) != row.changeset_sha256:
        issues.append("changeset_payload_tampered")
    if canonical_json_hash(row.base_state_payload) != row.base_state_hash:
        issues.append("changeset_base_state_hash_mismatch")
    if (
        canonical_json_hash(row.canonical_metadata_payload)
        != row.canonical_metadata_hash
    ):
        issues.append("changeset_metadata_hash_mismatch")
    if (
        canonical_json_hash(row.canonical_ingestion_command_payload)
        != row.canonical_ingestion_command_hash
    ):
        issues.append("changeset_ingestion_hash_mismatch")

    operations = (
        db.query(ExecutionTaskChangeSetOperation)
        .filter(ExecutionTaskChangeSetOperation.change_set_id == row.id)
        .order_by(ExecutionTaskChangeSetOperation.operation_index)
        .all()
    )
    if len(operations) != row.operation_count:
        issues.append("changeset_operation_count_mismatch")
    expected_paths = {
        item["path"] for item in row.canonical_changeset_payload.get("operations", [])
    }
    actual_paths = {item.canonical_path for item in operations}
    if expected_paths != actual_paths:
        issues.append("changeset_operation_path_mismatch")

    return ChangeSetIntegrityResult(
        row.execution_plan_id,
        row.execution_task_id,
        not issues,
        tuple(sorted(set(issues))),
    )


__all__ = [
    "CHANGESET_FORMAT",
    "CHANGESET_MEDIA_TYPE",
    "CHANGESET_SCHEMA_VERSION",
    "ChangeSetError",
    "ChangeSetIngestionResult",
    "ChangeSetIngestionService",
    "ChangeSetIntegrityResult",
    "IngestChangeSetCommand",
    "MAX_OPERATIONS",
    "PROTECTED_PATH_SEGMENTS",
    "SUPPORTED_OPERATIONS",
    "validate_changeset_path",
    "verify_change_set_integrity",
]

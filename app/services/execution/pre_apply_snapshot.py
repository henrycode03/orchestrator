"""Immutable pre-apply bytes and scope authority for Phase 29D-3A.

This boundary captures only the exact files selected by one authorized
Controlled Apply attempt.  It never mutates the workspace, invokes Git, or
reconstructs bytes from repository state.  Bytes are owned by the existing
content-addressed store; database rows contain bounded references and hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Sequence

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    ExecutionEvidence,
    ExecutionTaskApplyAttempt,
    ExecutionTaskApplyAuthorization,
    ExecutionTaskApplyApproval,
    ExecutionTaskChangeSet,
    ExecutionTaskChangeSetOperation,
    ExecutionTaskPreApplySnapshot,
    ExecutionTaskPreApplySnapshotEntry,
    ExecutionWorkspaceBaseState,
    ExecutionWorkspacePathObservation,
    ExecutionWorkspaceTarget,
)
from app.services.execution.candidate_content import (
    CandidateContentError,
    CandidateContentStore,
    LocalContentAddressedStore,
    MAX_CANDIDATE_CONTENT_BYTES,
    _store_key,
)
from app.services.execution.changeset import (
    ChangeSetError,
    validate_changeset_path,
    verify_change_set_integrity,
)
from app.services.execution.workspace_authority import (
    verify_workspace_base_state_integrity,
    verify_workspace_target_integrity,
)
from app.services.planning.operator_review import canonical_json_hash


PRE_APPLY_SNAPSHOT_SCHEMA_VERSION = "execution-task-pre-apply-snapshot/1.0"
PRE_APPLY_CONTENT_REFERENCE_SCHEME = "pre-apply-content-sha256"
PRE_APPLY_SNAPSHOT_STATUSES = frozenset({"captured", "failed"})
MAX_PRE_APPLY_SNAPSHOT_BYTES = MAX_CANDIDATE_CONTENT_BYTES
_HASH_RE = r"[0-9a-f]{64}"


class PreApplySnapshotError(RuntimeError):
    """Bounded snapshot-authority failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class PreApplySnapshotOperation:
    operation: str
    canonical_path: str
    previous_exists: bool
    previous_entry_type: str
    previous_sha256: str | None
    previous_byte_length: int | None
    expected_post_apply_exists: bool
    expected_post_apply_sha256: str | None

    def request_payload(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "canonical_path": self.canonical_path,
            "previous_exists": bool(self.previous_exists),
            "previous_entry_type": self.previous_entry_type,
            "previous_sha256": self.previous_sha256,
            "previous_byte_length": self.previous_byte_length,
            "expected_post_apply_exists": bool(self.expected_post_apply_exists),
            "expected_post_apply_sha256": self.expected_post_apply_sha256,
        }


@dataclass(frozen=True)
class CapturePreApplySnapshotCommand:
    apply_attempt_id: int
    final_precondition_verification_payload: dict[str, Any]
    final_precondition_verification_hash: str
    operations: tuple[PreApplySnapshotOperation, ...]
    owner: str = "controlled-apply"


@dataclass(frozen=True)
class PreApplySnapshotCaptureResult:
    snapshot: ExecutionTaskPreApplySnapshot
    replayed: bool = False


@dataclass(frozen=True)
class PreApplySnapshotIntegrityResult:
    authority_id: int | None
    verified: bool
    issues: tuple[str, ...] = ()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _short_detail(value: object, limit: int = 1024) -> str:
    return str(value)[:limit]


def _content_reference(content_sha256: str) -> str:
    return f"{PRE_APPLY_CONTENT_REFERENCE_SCHEME}://{content_sha256}"


def _reference_hash(reference: str | None) -> str | None:
    prefix = f"{PRE_APPLY_CONTENT_REFERENCE_SCHEME}://"
    if not isinstance(reference, str) or not reference.startswith(prefix):
        return None
    value = reference[len(prefix) :]
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        return None
    return value


def _read_snapshot_bytes(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise PreApplySnapshotError(
            "snapshot_path_missing", "previous file disappeared during snapshot"
        ) from exc
    except OSError as exc:
        raise PreApplySnapshotError(
            "snapshot_path_unreadable", "previous file could not be opened read-only"
        ) from exc
    chunks: list[bytes] = []
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PreApplySnapshotError(
                "snapshot_special_file", "previous entry is not a regular file"
            )
        if before.st_size > max_bytes:
            raise PreApplySnapshotError(
                "snapshot_file_too_large", "previous file exceeds snapshot bound"
            )
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise PreApplySnapshotError(
                    "snapshot_file_too_large", "previous file exceeds snapshot bound"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            total,
        ):
            raise PreApplySnapshotError(
                "snapshot_path_changed", "previous file changed during snapshot"
            )
    except OSError as exc:
        raise PreApplySnapshotError(
            "snapshot_path_unreadable", "previous file could not be read"
        ) from exc
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _assert_absent(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PreApplySnapshotError(
            "snapshot_path_unreadable", "create target could not be inspected"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PreApplySnapshotError(
            "snapshot_symlink_rejected", "create target is a symlink"
        )
    raise PreApplySnapshotError(
        "snapshot_path_changed", "create target is no longer absent"
    )


def _verify_parent_directories(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise PreApplySnapshotError(
            "snapshot_operation_path_invalid", "snapshot path escapes workspace root"
        ) from exc
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise PreApplySnapshotError(
                "snapshot_parent_unreadable",
                "snapshot parent directory is unavailable",
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PreApplySnapshotError(
                "snapshot_parent_invalid",
                "snapshot parent is not a real directory",
            )


def _scope_payload(attempt: ExecutionTaskApplyAttempt) -> dict[str, Any]:
    return {
        "execution_plan_id": attempt.execution_plan_id,
        "execution_task_id": attempt.execution_task_id,
        "execution_task_attempt_id": attempt.execution_task_attempt_id,
        "attempt_generation": attempt.attempt_generation,
        "apply_attempt_id": attempt.id,
        "apply_attempt_hash": attempt.canonical_command_hash,
        "change_set_id": attempt.change_set_id,
        "change_set_hash": attempt.change_set_hash,
        "authorization_id": attempt.authorization_id,
        "authorization_hash": attempt.authorization_hash,
        "approval_id": attempt.approval_id,
        "approval_hash": attempt.approval_hash,
        "workspace_target_id": attempt.workspace_target_id,
        "workspace_target_hash": attempt.workspace_target_hash,
        "base_state_id": attempt.base_state_id,
        "base_state_hash": attempt.base_state_hash,
    }


def _entry_payload(
    *,
    operation: PreApplySnapshotOperation,
    previous_content_reference: str | None,
    previous_storage_key: str | None,
) -> dict[str, Any]:
    return {
        **operation.request_payload(),
        "previous_content_reference": previous_content_reference,
        "previous_storage_key": previous_storage_key,
    }


class PreApplySnapshotService:
    """Capture and verify one immutable snapshot for one apply attempt."""

    def __init__(
        self, db: Session, *, store: CandidateContentStore | None = None, now=None
    ):
        self.db = db
        self.store = store or LocalContentAddressedStore()
        self._now = now or _utc_now

    def capture(
        self, command: CapturePreApplySnapshotCommand
    ) -> PreApplySnapshotCaptureResult:
        attempt = self.db.get(ExecutionTaskApplyAttempt, int(command.apply_attempt_id))
        if attempt is None:
            raise PreApplySnapshotError(
                "snapshot_apply_attempt_missing", "apply attempt does not exist"
            )
        request_payload = {
            "schema_version": PRE_APPLY_SNAPSHOT_SCHEMA_VERSION,
            "scope": _scope_payload(attempt),
            "final_precondition_verification": command.final_precondition_verification_payload,
            "final_precondition_verification_hash": command.final_precondition_verification_hash,
            "operations": [item.request_payload() for item in command.operations],
        }
        capture_hash = canonical_json_hash(request_payload)
        existing = (
            self.db.query(ExecutionTaskPreApplySnapshot)
            .filter(ExecutionTaskPreApplySnapshot.apply_attempt_id == attempt.id)
            .one_or_none()
        )
        if existing is not None:
            if existing.capture_command_hash != capture_hash:
                raise PreApplySnapshotError(
                    "snapshot_idempotency_conflict",
                    "apply attempt is bound to a different snapshot request",
                )
            return PreApplySnapshotCaptureResult(existing, replayed=True)

        self._verify_capture_scope(attempt, command)
        root = Path(attempt.workspace_target.normalized_realpath)
        captured_payloads: list[dict[str, Any]] = []
        stored_keys: list[str] = []
        failure_reason: str | None = None
        failure_detail: str | None = None
        for item in command.operations:
            try:
                reference = None
                storage_key = None
                path = root / item.canonical_path
                _verify_parent_directories(root, path)
                if item.previous_exists:
                    if item.previous_entry_type != "regular_file":
                        raise PreApplySnapshotError(
                            "snapshot_special_file",
                            "previous entry type is not a regular file",
                        )
                    data = _read_snapshot_bytes(
                        path, max_bytes=MAX_PRE_APPLY_SNAPSHOT_BYTES
                    )
                    digest = hashlib.sha256(data).hexdigest()
                    if digest != item.previous_sha256:
                        raise PreApplySnapshotError(
                            "snapshot_previous_hash_mismatch",
                            "captured previous bytes do not match final verification",
                        )
                    if len(data) != item.previous_byte_length:
                        raise PreApplySnapshotError(
                            "snapshot_previous_size_mismatch",
                            "captured previous length does not match final verification",
                        )
                    stored = self.store.put(data)
                    if stored.content_sha256 != digest or stored.byte_length != len(
                        data
                    ):
                        raise PreApplySnapshotError(
                            "snapshot_content_integrity_failure",
                            "content store returned a mismatched byte authority",
                        )
                    reference = _content_reference(digest)
                    storage_key = stored.storage_key
                    stored_keys.append(storage_key)
                else:
                    _assert_absent(path)
                captured_payloads.append(
                    _entry_payload(
                        operation=item,
                        previous_content_reference=reference,
                        previous_storage_key=storage_key,
                    )
                )
            except (PreApplySnapshotError, CandidateContentError) as exc:
                failure_reason = (
                    exc.code
                    if isinstance(exc, PreApplySnapshotError)
                    else "snapshot_content_store_failure"
                )[:64]
                failure_detail = _short_detail(exc)
                break

        status = "captured" if failure_reason is None else "failed"
        payload = {
            **request_payload,
            "status": status,
            "failure_reason": failure_reason,
            "failure_detail": failure_detail,
            "entries": captured_payloads,
        }
        row = ExecutionTaskPreApplySnapshot(
            **_scope_payload(attempt),
            final_precondition_verification_hash=command.final_precondition_verification_hash,
            capture_command_hash=capture_hash,
            status=status,
            failure_reason=failure_reason,
            failure_detail=failure_detail,
            expected_entry_count=len(command.operations),
            captured_entry_count=len(captured_payloads),
            canonical_payload=payload,
            canonical_sha256=canonical_json_hash(payload),
            snapshot_idempotency_key=f"pre-apply-snapshot:{attempt.id}",
            created_at=self._now(),
        )
        try:
            with self.db.begin_nested():
                self.db.add(row)
                self.db.flush()
                for index, entry_payload in enumerate(captured_payloads):
                    self.db.add(
                        ExecutionTaskPreApplySnapshotEntry(
                            snapshot_id=row.id,
                            entry_index=index,
                            operation=entry_payload["operation"],
                            canonical_path=entry_payload["canonical_path"],
                            previous_exists=entry_payload["previous_exists"],
                            previous_entry_type=entry_payload["previous_entry_type"],
                            previous_sha256=entry_payload["previous_sha256"],
                            previous_byte_length=entry_payload["previous_byte_length"],
                            previous_content_reference=entry_payload[
                                "previous_content_reference"
                            ],
                            previous_storage_key=entry_payload["previous_storage_key"],
                            expected_post_apply_exists=entry_payload[
                                "expected_post_apply_exists"
                            ],
                            expected_post_apply_sha256=entry_payload[
                                "expected_post_apply_sha256"
                            ],
                            canonical_entry_payload=entry_payload,
                            canonical_entry_hash=canonical_json_hash(entry_payload),
                        )
                    )
                self.db.flush()
        except IntegrityError as exc:
            replay = (
                self.db.query(ExecutionTaskPreApplySnapshot)
                .filter(ExecutionTaskPreApplySnapshot.apply_attempt_id == attempt.id)
                .one_or_none()
            )
            if replay is not None and replay.capture_command_hash == capture_hash:
                return PreApplySnapshotCaptureResult(replay, replayed=True)
            self._cleanup_unlinked(stored_keys)
            raise PreApplySnapshotError(
                "snapshot_insert_conflict",
                "snapshot conflicts with canonical authority",
            ) from exc
        self._cleanup_unlinked(stored_keys)
        return PreApplySnapshotCaptureResult(row)

    def _verify_capture_scope(
        self,
        attempt: ExecutionTaskApplyAttempt,
        command: CapturePreApplySnapshotCommand,
    ) -> None:
        authorization = self.db.get(
            ExecutionTaskApplyAuthorization, attempt.authorization_id
        )
        approval = self.db.get(ExecutionTaskApplyApproval, attempt.approval_id)
        change_set = self.db.get(ExecutionTaskChangeSet, attempt.change_set_id)
        target = self.db.get(ExecutionWorkspaceTarget, attempt.workspace_target_id)
        base_state = self.db.get(ExecutionWorkspaceBaseState, attempt.base_state_id)
        if any(
            item is None
            for item in (authorization, approval, change_set, target, base_state)
        ):
            raise PreApplySnapshotError(
                "snapshot_scope_missing", "snapshot scope authority is incomplete"
            )
        assert authorization is not None
        assert approval is not None
        assert change_set is not None
        assert target is not None
        assert base_state is not None
        if (
            attempt.change_set_hash != change_set.changeset_sha256
            or attempt.authorization_hash != authorization.canonical_decision_hash
            or attempt.approval_hash != approval.canonical_approval_hash
            or attempt.workspace_target_hash != target.canonical_target_hash
            or attempt.base_state_hash != base_state.canonical_observation_hash
        ):
            raise PreApplySnapshotError(
                "snapshot_scope_integrity_failure",
                "apply scope hash does not match immutable authorities",
            )
        if not verify_change_set_integrity(
            self.db, change_set.id, store=self.store
        ).verified:
            raise PreApplySnapshotError(
                "snapshot_changeset_integrity_failure",
                "ChangeSet integrity verification failed",
            )
        if not verify_workspace_target_integrity(self.db, target.id).verified:
            raise PreApplySnapshotError(
                "snapshot_target_integrity_failure",
                "workspace target integrity verification failed",
            )
        if not verify_workspace_base_state_integrity(self.db, base_state.id).verified:
            raise PreApplySnapshotError(
                "snapshot_base_state_integrity_failure",
                "workspace base-state integrity verification failed",
            )
        if (
            canonical_json_hash(command.final_precondition_verification_payload)
            != command.final_precondition_verification_hash
        ):
            raise PreApplySnapshotError(
                "snapshot_final_verification_hash_mismatch",
                "final precondition verification hash is invalid",
            )
        rows = (
            self.db.query(ExecutionTaskChangeSetOperation)
            .filter(ExecutionTaskChangeSetOperation.change_set_id == change_set.id)
            .order_by(ExecutionTaskChangeSetOperation.operation_index)
            .all()
        )
        if len(rows) != len(command.operations) or not rows:
            raise PreApplySnapshotError(
                "snapshot_operation_scope_mismatch",
                "snapshot operation count does not match ChangeSet",
            )
        for row, item in zip(rows, command.operations):
            if (
                row.operation != item.operation
                or row.canonical_path != item.canonical_path
                or row.content_sha256 != item.expected_post_apply_sha256
                or (row.operation == "delete_file")
                != (not item.expected_post_apply_exists)
            ):
                raise PreApplySnapshotError(
                    "snapshot_operation_scope_mismatch",
                    "snapshot operation does not match ChangeSet",
                )
            try:
                if validate_changeset_path(item.canonical_path) != item.canonical_path:
                    raise PreApplySnapshotError(
                        "snapshot_operation_path_invalid",
                        "snapshot path is not canonical",
                    )
            except ChangeSetError as exc:
                raise PreApplySnapshotError(
                    "snapshot_operation_path_invalid", exc.message
                ) from exc

    def _cleanup_unlinked(self, storage_keys: Sequence[str]) -> None:
        linked = {
            row.previous_storage_key
            for row in self.db.query(
                ExecutionTaskPreApplySnapshotEntry.previous_storage_key
            )
            .filter(
                ExecutionTaskPreApplySnapshotEntry.previous_storage_key.is_not(None)
            )
            .all()
        }
        linked |= {
            row.storage_key
            for row in self.db.query(ExecutionEvidence.storage_key).all()
        }
        from app.models import ExecutionTaskCandidateContent

        linked |= {
            row.storage_key
            for row in self.db.query(ExecutionTaskCandidateContent.storage_key).all()
        }
        for key in storage_keys:
            if key not in linked:
                self.store.delete_if_unreferenced(key)


def verify_pre_apply_snapshot_integrity(
    db: Session,
    snapshot_id: int,
    *,
    store: CandidateContentStore | None = None,
) -> PreApplySnapshotIntegrityResult:
    row = db.get(ExecutionTaskPreApplySnapshot, int(snapshot_id))
    if row is None:
        return PreApplySnapshotIntegrityResult(None, False, ("snapshot_missing",))
    reader = store or LocalContentAddressedStore()
    issues: list[str] = []
    change_set = None
    base_state = None
    try:
        if canonical_json_hash(row.canonical_payload) != row.canonical_sha256:
            issues.append("snapshot_canonical_hash_mismatch")
        payload = row.canonical_payload
        if payload.get("status") != row.status:
            issues.append("snapshot_status_mismatch")
        if payload.get("failure_reason") != row.failure_reason:
            issues.append("snapshot_failure_reason_mismatch")
        final_payload = payload.get("final_precondition_verification")
        if (
            canonical_json_hash(final_payload)
            != row.final_precondition_verification_hash
        ):
            issues.append("snapshot_final_verification_hash_mismatch")
        capture_payload = {
            key: payload.get(key)
            for key in (
                "schema_version",
                "scope",
                "final_precondition_verification",
                "final_precondition_verification_hash",
                "operations",
            )
        }
        if canonical_json_hash(capture_payload) != row.capture_command_hash:
            issues.append("snapshot_capture_command_hash_mismatch")
        attempt = db.get(ExecutionTaskApplyAttempt, row.apply_attempt_id)
        if attempt is None:
            issues.append("snapshot_apply_attempt_missing")
        else:
            expected_scope = _scope_payload(attempt)
            if payload.get("scope") != expected_scope:
                issues.append("snapshot_scope_payload_mismatch")
            for field, expected in expected_scope.items():
                if field == "apply_attempt_id":
                    actual = row.apply_attempt_id
                elif hasattr(row, field):
                    actual = getattr(row, field)
                else:
                    continue
                if actual != expected:
                    issues.append(f"snapshot_{field}_mismatch")
            change_set = db.get(ExecutionTaskChangeSet, row.change_set_id)
            base_state = db.get(ExecutionWorkspaceBaseState, row.base_state_id)
            if change_set is None:
                issues.append("snapshot_changeset_missing")
            if base_state is None:
                issues.append("snapshot_base_state_missing")
        entries = (
            db.query(ExecutionTaskPreApplySnapshotEntry)
            .filter(ExecutionTaskPreApplySnapshotEntry.snapshot_id == row.id)
            .order_by(ExecutionTaskPreApplySnapshotEntry.entry_index)
            .all()
        )
        if len(entries) != row.captured_entry_count:
            issues.append("snapshot_entry_count_mismatch")
        payload_entries = payload.get("entries")
        if payload_entries != [entry.canonical_entry_payload for entry in entries]:
            issues.append("snapshot_entry_payload_mismatch")
        final_operations = (
            final_payload.get("operations", [])
            if isinstance(final_payload, dict)
            else []
        )
        if row.status == "captured" and final_operations != [
            {
                key: entry.canonical_entry_payload.get(key)
                for key in (
                    "operation",
                    "canonical_path",
                    "previous_exists",
                    "previous_entry_type",
                    "previous_sha256",
                    "previous_byte_length",
                    "expected_post_apply_exists",
                    "expected_post_apply_sha256",
                )
            }
            for entry in entries
        ]:
            issues.append("snapshot_final_operation_payload_mismatch")
        if (
            row.status == "captured"
            and row.captured_entry_count != row.expected_entry_count
        ):
            issues.append("snapshot_incomplete_captured_status")
            if row.status == "failed" and not row.failure_reason:
                issues.append("snapshot_failed_reason_missing")
        change_set_operations = (
            db.query(ExecutionTaskChangeSetOperation)
            .filter(ExecutionTaskChangeSetOperation.change_set_id == row.change_set_id)
            .order_by(ExecutionTaskChangeSetOperation.operation_index)
            .all()
            if change_set is not None
            else []
        )
        base_observations = (
            {
                item.path: item
                for item in db.query(ExecutionWorkspacePathObservation)
                .filter(
                    ExecutionWorkspacePathObservation.base_state_id == row.base_state_id
                )
                .all()
            }
            if base_state is not None
            else {}
        )
        for index, entry in enumerate(entries):
            if entry.entry_index != index:
                issues.append(f"snapshot_entry_sequence_mismatch:{entry.id}")
            if (
                canonical_json_hash(entry.canonical_entry_payload)
                != entry.canonical_entry_hash
            ):
                issues.append(f"snapshot_entry_hash_mismatch:{entry.id}")
            if (
                entry.canonical_entry_payload.get("canonical_path")
                != entry.canonical_path
            ):
                issues.append(f"snapshot_entry_path_payload_mismatch:{entry.id}")
            if not isinstance(final_payload, dict):
                issues.append(f"snapshot_final_payload_malformed:{entry.id}")
            if entry.previous_entry_type not in {"absent", "regular_file"}:
                issues.append(f"snapshot_entry_type_invalid:{entry.id}")
            if entry.previous_exists:
                reference_hash = _reference_hash(entry.previous_content_reference)
                if reference_hash != entry.previous_sha256:
                    issues.append(f"snapshot_entry_reference_mismatch:{entry.id}")
                expected_key = _store_key(str(entry.previous_sha256))
                if entry.previous_storage_key != expected_key:
                    issues.append(f"snapshot_entry_storage_key_mismatch:{entry.id}")
                try:
                    data = reader.read(entry.previous_storage_key)
                    if len(data) != entry.previous_byte_length:
                        issues.append(f"snapshot_entry_size_mismatch:{entry.id}")
                    if hashlib.sha256(data).hexdigest() != entry.previous_sha256:
                        issues.append(
                            f"snapshot_entry_content_hash_mismatch:{entry.id}"
                        )
                except (CandidateContentError, TypeError, ValueError):
                    issues.append(f"snapshot_entry_content_unavailable:{entry.id}")
            elif any(
                value is not None
                for value in (
                    entry.previous_sha256,
                    entry.previous_byte_length,
                    entry.previous_content_reference,
                    entry.previous_storage_key,
                )
            ):
                issues.append(f"snapshot_entry_absent_shape_mismatch:{entry.id}")
            expected_post = entry.operation != "delete_file"
            if entry.expected_post_apply_exists != expected_post:
                issues.append(f"snapshot_entry_post_state_mismatch:{entry.id}")
            if (
                entry.operation == "delete_file"
                and entry.expected_post_apply_sha256 is not None
            ):
                issues.append(f"snapshot_entry_delete_post_hash:{entry.id}")
            if index < len(change_set_operations):
                operation = change_set_operations[index]
                if (
                    operation.operation != entry.operation
                    or operation.canonical_path != entry.canonical_path
                    or operation.content_sha256 != entry.expected_post_apply_sha256
                    or (operation.operation == "delete_file")
                    != (not entry.expected_post_apply_exists)
                ):
                    issues.append(f"snapshot_entry_changeset_mismatch:{entry.id}")
            elif row.status == "captured":
                issues.append(f"snapshot_entry_changeset_missing:{entry.id}")
            observation = base_observations.get(entry.canonical_path)
            if observation is not None:
                expected_entry_type = (
                    "missing"
                    if entry.previous_entry_type == "absent"
                    else entry.previous_entry_type
                )
                if (
                    observation.operation != entry.operation
                    or observation.exists != entry.previous_exists
                    or observation.entry_type != expected_entry_type
                    or observation.content_sha256 != entry.previous_sha256
                    or observation.byte_length != entry.previous_byte_length
                ):
                    issues.append(f"snapshot_entry_base_state_mismatch:{entry.id}")
            elif row.status == "captured":
                issues.append(f"snapshot_entry_base_state_missing:{entry.id}")
    except (AttributeError, TypeError, ValueError, KeyError):
        issues.append("snapshot_payload_malformed")
    return PreApplySnapshotIntegrityResult(
        row.id, not issues, tuple(sorted(set(issues)))
    )


__all__ = [
    "CapturePreApplySnapshotCommand",
    "MAX_PRE_APPLY_SNAPSHOT_BYTES",
    "PRE_APPLY_CONTENT_REFERENCE_SCHEME",
    "PRE_APPLY_SNAPSHOT_SCHEMA_VERSION",
    "PreApplySnapshotCaptureResult",
    "PreApplySnapshotError",
    "PreApplySnapshotIntegrityResult",
    "PreApplySnapshotOperation",
    "PreApplySnapshotService",
    "verify_pre_apply_snapshot_integrity",
]

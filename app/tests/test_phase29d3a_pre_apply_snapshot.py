"""Focused Phase 29D-3A immutable pre-apply snapshot coverage."""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import create_engine, inspect, text

from app.db_migrations import _migration_049_pre_apply_snapshot_authority
from app.models import (
    Base,
    ExecutionTaskApplyResult,
    ExecutionTaskPreApplySnapshot,
    ExecutionTaskPreApplySnapshotEntry,
)
from app.services.execution.apply_execution import (
    ApplyExecutionService,
    ExecuteApplyCommand,
    verify_apply_result_integrity,
)
from app.services.execution.candidate_content import (
    CandidateContentError,
    cleanup_unlinked_candidate_content,
)
from app.services.execution.pre_apply_snapshot import (
    verify_pre_apply_snapshot_integrity,
)
from app.services.planning.operator_review import canonical_json_hash

from test_phase29d3_controlled_apply import _apply_authority


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_snapshot_captures_create_replace_delete_previous_state(
    db_session, tmp_path, monkeypatch
):
    old = b"old bytes\n"
    removed = b"deleted bytes\n"
    context = _apply_authority(
        db_session,
        tmp_path,
        monkeypatch,
        initial_files={"src/replace.txt": old, "src/delete.txt": removed},
        operations=[
            {
                "operation": "create_file",
                "path": "src/create.txt",
                "content_reference": "placeholder",
                "content_key": "snapshot-create",
                "content": b"created\n",
            },
            {
                "operation": "replace_file",
                "path": "src/replace.txt",
                "expected_previous_sha256": _sha(old),
                "content_reference": "placeholder",
                "content_key": "snapshot-replace",
                "content": b"replaced\n",
            },
            {
                "operation": "delete_file",
                "path": "src/delete.txt",
                "expected_previous_sha256": _sha(removed),
            },
        ],
    )

    result = (
        ApplyExecutionService(db_session, store=context["store"])
        .execute(ExecuteApplyCommand(context["attempt"].id))
        .result
    )
    db_session.commit()

    snapshot = db_session.query(ExecutionTaskPreApplySnapshot).one()
    entries = (
        db_session.query(ExecutionTaskPreApplySnapshotEntry)
        .filter(ExecutionTaskPreApplySnapshotEntry.snapshot_id == snapshot.id)
        .order_by(ExecutionTaskPreApplySnapshotEntry.entry_index)
        .all()
    )
    assert result.status == "applied"
    assert snapshot.status == "captured"
    assert snapshot.apply_attempt_id == context["attempt"].id
    assert result.pre_apply_snapshot_id == snapshot.id
    assert len(entries) == 3
    assert entries[0].previous_exists is False
    assert entries[0].previous_content_reference is None
    assert entries[1].previous_exists is True
    assert entries[1].previous_sha256 == _sha(old)
    assert entries[2].previous_sha256 == _sha(removed)
    assert context["store"].read(entries[1].previous_storage_key) == old
    assert context["store"].read(entries[2].previous_storage_key) == removed
    assert verify_pre_apply_snapshot_integrity(
        db_session, snapshot.id, store=context["store"]
    ).verified
    assert verify_apply_result_integrity(
        db_session, result.id, store=context["store"]
    ).verified


def test_snapshot_is_durable_before_mutation(db_session, tmp_path, monkeypatch):
    old = b"before mutation\n"
    context = _apply_authority(
        db_session,
        tmp_path,
        monkeypatch,
        initial_files={"src/file.txt": old},
        operations=[
            {
                "operation": "replace_file",
                "path": "src/file.txt",
                "expected_previous_sha256": _sha(old),
                "content_reference": "placeholder",
                "content_key": "snapshot-order",
                "content": b"after mutation\n",
            }
        ],
    )
    observed = {"called": False}
    real_apply = ApplyExecutionService._apply_atomically

    def assert_snapshot_before_mutation(service, operations):
        snapshot = db_session.query(ExecutionTaskPreApplySnapshot).one()
        assert snapshot.status == "captured"
        assert verify_pre_apply_snapshot_integrity(
            db_session, snapshot.id, store=context["store"]
        ).verified
        assert (context["root"] / "src/file.txt").read_bytes() == old
        observed["called"] = True
        return real_apply(service, operations)

    monkeypatch.setattr(
        ApplyExecutionService, "_apply_atomically", assert_snapshot_before_mutation
    )
    result = (
        ApplyExecutionService(db_session, store=context["store"])
        .execute(ExecuteApplyCommand(context["attempt"].id))
        .result
    )
    assert observed["called"] is True
    assert result.status == "applied"


def test_snapshot_failure_blocks_without_mutation_and_preserves_failed_authority(
    db_session, tmp_path, monkeypatch
):
    old = b"must remain\n"
    context = _apply_authority(
        db_session,
        tmp_path,
        monkeypatch,
        initial_files={"src/file.txt": old},
        operations=[
            {
                "operation": "replace_file",
                "path": "src/file.txt",
                "expected_previous_sha256": _sha(old),
                "content_reference": "placeholder",
                "content_key": "snapshot-failure",
                "content": b"new\n",
            }
        ],
    )

    def fail_snapshot_put(_content):
        raise CandidateContentError(
            "injected_snapshot_store_failure", "injected snapshot store failure"
        )

    monkeypatch.setattr(context["store"], "put", fail_snapshot_put)
    result = (
        ApplyExecutionService(db_session, store=context["store"])
        .execute(ExecuteApplyCommand(context["attempt"].id))
        .result
    )
    db_session.commit()

    snapshot = db_session.query(ExecutionTaskPreApplySnapshot).one()
    assert result.status == "blocked"
    assert result.failure_reason == "pre_apply_snapshot_failed"
    assert snapshot.status == "failed"
    assert snapshot.failure_reason == "snapshot_content_store_failure"
    assert (context["root"] / "src/file.txt").read_bytes() == old


def test_snapshot_replay_and_tamper_detection(db_session, tmp_path, monkeypatch):
    context = _apply_authority(
        db_session,
        tmp_path,
        monkeypatch,
        operations=[
            {
                "operation": "create_file",
                "path": "src/file.txt",
                "content_reference": "placeholder",
                "content_key": "snapshot-replay",
                "content": b"content\n",
            }
        ],
    )
    service = ApplyExecutionService(db_session, store=context["store"])
    first = service.execute(ExecuteApplyCommand(context["attempt"].id)).result
    db_session.commit()
    replay = service.execute(ExecuteApplyCommand(context["attempt"].id))
    assert replay.replayed is True
    assert replay.result.id == first.id
    assert db_session.query(ExecutionTaskPreApplySnapshot).count() == 1

    entry = db_session.query(ExecutionTaskPreApplySnapshotEntry).one()
    db_session.execute(
        text(
            "UPDATE execution_task_pre_apply_snapshot_entries "
            "SET canonical_entry_payload = :payload WHERE id = :id"
        ),
        {"payload": json.dumps({"tampered": True}), "id": entry.id},
    )
    db_session.commit()
    snapshot = db_session.query(ExecutionTaskPreApplySnapshot).one()
    assert not verify_pre_apply_snapshot_integrity(
        db_session, snapshot.id, store=context["store"]
    ).verified


def test_snapshot_blob_retention_and_cleanup(db_session, tmp_path, monkeypatch):
    old = b"retain this exact blob"
    context = _apply_authority(
        db_session,
        tmp_path,
        monkeypatch,
        initial_files={"src/file.txt": old},
        operations=[
            {
                "operation": "replace_file",
                "path": "src/file.txt",
                "expected_previous_sha256": _sha(old),
                "content_reference": "placeholder",
                "content_key": "snapshot-retention",
                "content": b"new\n",
            }
        ],
    )
    result = (
        ApplyExecutionService(db_session, store=context["store"])
        .execute(ExecuteApplyCommand(context["attempt"].id))
        .result
    )
    db_session.commit()
    entry = db_session.query(ExecutionTaskPreApplySnapshotEntry).one()
    key = entry.previous_storage_key
    assert key in set(context["store"].list_storage_keys())
    assert key not in set(
        cleanup_unlinked_candidate_content(db_session, store=context["store"])
    )

    db_session.execute(
        text("DELETE FROM execution_task_apply_results WHERE id = :id"),
        {"id": result.id},
    )
    db_session.execute(
        text(
            "DELETE FROM execution_task_pre_apply_snapshot_entries WHERE snapshot_id = :id"
        ),
        {"id": entry.snapshot_id},
    )
    db_session.execute(
        text("DELETE FROM execution_task_pre_apply_snapshots WHERE id = :id"),
        {"id": entry.snapshot_id},
    )
    db_session.commit()
    assert key in set(
        cleanup_unlinked_candidate_content(db_session, store=context["store"])
    )
    assert key not in set(context["store"].list_storage_keys())


def test_snapshot_migration_is_additive_and_replay_safe(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    Base.metadata.create_all(engine)
    _migration_049_pre_apply_snapshot_authority(engine)
    _migration_049_pre_apply_snapshot_authority(engine)
    tables = set(inspect(engine).get_table_names())
    assert "execution_task_pre_apply_snapshots" in tables
    assert "execution_task_pre_apply_snapshot_entries" in tables
    columns = {
        column["name"]
        for column in inspect(engine).get_columns("execution_task_apply_results")
    }
    assert {"pre_apply_snapshot_id", "pre_apply_snapshot_hash"} <= columns


def test_historical_apply_result_without_snapshot_remains_compatible(
    db_session, tmp_path, monkeypatch
):
    context = _apply_authority(
        db_session,
        tmp_path,
        monkeypatch,
        operations=[
            {
                "operation": "create_file",
                "path": "src/file.txt",
                "content_reference": "placeholder",
                "content_key": "snapshot-historical",
                "content": b"content\n",
            }
        ],
    )
    result = (
        ApplyExecutionService(db_session, store=context["store"])
        .execute(ExecuteApplyCommand(context["attempt"].id))
        .result
    )
    db_session.commit()
    payload = dict(result.canonical_payload)
    payload.pop("pre_apply_snapshot_id", None)
    payload.pop("pre_apply_snapshot_hash", None)
    db_session.execute(
        text(
            "UPDATE execution_task_apply_results SET "
            "pre_apply_snapshot_id = NULL, pre_apply_snapshot_hash = NULL, "
            "canonical_payload = :payload, canonical_sha256 = :digest "
            "WHERE id = :id"
        ),
        {
            "payload": json.dumps(payload),
            "digest": canonical_json_hash(payload),
            "id": result.id,
        },
    )
    db_session.commit()
    historical = db_session.get(ExecutionTaskApplyResult, result.id)
    assert historical.pre_apply_snapshot_id is None
    assert verify_apply_result_integrity(
        db_session, result.id, store=context["store"]
    ).verified

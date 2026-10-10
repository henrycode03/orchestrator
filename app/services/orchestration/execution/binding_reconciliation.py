"""Crash-residue reconciliation for ephemeral OpenClaw bindings (Phase 37 Pre-B-F2B).

Normal release lives in ``ExecutorWorkspaceBinding.release``. This module only
handles what a release could not: residue left by SIGKILL, container/host
restarts, or an interrupted cleanup.

Liveness authority is the binding's exclusive ``flock``: the kernel drops it
when every holder (the worker and any inherited OpenClaw child) has exited,
even on SIGKILL. PID, age and directory-name prefix are never sufficient
evidence on their own.

Report-only by default. ``reconcile_binding_artifacts(apply=True)`` removes
only ``STALE_CONFIRMED`` artifacts inside the approved root, and only after
revalidating them while holding their lock. Legacy ``/tmp`` directories
created before this contract are inventoried, never removed.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import shutil
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.services.orchestration.execution.executor_workspace_binding import (
    BINDING_ARTIFACT_TYPE,
    BINDING_DIR_PREFIX,
    BINDING_LOCK_NAME,
    BINDING_METADATA_NAME,
    BINDING_METADATA_SCHEMA,
    ExecutorWorkspaceBindingError,
    binding_artifact_root,
    read_boot_id,
    read_pid_namespace,
    read_process_start_ticks,
    validate_binding_artifact_root,
)

logger = logging.getLogger(__name__)

ACTIVE = "ACTIVE"
STALE_CONFIRMED = "STALE_CONFIRMED"
STALE_SUSPECTED = "STALE_SUSPECTED"
UNKNOWN_OWNERSHIP = "UNKNOWN_OWNERSHIP"
INVALID_METADATA = "INVALID_METADATA"
LEGACY_UNMANAGED = "LEGACY_UNMANAGED"
UNSAFE_PATH = "UNSAFE_PATH"

REPORTED = "REPORTED"
REMOVED = "REMOVED"
REVALIDATION_FAILED = "REVALIDATION_FAILED"
REMOVAL_FAILED = "REMOVAL_FAILED"

_REQUIRED_METADATA_KEYS = (
    "schema",
    "artifact_type",
    "binding_id",
    "artifact_dir_name",
    "created_at_epoch",
    "owner_uid",
)


@dataclass
class BindingArtifactReport:
    path: str
    classification: str
    reason: str
    action: str = REPORTED
    binding_id: Optional[str] = None
    age_seconds: Optional[float] = None
    credential_bearing: bool = False
    owner_pid: Optional[int] = None
    task_execution_id: Optional[Any] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _open_nofollow(path: Path, flags: int) -> int:
    return os.open(path, flags | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))


def _read_metadata(directory: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (metadata, problem). Never follows symlinks."""

    try:
        fd = _open_nofollow(directory / BINDING_METADATA_NAME, os.O_RDONLY)
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"unreadable:{errno.errorcode.get(exc.errno, exc.errno)}"
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None, "not_regular_file"
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            raw = handle.read(64 * 1024)
    finally:
        if fd >= 0:
            os.close(fd)
    try:
        metadata = json.loads(raw)
    except ValueError:
        return None, "malformed_json"
    if not isinstance(metadata, dict):
        return None, "not_an_object"
    missing = [key for key in _REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        return None, f"missing_keys:{','.join(missing)}"
    if metadata.get("schema") != BINDING_METADATA_SCHEMA:
        return None, "unknown_schema"
    if metadata.get("artifact_type") != BINDING_ARTIFACT_TYPE:
        return None, "unknown_artifact_type"
    if metadata.get("artifact_dir_name") != directory.name:
        # Metadata copied/moved into a different directory is not ownership.
        return None, "artifact_dir_name_mismatch"
    if metadata.get("owner_uid") != os.getuid():
        return None, "owner_uid_mismatch"
    return metadata, ""


def _try_lock(directory: Path) -> Tuple[Optional[int], str]:
    """Try to take the binding lock. Returns (fd, problem); fd None if held."""

    try:
        fd = _open_nofollow(directory / BINDING_LOCK_NAME, os.O_RDWR)
    except FileNotFoundError:
        return None, "lock_missing"
    except OSError as exc:
        return None, f"lock_unopenable:{errno.errorcode.get(exc.errno, exc.errno)}"
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None, "lock_not_regular_file"
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None, "lock_held"
    except OSError as exc:
        os.close(fd)
        return None, f"lock_error:{errno.errorcode.get(exc.errno, exc.errno)}"
    return fd, ""


def _credential_bearing(directory: Path) -> bool:
    return os.path.lexists(directory / "agent" / "auth-profiles.json")


def _owner_process_still_matches(metadata: Dict[str, Any]) -> bool:
    """True only if the recorded owner provably still exists in this namespace.

    Used to downgrade (never to upgrade) a lock-free artifact: a live owner
    whose lock is free means release is mid-failure, so stay conservative.
    PID equality alone is never treated as proof -- boot id, PID namespace
    and process start time must all match.
    """

    pid = metadata.get("owner_pid")
    if not isinstance(pid, int):
        return False
    if metadata.get("boot_id") is None or metadata.get("boot_id") != read_boot_id():
        return False
    namespace = metadata.get("owner_pid_namespace")
    if namespace is None or namespace != read_pid_namespace():
        return False
    recorded_start = metadata.get("owner_pid_start_ticks")
    return recorded_start is not None and recorded_start == read_process_start_ticks(
        pid
    )


def _classify_entry(
    entry: Path, *, now: float, keep_lock: bool = False
) -> Tuple[BindingArtifactReport, Optional[int], Optional[os.stat_result]]:
    """Classify one root entry. Returns (report, held_lock_fd, dir_stat)."""

    report = BindingArtifactReport(
        path=str(entry), classification=UNKNOWN_OWNERSHIP, reason=""
    )
    try:
        entry_stat = os.lstat(entry)
    except FileNotFoundError:
        report.reason = "vanished"
        return report, None, None
    if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISDIR(entry_stat.st_mode):
        report.classification = UNSAFE_PATH
        report.reason = "not_a_real_directory"
        return report, None, None
    report.age_seconds = round(now - entry_stat.st_mtime, 1)
    report.credential_bearing = _credential_bearing(entry)
    if not entry.name.startswith(BINDING_DIR_PREFIX):
        report.reason = "unexpected_name"
        return report, None, entry_stat
    if entry_stat.st_uid != os.getuid():
        report.reason = "foreign_uid"
        return report, None, entry_stat

    metadata, problem = _read_metadata(entry)
    if metadata is None:
        if problem == "missing":
            lock_fd, lock_problem = _try_lock(entry)
            if lock_fd is not None:
                os.close(lock_fd)
                # Creator died between lock and metadata, or a cleanup was
                # interrupted after removing metadata. Not provable either way.
                report.classification = STALE_SUSPECTED
                report.reason = "metadata_missing_lock_free"
            elif lock_problem == "lock_held":
                report.classification = ACTIVE
                report.reason = "metadata_missing_lock_held_initializing"
            else:
                report.classification = UNKNOWN_OWNERSHIP
                report.reason = f"metadata_missing_{lock_problem}"
            return report, None, entry_stat
        report.classification = INVALID_METADATA
        report.reason = problem
        return report, None, entry_stat

    report.binding_id = str(metadata.get("binding_id"))
    report.owner_pid = metadata.get("owner_pid")
    report.task_execution_id = metadata.get("task_execution_id")
    report.details = {
        "owner_hostname": metadata.get("owner_hostname"),
        "boot_id_matches": metadata.get("boot_id") == read_boot_id(),
        "pid_namespace_matches": (
            metadata.get("owner_pid_namespace") == read_pid_namespace()
        ),
    }
    lock_fd, lock_problem = _try_lock(entry)
    if lock_fd is None:
        if lock_problem == "lock_held":
            report.classification = ACTIVE
            report.reason = "binding_lock_held"
        else:
            report.classification = INVALID_METADATA
            report.reason = lock_problem
        return report, None, entry_stat
    if _owner_process_still_matches(metadata):
        os.close(lock_fd)
        report.classification = STALE_SUSPECTED
        report.reason = "lock_free_but_owner_process_alive"
        return report, None, entry_stat
    report.classification = STALE_CONFIRMED
    report.reason = "binding_lock_free_owner_gone"
    if keep_lock:
        return report, lock_fd, entry_stat
    os.close(lock_fd)
    return report, None, entry_stat


def _remove_locked_artifact(entry: Path) -> List[Tuple[str, str]]:
    """Remove a revalidated artifact while its lock is held.

    Order keeps an interruption reconcilable: payload first, then metadata,
    then the lock file, then the directory itself.
    """

    failures: List[Tuple[str, str]] = []
    for child in sorted(os.listdir(entry)):
        if child in (BINDING_METADATA_NAME, BINDING_LOCK_NAME):
            continue
        child_path = entry / child
        try:
            if os.path.islink(child_path) or not os.path.isdir(child_path):
                os.unlink(child_path)
            else:
                shutil.rmtree(
                    child_path,
                    onerror=lambda _fn, path, exc_info: failures.append(
                        (path, exc_info[0].__name__)
                    ),
                )
        except OSError as exc:
            failures.append((str(child_path), type(exc).__name__))
    if failures:
        return failures
    for name in (BINDING_METADATA_NAME, BINDING_LOCK_NAME):
        try:
            os.unlink(entry / name)
        except FileNotFoundError:
            pass
        except OSError as exc:
            failures.append((name, type(exc).__name__))
            return failures
    try:
        os.rmdir(entry)
    except OSError as exc:
        failures.append((str(entry), type(exc).__name__))
    return failures


def _audit(report: BindingArtifactReport) -> None:
    logger.info(
        "[BINDING_RECONCILIATION] %s",
        json.dumps(
            {
                "path": report.path,
                "binding_id": report.binding_id,
                "classification": report.classification,
                "reason": report.reason,
                "action": report.action,
                "credential_bearing": report.credential_bearing,
            },
            sort_keys=True,
        ),
    )


def reconcile_binding_artifacts(
    root: Optional[Path] = None, *, apply: bool = False
) -> Dict[str, Any]:
    """Classify managed binding artifacts; remove STALE_CONFIRMED only if ``apply``.

    Concurrency-safe across processes: removal holds the artifact's lock, so
    a live owner (or a second reconciler) can never race the deletion.
    """

    approved_root = Path(root) if root is not None else binding_artifact_root()
    payload: Dict[str, Any] = {
        "root": str(approved_root),
        "apply": apply,
        "artifacts": [],
        "root_status": "ok",
    }
    if not os.path.lexists(approved_root):
        payload["root_status"] = "absent"
        return payload
    try:
        validate_binding_artifact_root(approved_root)
    except ExecutorWorkspaceBindingError as exc:
        payload["root_status"] = f"refused: {exc}"
        return payload

    now = time.time()
    resolved_root = approved_root.resolve()
    for name in sorted(os.listdir(approved_root)):
        entry = approved_root / name
        report, lock_fd, first_stat = _classify_entry(entry, now=now, keep_lock=apply)
        if lock_fd is not None:
            try:
                report.action = _revalidate_and_remove(
                    entry, resolved_root, first_stat, report
                )
            finally:
                os.close(lock_fd)
        _audit(report)
        payload["artifacts"].append(report.to_dict())
    payload["summary"] = _summarize(payload["artifacts"])
    return payload


def _revalidate_and_remove(
    entry: Path,
    resolved_root: Path,
    first_stat: Optional[os.stat_result],
    report: BindingArtifactReport,
) -> str:
    try:
        current = os.lstat(entry)
    except FileNotFoundError:
        report.reason = "vanished_before_removal"
        return REVALIDATION_FAILED
    if (
        first_stat is None
        or stat.S_ISLNK(current.st_mode)
        or (current.st_dev, current.st_ino) != (first_stat.st_dev, first_stat.st_ino)
        or entry.resolve().parent != resolved_root
    ):
        report.reason = "path_changed_before_removal"
        return REVALIDATION_FAILED
    metadata, problem = _read_metadata(entry)
    if metadata is None or str(metadata.get("binding_id")) != report.binding_id:
        report.reason = f"metadata_changed_before_removal:{problem or 'binding_id'}"
        return REVALIDATION_FAILED
    failures = _remove_locked_artifact(entry)
    if failures:
        report.details["removal_failures"] = failures[:10]
        return REMOVAL_FAILED
    return REMOVED


def inventory_legacy_tmp_bindings(tmp_root: Optional[Path] = None) -> Dict[str, Any]:
    """Read-only inventory of pre-F2B ``/tmp`` binding directories.

    Never removes anything and never reads file contents.
    """

    base = Path(tmp_root) if tmp_root is not None else Path(tempfile.gettempdir())
    now = time.time()
    artifacts = []
    for entry in sorted(base.glob(BINDING_DIR_PREFIX + "*")):
        report = BindingArtifactReport(
            path=str(entry),
            classification=LEGACY_UNMANAGED,
            reason="created_outside_managed_root",
        )
        try:
            entry_stat = os.lstat(entry)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISDIR(entry_stat.st_mode):
            report.classification = UNSAFE_PATH
            report.reason = "not_a_real_directory"
        else:
            report.age_seconds = round(now - entry_stat.st_mtime, 1)
            report.credential_bearing = _credential_bearing(entry)
            report.details = {
                "mode": oct(entry_stat.st_mode & 0o777),
                "uid": entry_stat.st_uid,
                "has_ownership_metadata": os.path.lexists(
                    entry / BINDING_METADATA_NAME
                ),
            }
        artifacts.append(report.to_dict())
    return {
        "tmp_root": str(base),
        "artifacts": artifacts,
        "summary": _summarize(artifacts),
    }


def _summarize(artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_class: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    for item in artifacts:
        by_class[item["classification"]] = by_class.get(item["classification"], 0) + 1
        by_action[item["action"]] = by_action.get(item["action"], 0) + 1
    return {
        "total": len(artifacts),
        "credential_bearing": sum(
            1 for item in artifacts if item["credential_bearing"]
        ),
        "by_classification": by_class,
        "by_action": by_action,
    }

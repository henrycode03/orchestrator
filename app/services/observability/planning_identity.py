"""Creation-time identity snapshots for planning and execution records."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.orm import Session

from app.services.observability.build_identity import build_identity_payload


def _fingerprint(payload: dict[str, Any], reasoning_profile: str | None) -> str:
    """Hash stable, non-secret build/config identity fields only."""
    fingerprint_payload = {
        key: payload.get(key)
        for key in (
            "version",
            "git_sha",
            "build_git_sha",
            "repo_git_sha",
            "build_time",
            "image_tag",
            "image_id",
            "runtime_profile",
            "active_backend_lanes",
            "active_model_names",
            "config_source",
        )
    }
    fingerprint_payload["reasoning_profile"] = reasoning_profile
    encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def active_planning_identity(db: Session) -> dict[str, str | None]:
    """Snapshot the active planning lane without retaining configuration secrets."""
    from app.services.agents.agent_runtime import resolve_planning_runtime_configuration

    configuration = resolve_planning_runtime_configuration(db)
    payload = build_identity_payload(db, planning_configuration=configuration)
    reasoning_profile = configuration.adaptation_profile
    return {
        "planning_backend": payload["planning_backend"],
        "planner_model": payload["planner_model"],
        "reasoning_profile": reasoning_profile,
        "configuration_fingerprint": _fingerprint(payload, reasoning_profile),
    }


def active_execution_identity(db: Session) -> dict[str, str | None]:
    """Snapshot both role lanes at task-execution creation time."""
    from app.services.agents.agent_runtime import (
        BackendRole,
        resolve_runtime_configuration,
    )

    configuration = resolve_runtime_configuration(db, BackendRole.EXECUTION)
    payload = build_identity_payload(db)
    execution_profile = configuration.adaptation_profile
    fingerprint_payload = {
        **payload,
        "active_backend_lanes": {"execution": configuration.backend_name},
        "active_model_names": {"execution": configuration.model_family},
    }
    return {
        "planning_backend": payload["planning_backend"],
        "execution_backend": configuration.backend_name,
        "planner_model": payload["planner_model"],
        "executor_model": configuration.model_family,
        "execution_adaptation_profile": execution_profile,
        "configuration_fingerprint": _fingerprint(
            fingerprint_payload, execution_profile
        ),
    }

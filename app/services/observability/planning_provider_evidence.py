"""Bounded evidence for one Planning provider invocation.

This module deliberately owns only planning-call evidence.  The append-only
orchestration event journal remains the lifecycle index; the per-attempt files
hold the bounded prompt/response material needed to interpret that index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from app.services.orchestration.events.event_types import EventType
from app.services.workspace.control_state_paths import (
    control_state_root,
    project_control_state_location,
)
from app.services.workspace.permissions import ensure_shared_permissions
from app.services.workspace.system_settings import get_effective_runtime_root

logger = logging.getLogger(__name__)

PLANNING_EVIDENCE_DIRECTORY = "planning-evidence"
MAX_RETAINED_PROMPT_CHARS = 100_000
MAX_RETAINED_VISIBLE_CHARS = 100_000
MAX_RETAINED_REASONING_CHARS = 20_000
MAX_RETAINED_PARTIAL_CHARS = 8_000


def append_orchestration_event(**kwargs: Any) -> dict[str, Any]:
    """Lazy bridge to the existing event journal, avoiding import cycles."""

    from app.services.orchestration.state.persistence import (
        append_orchestration_event as append_event,
    )

    return append_event(**kwargs)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _token_estimate(text: str) -> int:
    return (len(text) + 3) // 4


def safe_endpoint_class(value: Any) -> str | None:
    """Return an endpoint class with credentials and query data removed."""

    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return f"{parsed.scheme}://{host}{parsed.path}".rstrip("/")
    return raw[:255]


def _text_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item.get("text", "") for item in value if isinstance(item, Mapping)]
        if all(isinstance(part, str) for part in parts):
            return "".join(parts)
    return None


def _normalized_usage(value: Any) -> dict[str, int | None] | None:
    if not isinstance(value, Mapping):
        return None
    prompt = value.get("prompt_tokens", value.get("input_tokens"))
    completion = value.get("completion_tokens", value.get("output_tokens"))
    total = value.get("total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    return {
        "prompt_tokens": prompt if isinstance(prompt, int) else None,
        "completion_tokens": completion if isinstance(completion, int) else None,
        "total_tokens": total if isinstance(total, int) else None,
    }


def inspect_chat_completion_response(body: Any) -> dict[str, Any]:
    """Extract only normal OpenAI-compatible response fields for evidence."""

    result: dict[str, Any] = {
        "response_received": True,
        "visible_content": None,
        "reasoning_content": None,
        "usage": (
            _normalized_usage(body.get("usage")) if isinstance(body, Mapping) else None
        ),
        "finish_reason": None,
        "provider_request_correlation_id": (
            str(body.get("id"))[:255]
            if isinstance(body, Mapping) and body.get("id") is not None
            else None
        ),
    }
    choices = body.get("choices") if isinstance(body, Mapping) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, Mapping) else None
    if isinstance(message, Mapping):
        result["visible_content"] = _text_value(message.get("content"))
        result["reasoning_content"] = _text_value(
            message.get("reasoning_content") or message.get("reasoning")
        )
    if isinstance(first, Mapping) and first.get("finish_reason") is not None:
        result["finish_reason"] = str(first["finish_reason"])[:255]
    return result


def _bounded_text(value: Any, limit: int) -> tuple[str | None, bool]:
    text = _text_value(value)
    if text is None and value is not None:
        text = str(value)
    if text is None:
        return None, False
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _runtime_identity(runtime_service: Any) -> dict[str, Any]:
    task = getattr(runtime_service, "task_model", None)
    project_id = getattr(runtime_service, "project_id", None)
    if project_id is None:
        project_id = getattr(task, "project_id", None)
    return {
        "db": getattr(runtime_service, "db", None),
        "project_id": project_id,
        "session_id": getattr(runtime_service, "session_id", None),
        "task_id": getattr(runtime_service, "task_id", None),
        "task_execution_id": getattr(runtime_service, "task_execution_id", None),
    }


def _task_attempt_number(db: Any, task_execution_id: Any) -> int | None:
    if db is None or task_execution_id is None:
        return None
    try:
        from app.models import TaskExecution

        execution = (
            db.query(TaskExecution)
            .filter(TaskExecution.id == task_execution_id)
            .first()
        )
        value = getattr(execution, "attempt_number", None)
        return int(value) if value is not None else None
    except Exception:
        return None


def _resolve_control_state_location(
    *, db: Any, project_id: int | None, control_state_location: Any
) -> Any:
    if control_state_location is not None:
        return control_state_location
    if project_id is None:
        return None
    return project_control_state_location(
        get_effective_runtime_root(db), project_id, db=db
    )


def begin_planning_provider_evidence(
    *,
    control_state_location: Any = None,
    db: Any = None,
    project_id: int | None,
    task_id: int | None,
    session_id: int | None,
    task_execution_id: int | None,
    attempt: int | None,
    model: str | None,
    provider_endpoint_class: str | None,
    effective_timeout_seconds: float | int,
    transport_timeout_seconds: float | int,
    prompt: str,
    invocation_kind: str,
    provider_api_streaming: bool | None,
    partial_response_available_in_current_nonstreaming_path: bool | None = None,
) -> "PlanningProviderEvidence":
    """Persist request evidence and the provider-start event before I/O."""

    location = _resolve_control_state_location(
        db=db, project_id=project_id, control_state_location=control_state_location
    )
    recorder = PlanningProviderEvidence(
        control_state_location=location,
        project_id=project_id,
        task_id=task_id,
        session_id=session_id,
        task_execution_id=task_execution_id,
        attempt=attempt,
        model=model,
        provider_endpoint_class=safe_endpoint_class(provider_endpoint_class),
        effective_timeout_seconds=effective_timeout_seconds,
        transport_timeout_seconds=transport_timeout_seconds,
        prompt=prompt,
        invocation_kind=invocation_kind,
        provider_api_streaming=provider_api_streaming,
        partial_response_available_in_current_nonstreaming_path=(
            partial_response_available_in_current_nonstreaming_path
        ),
    )
    recorder.start()
    return recorder


def begin_planning_provider_evidence_from_runtime(
    runtime_service: Any,
    *,
    prompt: str,
    model: str | None,
    provider_endpoint_class: str | None,
    effective_timeout_seconds: float | int,
    transport_timeout_seconds: float | int,
    invocation_kind: str,
    provider_api_streaming: bool | None,
    partial_response_available_in_current_nonstreaming_path: bool | None = None,
) -> "PlanningProviderEvidence":
    identity = _runtime_identity(runtime_service)
    return begin_planning_provider_evidence(
        db=identity["db"],
        project_id=identity["project_id"],
        task_id=identity["task_id"],
        session_id=identity["session_id"],
        task_execution_id=identity["task_execution_id"],
        attempt=_task_attempt_number(identity["db"], identity["task_execution_id"]),
        model=model,
        provider_endpoint_class=provider_endpoint_class,
        effective_timeout_seconds=effective_timeout_seconds,
        transport_timeout_seconds=transport_timeout_seconds,
        prompt=prompt,
        invocation_kind=invocation_kind,
        provider_api_streaming=provider_api_streaming,
        partial_response_available_in_current_nonstreaming_path=(
            partial_response_available_in_current_nonstreaming_path
        ),
    )


@dataclass
class PlanningProviderEvidence:
    control_state_location: Any
    project_id: int | None
    task_id: int | None
    session_id: int | None
    task_execution_id: int | None
    attempt: int | None
    model: str | None
    provider_endpoint_class: str | None
    effective_timeout_seconds: float | int
    transport_timeout_seconds: float | int
    prompt: str
    invocation_kind: str
    provider_api_streaming: bool | None
    partial_response_available_in_current_nonstreaming_path: bool | None
    attempt_id: str = field(default_factory=lambda: f"planning-{uuid.uuid4().hex}")
    provider_started_at: str = field(default_factory=_utc_now)
    _monotonic_started_at: float = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)
    _terminalized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        import time

        self._monotonic_started_at = time.monotonic()
        prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        self._metadata = {
            "schema": "planning_provider_evidence.v1",
            "status": "started",
            "attempt_id": self.attempt_id,
            "attempt": self.attempt,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "task_execution_id": self.task_execution_id,
            "model": self.model,
            "provider_endpoint_class": self.provider_endpoint_class,
            "invocation_kind": self.invocation_kind,
            "prompt_sha256": prompt_hash,
            "prompt_chars": len(self.prompt),
            "prompt_token_estimate": _token_estimate(self.prompt),
            "provider_started_at": self.provider_started_at,
            "effective_timeout_seconds": self.effective_timeout_seconds,
            "transport_timeout_seconds": self.transport_timeout_seconds,
            "provider_api_streaming": self.provider_api_streaming,
            "partial_response_available_in_current_nonstreaming_path": (
                self.partial_response_available_in_current_nonstreaming_path
            ),
            "prompt_reconstructable": len(self.prompt) <= MAX_RETAINED_PROMPT_CHARS,
            "response_received": False,
            "visible_chars": 0,
            "reasoning_chars": 0,
            "partial_content_chars": 0,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def artifact_directory(self) -> Path:
        if self.control_state_location is None:
            return Path(tempfile.gettempdir()) / "orchestrator-no-evidence"
        return (
            control_state_root(self.control_state_location)
            / PLANNING_EVIDENCE_DIRECTORY
            / self.attempt_id
        )

    def _artifact_reference(self) -> str:
        return str(self.artifact_directory / "evidence.json")

    def _event_details(self, *, terminal: bool = False) -> dict[str, Any]:
        details = {
            key: value
            for key, value in self._metadata.items()
            if key
            not in {
                "schema",
                "status",
                "response_received",
                "visible_chars",
                "reasoning_chars",
                "partial_content_chars",
            }
        }
        details["evidence_artifact"] = self._artifact_reference()
        if terminal:
            details.update(
                {
                    "elapsed_seconds": self._metadata.get("elapsed_seconds"),
                    "response_received": self._metadata.get("response_received"),
                    "visible_chars": self._metadata.get("visible_chars", 0),
                    "reasoning_chars": self._metadata.get("reasoning_chars", 0),
                    "usage": self._metadata.get("usage"),
                    "finish_reason": self._metadata.get("finish_reason"),
                    "exception_type": self._metadata.get("exception_type"),
                }
            )
        return details

    def _append_event(self, event_type: str, *, terminal: bool = False) -> None:
        if self.control_state_location is None:
            return
        try:
            append_orchestration_event(
                project_dir=self.control_state_location,
                session_id=self.session_id or 0,
                task_id=self.task_id or 0,
                event_type=event_type,
                details=self._event_details(terminal=terminal),
            )
        except Exception as exc:
            logger.warning(
                "Planning provider evidence event could not be persisted: %s", exc
            )

    def start(self) -> None:
        if self.control_state_location is None:
            self._metadata["evidence_persistence"] = "unavailable_no_control_state"
            return
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        ensure_shared_permissions(self.artifact_directory)
        if len(self.prompt) <= MAX_RETAINED_PROMPT_CHARS:
            _write_text(self.artifact_directory / "planning-request.txt", self.prompt)
        self._metadata["evidence_artifact"] = self._artifact_reference()
        self._metadata["request_artifact"] = str(
            self.artifact_directory / "planning-request.txt"
        )
        _write_json(self.artifact_directory / "planning-response.json", {})
        _write_json(self.artifact_directory / "evidence.json", self._metadata)
        self._append_event(EventType.PLANNING_PROVIDER_STARTED)

    def _elapsed(self) -> float:
        import time

        return round(time.monotonic() - self._monotonic_started_at, 3)

    def _finish(
        self,
        *,
        status: str,
        response_received: bool,
        visible_content: Any = None,
        reasoning_content: Any = None,
        usage: Any = None,
        finish_reason: Any = None,
        provider_request_correlation_id: Any = None,
        partial_content_snapshot: Any = None,
        exception_type: str | None = None,
        partial_response_available: bool | None = None,
    ) -> dict[str, Any]:
        if self._terminalized:
            return self.metadata
        self._terminalized = True
        visible, visible_truncated = _bounded_text(
            visible_content, MAX_RETAINED_VISIBLE_CHARS
        )
        reasoning, reasoning_truncated = _bounded_text(
            reasoning_content, MAX_RETAINED_REASONING_CHARS
        )
        partial, partial_truncated = _bounded_text(
            partial_content_snapshot, MAX_RETAINED_PARTIAL_CHARS
        )
        ended_at = _utc_now()
        self._metadata.update(
            {
                "status": status,
                "provider_ended_at": ended_at,
                "response_timestamp": ended_at,
                "elapsed_seconds": self._elapsed(),
                "response_received": bool(response_received),
                "visible_chars": len(visible or ""),
                "reasoning_chars": len(reasoning or ""),
                "partial_content_chars": len(partial or ""),
                "visible_content_truncated": visible_truncated,
                "reasoning_content_truncated": reasoning_truncated,
                "partial_content_truncated": partial_truncated,
                "usage": _normalized_usage(usage),
                "finish_reason": (
                    str(finish_reason)[:255] if finish_reason is not None else None
                ),
                "provider_request_correlation_id": (
                    str(provider_request_correlation_id)[:255]
                    if provider_request_correlation_id is not None
                    else None
                ),
                "exception_type": exception_type,
            }
        )
        if partial_response_available is not None:
            self._metadata["partial_response_available"] = partial_response_available
        response_payload = {
            "response_received": bool(response_received),
            "visible_content": visible,
            "reasoning_content": reasoning,
            "usage": _normalized_usage(usage),
            "finish_reason": self._metadata["finish_reason"],
            "provider_request_correlation_id": self._metadata[
                "provider_request_correlation_id"
            ],
            "partial_content_snapshot": partial,
        }
        if self.control_state_location is None:
            return self.metadata
        _write_json(
            self.artifact_directory / "planning-response.json", response_payload
        )
        _write_json(self.artifact_directory / "evidence.json", self._metadata)
        self._append_event(
            (
                EventType.PLANNING_PROVIDER_COMPLETED
                if status == "completed"
                else EventType.PLANNING_PROVIDER_FAILED
            ),
            terminal=True,
        )
        return self.metadata

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        return self._finish(status="completed", response_received=True, **kwargs)

    def fail(
        self,
        exception: BaseException,
        *,
        response_received: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._finish(
            status="failed",
            response_received=response_received,
            exception_type=type(exception).__name__,
            **kwargs,
        )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_shared_permissions(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    ensure_shared_permissions(temporary)
    temporary.replace(path)
    ensure_shared_permissions(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


__all__ = [
    "MAX_RETAINED_PROMPT_CHARS",
    "PlanningProviderEvidence",
    "begin_planning_provider_evidence",
    "begin_planning_provider_evidence_from_runtime",
    "inspect_chat_completion_response",
    "safe_endpoint_class",
]

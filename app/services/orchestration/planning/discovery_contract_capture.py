"""Small artifact-first capture seam for one discovery contract evaluation."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


DISCOVERY_ACTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DiscoveryAction",
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action": {"const": "search_text"},
                "query": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["action", "query", "paths"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"const": "read_file"},
                "path": {"type": "string"},
            },
            "required": ["action", "path"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action": {"const": "stop"}},
            "required": ["action"],
            "additionalProperties": False,
        },
    ],
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _redact_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    secret_markers = ("authorization", "api-key", "apikey", "token", "secret")
    for key, value in headers.items():
        normalized_key = str(key).lower()
        redacted[str(key)] = (
            "<redacted>"
            if any(marker in normalized_key for marker in secret_markers)
            else str(value)
        )
    return redacted


class DiscoveryContractCapture:
    """Persist each discovery boundary as soon as that boundary is reached."""

    def __init__(self, path: str | Path, *, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.document: dict[str, Any] = {
            "schema_version": "phase34c5",
            "run_id": run_id or uuid4().hex,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "request": {},
            "response": {},
            "stages": {},
            "parser": {},
            "action": {},
        }

    @classmethod
    def load(cls, path: str | Path) -> "DiscoveryContractCapture":
        capture = cls(path)
        if capture.path.exists():
            capture.document = json.loads(capture.path.read_text(encoding="utf-8"))
        return capture

    @classmethod
    def from_metadata(cls, metadata: Any) -> "DiscoveryContractCapture | None":
        if not isinstance(metadata, Mapping):
            return None
        path = metadata.get("discovery_contract_capture_path")
        if not path:
            return None
        return cls(path, run_id=str(metadata.get("discovery_contract_run_id") or ""))

    def _persist(self) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o777)
        try:
            os.chmod(parent, 0o777)
        except OSError:
            pass
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        payload = json.dumps(
            _json_safe(self.document), ensure_ascii=True, separators=(",", ":")
        )
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self.path)
            try:
                os.chmod(self.path, 0o666)
            except OSError:
                pass
        finally:
            if temporary.exists():
                temporary.unlink()

    def record_request(
        self,
        *,
        endpoint: str,
        model: str,
        backend_role: str | None,
        low_resource_single_model: bool,
        diagnostic_label: str | None,
        user_prompt: str,
        system_message: str,
        body: Mapping[str, Any],
        headers: Mapping[str, Any],
        timeout_seconds: int | float,
        adaptation_profile: str | None,
    ) -> None:
        self.document["request"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "requested_model": model,
            "backend_role": backend_role,
            "low_resource_single_model": low_resource_single_model,
            "diagnostic_label": diagnostic_label,
            "canonical_user_prompt_sha256": hashlib.sha256(
                user_prompt.encode("utf-8")
            ).hexdigest(),
            "system_message": system_message,
            "user_message": user_prompt,
            "outbound_json_body": _json_safe(body),
            "response_format": _json_safe(body.get("response_format")),
            "json_schema": _json_safe(body.get("json_schema")),
            "grammar": _json_safe(body.get("grammar")),
            "http_headers": _redact_headers(headers),
            "timeout_seconds": timeout_seconds,
            "adaptation_profile": adaptation_profile,
        }
        self._persist()

    def record_http_response(
        self,
        *,
        status_code: int,
        raw_body: bytes,
        content_type: str | None,
    ) -> None:
        self.document["response"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "http_status": status_code,
            "content_type": content_type,
            "raw_body_text": raw_body.decode("utf-8", errors="replace"),
            "raw_body_bytes_base64": base64.b64encode(raw_body).decode("ascii"),
            "raw_body_sha256": hashlib.sha256(raw_body).hexdigest(),
            "raw_body_byte_length": len(raw_body),
            "raw_response_captured": True,
        }
        self._persist()

    def record_response_metadata(
        self, *, response_model: Any = None, finish_reason: Any = None
    ) -> None:
        self.document["response"].update(
            {
                "response_model": _json_safe(response_model),
                "finish_stop_reason": _json_safe(finish_reason),
            }
        )
        self._persist()

    def record_response_decode_failure(self, reason: str) -> None:
        self.document["response"]["json_decode_error"] = str(reason)[:500]
        self._persist()

    def record_message_content(self, value: Any) -> None:
        self.document["stages"]["message_content"] = _json_safe(value)
        self._persist()

    def record_extracted_content(self, value: Any) -> None:
        self.document["stages"]["extracted_content"] = _json_safe(value)
        self._persist()

    def record_normalized_content(self, value: Any) -> None:
        self.document["stages"]["normalized_content"] = _json_safe(value)
        self._persist()

    def record_parser_input(self, value: str) -> None:
        self.document["stages"]["parser_input"] = value
        self._persist()

    def record_parser_result(
        self,
        *,
        success: bool,
        reason: str | None = None,
        action: str | None = None,
    ) -> None:
        self.document["parser"] = {
            "success": bool(success),
            "reason": reason,
            "action": action,
        }
        self._persist()

    def record_action_result(
        self,
        *,
        validation_pass: bool,
        executable: bool,
        action: str,
        reason: str | None,
    ) -> None:
        self.document["action"] = {
            "validation_pass": bool(validation_pass),
            "executable": bool(executable),
            "action": action,
            "reason": reason,
        }
        self._persist()

    def record_error(self, *, layer: str, reason: str) -> None:
        self.document.setdefault("errors", []).append(
            {"layer": layer, "reason": str(reason)[:500]}
        )
        self._persist()

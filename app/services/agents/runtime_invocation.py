"""Provider-neutral, per-invocation runtime controls.

RoleRuntimeConfiguration owns durable role/backend/model/profile selection.
This module owns only transient controls needed to preserve an invocation
contract at the provider adapter boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping


_ALLOWED_EXTRA_PROVIDER_OPTIONS = frozenset(
    {
        "chat_template_kwargs",
        "enable_thinking",
        "num_ctx",
        "repeat_penalty",
        "think",
        "top_p",
    }
)
_SECRET_KEY_MARKERS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


@dataclass(frozen=True)
class RuntimeInvocationOptions:
    """Validated transient controls for one role-owned runtime invocation."""

    timeout_seconds: float | None = None
    no_output_timeout_seconds: float | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning_enabled: bool | None = None
    stream: bool | None = None
    system_prompt: str | None = None
    extra_provider_options: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for field_name in ("timeout_seconds", "no_output_timeout_seconds"):
            value = getattr(self, field_name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{field_name} must be positive when provided")
        if self.max_output_tokens is not None and int(self.max_output_tokens) <= 0:
            raise ValueError("max_output_tokens must be positive when provided")
        if self.temperature is not None and not -2.0 <= float(self.temperature) <= 2.0:
            raise ValueError("temperature must be between -2 and 2")
        if self.stream is True:
            raise ValueError("streaming invocation options are not supported")

        raw_options = dict(self.extra_provider_options or {})
        for key in raw_options:
            normalized_key = str(key).strip().lower()
            if normalized_key in _ALLOWED_EXTRA_PROVIDER_OPTIONS:
                continue
            if any(marker in normalized_key for marker in _SECRET_KEY_MARKERS):
                raise ValueError("secrets are not permitted in invocation options")
            raise ValueError(f"unsupported provider invocation option: {key!r}")
        if "chat_template_kwargs" in raw_options:
            template_options = raw_options["chat_template_kwargs"]
            if not isinstance(template_options, Mapping) or set(template_options) - {
                "enable_thinking"
            }:
                raise ValueError("chat_template_kwargs contains unsupported keys")
        object.__setattr__(
            self,
            "extra_provider_options",
            MappingProxyType(raw_options) if raw_options else None,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a secret-free diagnostic representation."""

        payload = asdict(self)
        if self.extra_provider_options is not None:
            payload["extra_provider_options"] = dict(self.extra_provider_options)
        return payload

    @property
    def uses_legacy_chat_shape(self) -> bool:
        """Whether the caller requested an exact provider chat contract."""

        return any(
            value is not None
            for value in (
                self.max_output_tokens,
                self.temperature,
                self.reasoning_enabled,
                self.stream,
                self.system_prompt,
                self.extra_provider_options,
            )
        )

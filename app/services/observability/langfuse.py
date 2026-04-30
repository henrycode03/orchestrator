"""Fail-open Langfuse tracing helpers for orchestration flows."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any, Iterator, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_MAX_PREVIEW_CHARS = 600


def langfuse_tracing_enabled() -> bool:
    """Return True when tracing is enabled and minimally configured."""

    return bool(
        settings.ORCHESTRATOR_LANGFUSE_ENABLED
        and str(settings.LANGFUSE_PUBLIC_KEY or "").strip()
        and str(settings.LANGFUSE_SECRET_KEY or "").strip()
    )


def build_text_trace_payload(
    value: Any,
    *,
    max_preview_chars: int = _MAX_PREVIEW_CHARS,
) -> Optional[dict[str, Any]]:
    """Build a compact, low-risk payload for Langfuse input/output fields."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    preview = text[:max_preview_chars]
    if len(text) > max_preview_chars:
        preview = preview.rstrip() + "..."
    return {
        "preview": preview,
        "chars": len(text),
        "lines": text.count("\n") + 1,
    }


@lru_cache(maxsize=1)
def _get_langfuse_client() -> Any:
    if not langfuse_tracing_enabled():
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning(
            "Langfuse tracing enabled but SDK is not installed. Install `langfuse` to emit traces."
        )
        return None

    try:
        return Langfuse(
            public_key=str(settings.LANGFUSE_PUBLIC_KEY or "").strip(),
            secret_key=str(settings.LANGFUSE_SECRET_KEY or "").strip(),
            base_url=str(settings.LANGFUSE_BASE_URL or "").strip() or None,
            tracing_enabled=True,
            environment=str(settings.LANGFUSE_ENVIRONMENT or "").strip() or None,
            release=settings.VERSION,
        )
    except Exception as exc:
        logger.warning(
            "Langfuse client initialization failed; tracing disabled: %s", exc
        )
        return None


def reset_langfuse_client_for_tests() -> None:
    """Clear cached client so tests can change settings safely."""

    _get_langfuse_client.cache_clear()


@contextmanager
def start_langfuse_observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any = None,
    output: Any = None,
    metadata: Optional[dict[str, Any]] = None,
    status_message: Optional[str] = None,
    model: Optional[str] = None,
    usage_details: Optional[dict[str, int]] = None,
) -> Iterator[Any]:
    """Start an observation when configured, otherwise yield None."""

    client = _get_langfuse_client()
    if client is None:
        with nullcontext(None) as observation:
            yield observation
        return

    try:
        observation_cm = client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            output=output,
            metadata=metadata,
            status_message=status_message,
            model=model,
            usage_details=usage_details,
            version=settings.VERSION,
        )
    except Exception as exc:
        logger.warning("Langfuse observation start failed for %s: %s", name, exc)
        with nullcontext(None) as observation:
            yield observation
        return

    with observation_cm as observation:
        yield observation


def update_langfuse_observation(observation: Any, **kwargs: Any) -> None:
    """Best-effort update for a Langfuse observation."""

    if observation is None:
        return
    payload = {key: value for key, value in kwargs.items() if value is not None}
    if not payload:
        return
    try:
        observation.update(**payload)
    except Exception as exc:
        logger.debug("Langfuse observation update failed: %s", exc)


def flush_langfuse() -> None:
    """Flush background trace buffers without raising into app logic."""

    client = _get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.debug("Langfuse flush failed: %s", exc)

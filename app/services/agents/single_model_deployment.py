"""Shared endpoint rules for the opt-in low-resource single-model mode."""

from __future__ import annotations

from app.config import settings


def low_resource_single_model_enabled() -> bool:
    """Return whether one direct runtime owns every generation role."""

    return bool(getattr(settings, "LOW_RESOURCE_SINGLE_MODEL", False))


def canonical_generation_base_url(backend_name: str) -> str:
    """Resolve the one provider endpoint used by low-resource generation."""

    backend = str(backend_name or "").strip()
    if backend == "direct_ollama":
        return str(getattr(settings, "OLLAMA_BASE_URL", "") or "").strip().rstrip("/")
    if backend == "openai_chat_completions":
        return (
            str(getattr(settings, "PLANNING_DIRECT_BASE_URL", "") or "")
            .strip()
            .rstrip("/")
        )
    return ""


def canonical_generation_chat_base_url(backend_name: str) -> str:
    """Return a base URL with the OpenAI-compatible ``/v1`` path."""

    base_url = canonical_generation_base_url(backend_name)
    if str(backend_name or "").strip() == "direct_ollama" and base_url:
        if not base_url.endswith("/v1"):
            return f"{base_url}/v1"
    return base_url


def canonical_generation_api_key(backend_name: str) -> str:
    """Return the low-resource runtime's one optional API key."""

    if str(backend_name or "").strip() == "openai_chat_completions":
        return str(getattr(settings, "PLANNING_DIRECT_API_KEY", "") or "").strip()
    return ""

"""Provider-neutral runtime configuration passed to backend adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeConfiguration:
    """Resolved role ownership for one runtime factory invocation."""

    role: str | None
    backend_name: str
    model_family: str | None = None
    adaptation_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

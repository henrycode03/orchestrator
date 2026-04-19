"""Shared orchestration types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationVerdict:
    """Deterministic validation result for plans, steps, or completion."""

    stage: str
    status: str
    profile: str
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    used_small_model: bool = False

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "profile": self.profile,
            "reasons": list(self.reasons),
            "details": dict(self.details),
            "used_small_model": self.used_small_model,
        }

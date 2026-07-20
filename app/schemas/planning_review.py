"""Strict public schemas for Protocol v2 operator review actions."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


Hash = Annotated[
    str,
    StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_UNSAFE_MARKUP_RE = re.compile(r"(?:<[^>]{1,256}>|javascript\s*:|data\s*:)", re.I)
_CREDENTIAL_SHAPED_RE = re.compile(
    r"(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|secret|bearer)"
    r"\s*[:=]\s*\S+",
    re.I,
)


def _safe_text(value: Any, *, name: str, limit: int, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    if len(text) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    if _CONTROL_RE.search(text):
        raise ValueError(f"{name} contains control characters")
    if _UNSAFE_MARKUP_RE.search(text):
        raise ValueError(f"{name} contains unsafe markup")
    if _CREDENTIAL_SHAPED_RE.search(text):
        raise ValueError(f"{name} contains credential-shaped content")
    return text


class ReviewSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewPredecessorBindingRequest(ReviewSchema):
    checkpoint_id: int = Field(gt=0)
    content_hash: Hash


class ReviewCandidateBindingRequest(ReviewSchema):
    planning_session_id: int = Field(gt=0)
    project_id: int = Field(gt=0)
    protocol_version: Literal["v2"]
    session_generation_id: str = Field(min_length=1, max_length=256)
    stage_name: str = Field(min_length=1, max_length=100)
    stage_version: int = Field(ge=1)
    stage_generation_id: str = Field(min_length=1, max_length=256)
    candidate_checkpoint_id: int = Field(gt=0)
    candidate_checkpoint_version: int = Field(ge=1)
    candidate_content_hash: Hash
    validation_hash: Hash
    validator_version: str = Field(min_length=1, max_length=128)
    input_manifest_id: str = Field(min_length=1, max_length=256)
    input_manifest_hash: Hash
    predecessors: tuple[ReviewPredecessorBindingRequest, ...] = ()
    accepted_brief_checkpoint_id: int | None = Field(default=None, gt=0)
    accepted_brief_hash: Hash | None = None
    stage_configuration_fingerprint: Hash
    candidate_attempt_id: str | None = Field(default=None, max_length=128)

    _text_fields = field_validator(
        "session_generation_id",
        "stage_name",
        "stage_generation_id",
        "validator_version",
        "input_manifest_id",
        "candidate_attempt_id",
        mode="before",
    )(lambda value, info: _safe_text(value, name=info.field_name, limit=256))


class ReviewActionRequest(ReviewSchema):
    candidate_binding: ReviewCandidateBindingRequest = Field(
        validation_alias=AliasChoices(
            "candidate_binding", "binding", "expected_binding"
        )
    )
    review_head_sequence: int = Field(
        ge=1,
        validation_alias=AliasChoices(
            "review_head_sequence",
            "expected_head_sequence",
            "expected_review_head_sequence",
        ),
    )
    review_head_token: Hash = Field(
        validation_alias=AliasChoices(
            "review_head_token", "expected_head_token", "review_concurrency_token"
        )
    )
    idempotency_key: str = Field(min_length=1, max_length=128)

    @property
    def binding(self) -> ReviewCandidateBindingRequest:
        """Compatibility accessor used by the endpoint conversion layer."""

        return self.candidate_binding

    _idempotency = field_validator("idempotency_key", mode="before")(
        lambda value: _safe_text(
            value, name="idempotency_key", limit=128, required=True
        )
    )


class ApproveReviewRequest(ReviewActionRequest):
    comment: str = Field(min_length=1, max_length=4096)

    _comment = field_validator("comment", mode="before")(
        lambda value: _safe_text(value, name="comment", limit=4096, required=True)
    )


class RejectReviewRequest(ReviewActionRequest):
    reason: str = Field(min_length=1, max_length=4096)

    _reason = field_validator("reason", mode="before")(
        lambda value: _safe_text(value, name="reason", limit=4096, required=True)
    )


class CancelReviewRequest(ReviewActionRequest):
    reason: str = Field(min_length=1, max_length=4096)

    _reason = field_validator("reason", mode="before")(
        lambda value: _safe_text(value, name="reason", limit=4096, required=True)
    )


class AcknowledgeReviewRequest(ReviewActionRequest):
    comment: str = Field(min_length=1, max_length=4096)

    _comment = field_validator("comment", mode="before")(
        lambda value: _safe_text(value, name="comment", limit=4096, required=True)
    )


class RegenerateReviewRequest(ReviewActionRequest):
    reason: str = Field(min_length=1, max_length=4096)
    guidance: str | None = Field(default=None, max_length=2048)

    _reason = field_validator("reason", mode="before")(
        lambda value: _safe_text(value, name="reason", limit=4096, required=True)
    )
    _guidance = field_validator("guidance", mode="before")(
        lambda value: (
            _safe_text(value, name="guidance", limit=2048)
            if value is not None
            else None
        )
    )


class AmendReviewRequest(ReviewActionRequest):
    target_kind: Literal[
        "planning_brief", "structured_task_plan", "brief_record", "task_record"
    ]
    base_checkpoint_id: int = Field(gt=0)
    base_checkpoint_hash: Hash
    requested_change_kinds: tuple[str, ...] = Field(min_length=1, max_length=8)
    target_record_references: tuple[str, ...] = Field(default=(), max_length=16)
    instruction: str = Field(min_length=1, max_length=1024)
    regeneration_guidance: str | None = Field(default=None, max_length=1024)
    reason: str = Field(min_length=1, max_length=2048)

    _change_kinds = field_validator("requested_change_kinds", mode="before")(
        lambda values: tuple(
            _safe_text(value, name="requested_change_kind", limit=128, required=True)
            for value in values
        )
    )
    _references = field_validator("target_record_references", mode="before")(
        lambda values: tuple(
            _safe_text(value, name="target_record_reference", limit=256, required=True)
            for value in values
        )
    )
    _instruction = field_validator("instruction", mode="before")(
        lambda value: _safe_text(value, name="instruction", limit=1024, required=True)
    )
    _reason = field_validator("reason", mode="before")(
        lambda value: _safe_text(value, name="reason", limit=2048, required=True)
    )
    _guidance = field_validator("regeneration_guidance", mode="before")(
        lambda value: (
            _safe_text(value, name="regeneration_guidance", limit=1024)
            if value is not None
            else None
        )
    )


class ReviewEventResponse(ReviewSchema):
    event_id: str
    event_type: str
    event_sequence: int
    decision: str | None = None
    operator_subject: str | None = None
    created_at: datetime | None = None


class ReviewActionResponse(ReviewSchema):
    review_id: str
    event_id: str
    decision: str
    review_state: str
    candidate_checkpoint_id: int
    candidate_content_hash: Hash
    promotion_checkpoint_id: int | None = None
    promotion_content_hash: Hash | None = None
    promotion_reason: str | None = None
    current_accepted_artifact: dict[str, Any] | None = None
    current_accepted_brief: dict[str, Any] | None = None
    current_accepted_task_plan: dict[str, Any] | None = None
    planning_lifecycle_state: str
    completion_reevaluation: dict[str, Any] | None = None
    regeneration: dict[str, Any] | None = None
    amendment: dict[str, Any] | None = None
    idempotent_replay: bool = False


class ReviewSummaryResponse(ReviewSchema):
    review_id: str
    stage_name: str
    stage_version: int
    candidate_checkpoint_id: int
    candidate_content_hash: Hash
    validation_hash: Hash
    review_state: str
    review_required_reasons: tuple[str, ...]
    current_event_sequence: int
    review_head_token: Hash
    allowed_decisions: tuple[str, ...]
    current_accepted_artifact: dict[str, Any] | None = None
    terminal_decision: str | None = None
    promotion_checkpoint_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    stale: bool = False
    integrity_status: str = "valid"


class ReviewListResponse(ReviewSchema):
    items: tuple[ReviewSummaryResponse, ...]
    next_cursor: str | None = None


class ReviewDetailResponse(ReviewSummaryResponse):
    artifact_authority: Literal["review_candidate"] = "review_candidate"
    candidate_binding: ReviewCandidateBindingRequest
    candidate_content: str | None = None
    validation_evidence: dict[str, Any]
    lineage: dict[str, Any]
    structural_diff: dict[str, Any] | None = None
    event_history: tuple[ReviewEventResponse, ...]
    rejection_reason: str | None = None
    cancellation_reason: str | None = None
    command_identity: str | None = None
    amendment_id: str | None = None
    amendment_hash: Hash | None = None
    completion_impact: dict[str, Any]


__all__ = [
    "AcknowledgeReviewRequest",
    "AmendReviewRequest",
    "ApproveReviewRequest",
    "CancelReviewRequest",
    "Hash",
    "RegenerateReviewRequest",
    "RejectReviewRequest",
    "ReviewActionResponse",
    "ReviewCandidateBindingRequest",
    "ReviewDetailResponse",
    "ReviewEventResponse",
    "ReviewListResponse",
    "ReviewPredecessorBindingRequest",
    "ReviewSummaryResponse",
]

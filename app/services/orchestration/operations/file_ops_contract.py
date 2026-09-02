"""Structured file operation contract shared across orchestration modules."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Set

FILE_OP_FIELD_SETS: Mapping[str, Set[str]] = {
    "mkdir": {"op", "path"},
    "delete_file": {"op", "path"},
    "write_file": {"op", "path", "content"},
    "append_file": {"op", "path", "content"},
    "replace_in_file": {"op", "path", "old", "new"},
}
SEMANTIC_REPLACE_IN_FILE_FIELD_SET = {
    "op",
    "path",
    "selector",
    "new",
}
SUPPORTED_FILE_OPS = frozenset(FILE_OP_FIELD_SETS)
CONTENT_FILE_OPS = frozenset({"write_file", "append_file"})
REPLACE_IN_FILE_OLD_ALIASES = (
    "old_text",
    "search",
    "match",
    "pattern",
    "target",
    "old_string",
    "old_str",
    "oldText",
)
REPLACE_IN_FILE_NEW_ALIASES = (
    "replace",
    "replacement",
    "content",
    "new_string",
    "new_str",
    "newText",
)


class ReplaceOperationMode(str, Enum):
    """The one canonical mode classification for replace operations."""

    LEGACY_REPLACE = "LEGACY_REPLACE"
    SEMANTIC_REPLACE = "SEMANTIC_REPLACE"
    OTHER = "OTHER"
    INVALID_MIXED_REPLACE = "INVALID_MIXED_REPLACE"


class SemanticReplaceProjectionError(ValueError):
    """A semantic replace cannot be represented in the canonical projection."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class SemanticReplaceIntent:
    """Immutable canonical intent for one semantic ``replace_in_file``."""

    path: str
    selector: Any
    new: str

    @classmethod
    def from_operation(cls, operation: Mapping[str, Any]) -> "SemanticReplaceIntent":
        normalized = normalize_replace_in_file_aliases(operation)
        if (
            classify_replace_operation(normalized)
            is not ReplaceOperationMode.SEMANTIC_REPLACE
        ):
            raise SemanticReplaceProjectionError(
                "not_semantic_replace", "operation is not a semantic replace"
            )
        if set(normalized) != SEMANTIC_REPLACE_IN_FILE_FIELD_SET:
            raise SemanticReplaceProjectionError(
                "semantic_replace_shape_invalid",
                "semantic replace must contain exactly op, path, selector, and new",
            )
        path = normalized.get("path")
        if not isinstance(path, str) or not path.strip():
            raise SemanticReplaceProjectionError(
                "semantic_replace_path_invalid", "path must be a non-empty string"
            )
        new = normalized.get("new")
        if not isinstance(new, str):
            raise SemanticReplaceProjectionError(
                "semantic_replace_new_invalid", "new must be a string"
            )
        try:
            from app.services.orchestration.operations.source_region_identity import (
                SourceRegionIdentity,
            )
            from app.services.orchestration.validation.path_authority import declare

            selector = SourceRegionIdentity.from_dict(normalized.get("selector"))
            canonical_path = declare(path)
        except Exception as exc:
            code = getattr(exc, "code", "semantic_replace_selector_invalid")
            raise SemanticReplaceProjectionError(code, str(exc)) from exc
        if selector.canonical_path != canonical_path:
            raise SemanticReplaceProjectionError(
                "selector_path_mismatch",
                "selector canonical_path must equal the operation path",
            )
        return cls(path=canonical_path.value, selector=selector, new=new)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": "replace_in_file",
            "path": self.path,
            "selector": self.selector.to_dict(),
            "new": self.new,
        }


def classify_replace_operation(operation: Any) -> ReplaceOperationMode:
    """Classify a replace operation without inferring semantic intent."""

    if not isinstance(operation, Mapping):
        return ReplaceOperationMode.OTHER
    op_name = str(operation.get("op") or "").strip()
    if op_name != "replace_in_file":
        return ReplaceOperationMode.OTHER
    normalized = normalize_replace_in_file_aliases(operation)
    has_selector = "selector" in normalized
    legacy_keys = {"old", *REPLACE_IN_FILE_OLD_ALIASES}
    has_legacy_anchor = bool(legacy_keys.intersection(normalized))
    if has_selector and has_legacy_anchor:
        return ReplaceOperationMode.INVALID_MIXED_REPLACE
    if has_selector:
        return ReplaceOperationMode.SEMANTIC_REPLACE
    return ReplaceOperationMode.LEGACY_REPLACE


def canonicalize_replace_operation(operation: Mapping[str, Any]) -> dict[str, Any]:
    """Project accepted semantic input to one stable operation dictionary.

    Invalid input remains representable for the Validator to reject with its
    normal Plan-contract evidence. No legacy anchor is synthesized.
    """

    normalized = normalize_replace_in_file_aliases(operation)
    if (
        classify_replace_operation(normalized)
        is not ReplaceOperationMode.SEMANTIC_REPLACE
    ):
        return normalized
    try:
        return SemanticReplaceIntent.from_operation(normalized).to_dict()
    except SemanticReplaceProjectionError:
        return normalized


def replace_mode_transitions(
    previous_plan: Any, candidate_plan: Any
) -> tuple[dict[str, Any], ...]:
    """Return replace-operation mode changes at stable step/operation positions."""

    def modes(plan: Any) -> dict[tuple[int, int], ReplaceOperationMode]:
        result: dict[tuple[int, int], ReplaceOperationMode] = {}
        if not isinstance(plan, list):
            return result
        for step_index, step in enumerate(plan, start=1):
            if not isinstance(step, Mapping):
                continue
            step_number = step.get("step_number", step_index)
            if isinstance(step_number, bool) or not isinstance(step_number, int):
                continue
            for operation_index, operation in enumerate(step.get("ops") or [], start=1):
                mode = classify_replace_operation(operation)
                if mode in {
                    ReplaceOperationMode.LEGACY_REPLACE,
                    ReplaceOperationMode.SEMANTIC_REPLACE,
                }:
                    result[(step_number, operation_index)] = mode
        return result

    previous = modes(previous_plan)
    candidate = modes(candidate_plan)
    transitions: list[dict[str, Any]] = []
    for identity in sorted(set(previous).intersection(candidate)):
        if previous[identity] is candidate[identity]:
            continue
        transitions.append(
            {
                "step_number": identity[0],
                "operation_index": identity[1],
                "from": previous[identity].value,
                "to": candidate[identity].value,
            }
        )
    return tuple(transitions)


@dataclass(frozen=True)
class ReplaceModePreservation:
    """Outcome of restoring already-valid replace operations after a repair."""

    plan: Any
    preserved: tuple[dict[str, Any], ...]
    unpreserved: tuple[dict[str, Any], ...]


def _replace_operations_by_identity(
    plan: Any,
) -> dict[tuple[int, int], tuple[int, int, Any]]:
    """Index plan operations by the stable (step_number, operation_index) key."""

    result: dict[tuple[int, int], tuple[int, int, Any]] = {}
    if not isinstance(plan, list):
        return result
    for step_index, step in enumerate(plan, start=1):
        if not isinstance(step, Mapping):
            continue
        step_number = step.get("step_number", step_index)
        if isinstance(step_number, bool) or not isinstance(step_number, int):
            continue
        for operation_index, operation in enumerate(step.get("ops") or [], start=1):
            result[(step_number, operation_index)] = (
                step_index - 1,
                operation_index - 1,
                operation,
            )
    return result


def preserve_replace_operation_modes(
    previous_plan: Any, candidate_plan: Any
) -> ReplaceModePreservation:
    """Restore already-valid semantic replaces a narrow repair downgraded.

    A narrow repair may only change the invalid portion of a Plan.  When the
    repaired Plan keeps an operation at the same step/operation position but
    downgrades an already-valid ``SEMANTIC_REPLACE`` to ``LEGACY_REPLACE`` on
    the same path, the original operation is restored verbatim.  No legacy
    anchor is synthesized and no other mode transition is preserved: every
    other transition is reported as ``unpreserved`` so the caller can keep
    failing closed.
    """

    transitions = replace_mode_transitions(previous_plan, candidate_plan)
    if not transitions:
        return ReplaceModePreservation(candidate_plan, (), ())

    previous_operations = _replace_operations_by_identity(previous_plan)
    candidate_operations = _replace_operations_by_identity(candidate_plan)
    preserved: list[dict[str, Any]] = []
    unpreserved: list[dict[str, Any]] = []
    restorations: list[tuple[int, int, dict[str, Any]]] = []

    for transition in transitions:
        identity = (transition["step_number"], transition["operation_index"])
        previous_entry = previous_operations.get(identity)
        candidate_entry = candidate_operations.get(identity)
        if (
            transition["from"] != ReplaceOperationMode.SEMANTIC_REPLACE.value
            or transition["to"] != ReplaceOperationMode.LEGACY_REPLACE.value
            or previous_entry is None
            or candidate_entry is None
        ):
            unpreserved.append(transition)
            continue
        try:
            intent = SemanticReplaceIntent.from_operation(previous_entry[2])
        except SemanticReplaceProjectionError:
            # The original operation was not already valid; the repair is
            # entitled to change it and this seam must not restore it.
            unpreserved.append(transition)
            continue
        if not _same_declared_path(intent.path, candidate_entry[2]):
            # The repair retargeted the operation; that is not a mode drift.
            unpreserved.append(transition)
            continue
        restorations.append((previous_entry[0], candidate_entry[1], intent.to_dict()))
        preserved.append(dict(transition))

    if not restorations:
        return ReplaceModePreservation(candidate_plan, (), tuple(unpreserved))

    restored_plan = copy.deepcopy(candidate_plan)
    for step_offset, operation_offset, operation in restorations:
        restored_plan[step_offset]["ops"][operation_offset] = operation
    return ReplaceModePreservation(restored_plan, tuple(preserved), tuple(unpreserved))


def _same_declared_path(canonical_path: str, candidate_operation: Any) -> bool:
    from app.services.orchestration.validation.path_authority import declare

    if not isinstance(candidate_operation, Mapping):
        return False
    candidate_path = normalize_replace_in_file_aliases(candidate_operation).get("path")
    if not isinstance(candidate_path, str) or not candidate_path.strip():
        return False
    try:
        return declare(candidate_path).value == canonical_path
    except Exception:
        return False


def is_supported_file_op_name(op_name: Any) -> bool:
    return str(op_name or "") in SUPPORTED_FILE_OPS


def operation_has_file_op_path(operation: Any) -> bool:
    return (
        isinstance(operation, dict)
        and is_supported_file_op_name(operation.get("op"))
        and bool(str(operation.get("path") or "").strip())
    )


def validate_file_op_shape(operation: Any) -> bool:
    if not isinstance(operation, dict):
        return False

    operation = normalize_file_op_shape(operation)
    op_name = str(operation.get("op") or "")
    expected_keys = FILE_OP_FIELD_SETS.get(op_name)
    if expected_keys is None or set(operation.keys()) != expected_keys:
        if op_name == "replace_in_file" and set(operation.keys()) == (
            SEMANTIC_REPLACE_IN_FILE_FIELD_SET
        ):
            return isinstance(operation.get("new"), str) and _valid_selector(
                operation.get("selector")
            )
        return False

    if not isinstance(operation.get("path"), str):
        return False
    if op_name in CONTENT_FILE_OPS:
        return isinstance(operation.get("content"), str)
    if op_name == "replace_in_file":
        return isinstance(operation.get("old"), str) and isinstance(
            operation.get("new"), str
        )
    return True


def _valid_selector(selector: Any) -> bool:
    from app.services.orchestration.operations.source_region_identity import (
        SourceRegionIdentity,
    )

    try:
        SourceRegionIdentity.from_dict(selector)
    except (TypeError, ValueError):
        return False
    return True


def expected_file_op_keys(op_name: str) -> Set[str]:
    return set(FILE_OP_FIELD_SETS[str(op_name)])


def normalize_file_op_shape(operation: Mapping[str, Any]) -> Dict[str, Any]:
    if "op" not in operation and len(operation) == 1:
        wrapped_op_name, wrapped_payload = next(iter(operation.items()))
        if is_supported_file_op_name(wrapped_op_name) and isinstance(
            wrapped_payload, Mapping
        ):
            operation = {"op": wrapped_op_name, **dict(wrapped_payload)}

    op_name = str(operation.get("op") or "")
    if op_name == "replace_in_file":
        return canonicalize_replace_operation(operation)

    expected_keys = FILE_OP_FIELD_SETS.get(op_name)
    if expected_keys is None:
        return dict(operation)
    return {key: operation[key] for key in expected_keys if key in operation}


def normalize_replace_in_file_aliases(operation: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce common replace op aliases and drop unrelated metadata keys."""

    normalized: Dict[str, Any] = {
        key: operation[key]
        for key in ("op", "path", "old", "selector", "new")
        if key in operation
    }
    old_aliases = [key for key in REPLACE_IN_FILE_OLD_ALIASES if key in operation]
    new_aliases = [key for key in REPLACE_IN_FILE_NEW_ALIASES if key in operation]
    if "old" not in normalized:
        if len(old_aliases) == 1:
            normalized["old"] = operation[old_aliases[0]]
        elif len(old_aliases) > 1:
            for key in old_aliases:
                normalized[key] = operation[key]
    else:
        for key in old_aliases:
            normalized[key] = operation[key]

    if "new" not in normalized:
        if len(new_aliases) == 1:
            normalized["new"] = operation[new_aliases[0]]
        elif len(new_aliases) > 1:
            for key in new_aliases:
                normalized[key] = operation[key]
    else:
        for key in new_aliases:
            normalized[key] = operation[key]

    return normalized


def render_supported_file_ops() -> str:
    return ", ".join(sorted(SUPPORTED_FILE_OPS))

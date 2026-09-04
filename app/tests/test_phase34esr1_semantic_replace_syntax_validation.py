"""ESR1 provider-free semantic replacement replay regressions."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.services.orchestration.planning.source_materialization import (
    current_source_version_identity,
)
from app.services.orchestration.validation.rules.contract_python import (
    _plan_python_source_syntax_issues,
)


SOURCE = '''"""Formatting used by the customer greeting."""


def format_customer_name(value: str) -> str:
    """Normalize surrounding and repeated whitespace in a display name."""

    return " ".join(str(value).strip().split())
'''
PATH = "src/greeting/formatting.py"
START_BYTE = 202
END_BYTE = 209


def _operation(root: Path, new: str) -> dict:
    target = root / PATH
    selected = target.read_bytes()[START_BYTE:END_BYTE]
    return {
        "op": "replace_in_file",
        "path": PATH,
        "selector": {
            "schema_version": "source-region/1",
            "canonical_path": PATH,
            "expected_source_version": current_source_version_identity(target),
            "start_byte": START_BYTE,
            "end_byte": END_BYTE,
            "selected_region_sha256": hashlib.sha256(selected).hexdigest(),
            "derivation_kind": "exact_region",
        },
        "new": new,
    }


def _plan(operation: dict) -> list[dict]:
    return [{"step_number": 1, "ops": [operation]}]


def _write_source(root: Path) -> None:
    target = root / PATH
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE, encoding="utf-8")


def test_exact_case_b_semantic_payload_is_rejected_before_mutation(tmp_path):
    _write_source(tmp_path)
    operation = _operation(
        tmp_path,
        '''def format_customer_name(value: str) -> str:
    """Normalize surrounding and repeated whitespace in a display name."""

    if not value or not value.strip():
        return ""
    return " ".join(str(value).strip().split()).title()''',
    )

    issues = _plan_python_source_syntax_issues(_plan(operation), tmp_path)

    assert len(issues) == 1
    assert issues[0]["path"] == PATH
    assert issues[0]["line"] == 7
    assert "invalid syntax" in issues[0]["message"]


def test_valid_semantic_region_replacement_passes_syntax_gate(tmp_path):
    _write_source(tmp_path)
    operation = _operation(tmp_path, "strip().title()")

    assert _plan_python_source_syntax_issues(_plan(operation), tmp_path) == []


def test_stale_semantic_selector_is_left_to_identity_validation(tmp_path):
    _write_source(tmp_path)
    operation = _operation(tmp_path, "not valid Python !")
    operation["selector"]["selected_region_sha256"] = "0" * 64

    assert _plan_python_source_syntax_issues(_plan(operation), tmp_path) == []


def test_semantic_selector_for_another_path_is_not_simulated(tmp_path):
    _write_source(tmp_path)
    operation = _operation(tmp_path, "not valid Python !")
    operation["selector"]["canonical_path"] = "src/greeting/other.py"

    assert _plan_python_source_syntax_issues(_plan(operation), tmp_path) == []

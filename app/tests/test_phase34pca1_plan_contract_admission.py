"""PHASE34-PCA1 provider-free Plan admission regressions."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from app.services.orchestration.operations.file_ops_contract import (
    ReplaceOperationMode,
    classify_replace_operation,
)
from app.services.orchestration.operations.source_region_identity import (
    SourceRegionIdentity,
)
from app.services.orchestration.planning.source_materialization import (
    current_source_version_identity,
)
from app.services.orchestration.validation.accepted_path_authority import (
    accepted_plan_identity,
    plan_identity_text,
)
from app.services.orchestration.validation.validator import ValidatorService


FORMATTER_PATH = "src/greeting/formatting.py"
FORMATTER_TEST_PATH = "tests/test_formatting.py"
STORE_PATH = "src/store/checkout.py"
STORE_TEST_PATH = "tests/test_store.py"
INVOICE_PATH = "src/store/invoice_preview.py"

B_PROMPT = (
    "Customer-facing names need consistent display formatting. Normalize a name "
    "by trimming outer whitespace, collapsing internal whitespace to single "
    "spaces, title-casing each word, and returning an empty string when input "
    "is blank. Preserve existing behavior for already-normalized names and add "
    "regression coverage for repeated spaces and blank input. Keep it offline "
    "and runnable with the repository's existing test command."
)
C_PROMPT = (
    "During checkout, reject any zero or negative order total before a payment "
    "is sent, and raise a clear ValueError. Positive totals must still be sent "
    "unchanged. Keep invoice previews able to display a zero total, and add "
    "regression coverage for rejected submissions, preserved positive charging, "
    "and the unchanged preview behavior. Keep it offline and runnable with the "
    "repository's existing test command."
)

FORMATTER_SOURCE = (
    '"""Formatting used by the customer greeting."""\n\n\n'
    "def format_customer_name(value: str) -> str:\n"
    '    """Normalize surrounding and repeated whitespace in a display name."""\n\n'
    '    return " ".join(str(value).strip().split())\n'
)
FORMATTER_TEST = (
    "from greeting.formatting import format_customer_name\n\n"
    "def test_existing_display_name_keeps_word_spacing():\n"
    '    assert format_customer_name("Ada Lovelace") == "Ada Lovelace"\n'
)
STORE_SOURCE = (
    '"""Customer checkout behavior."""\n\n\n'
    "class RecordingGateway:\n"
    '    """Test-friendly gateway stand-in with no external dependency."""\n\n'
    "    def __init__(self) -> None:\n"
    "        self.charges: list[int] = []\n\n"
    "    def charge(self, amount_cents: int) -> str:\n"
    "        self.charges.append(amount_cents)\n"
    '        return f"charged:{amount_cents}"\n\n\n'
    "def submit_order(total_cents: int, gateway: RecordingGateway) -> str:\n"
    '    """Submit an order total to the payment gateway."""\n\n'
    "    return gateway.charge(total_cents)\n"
)
STORE_TEST = (
    "import pytest\n\n"
    "from store.checkout import RecordingGateway, submit_order\n"
    "from store.invoice_preview import displayed_total\n\n\n"
    "def test_positive_checkout_reaches_gateway():\n"
    "    gateway = RecordingGateway()\n\n"
    '    assert submit_order(1250, gateway) == "charged:1250"\n'
    "    assert gateway.charges == [1250]\n\n\n"
    "def test_preview_can_display_zero():\n"
    "    assert displayed_total(0) == 0\n"
)


def _seed_b_workspace(root: Path) -> None:
    (root / "src" / "greeting").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "greeting" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "greeting" / "formatting.py").write_text(
        FORMATTER_SOURCE, encoding="utf-8"
    )
    (root / "tests" / "test_formatting.py").write_text(FORMATTER_TEST, encoding="utf-8")


def _seed_c_workspace(root: Path) -> None:
    (root / "src" / "store").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "store" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "store" / "checkout.py").write_text(STORE_SOURCE, encoding="utf-8")
    (root / "src" / "store" / "invoice_preview.py").write_text(
        '"""Invoice preview behavior, which is not a payment submission."""\n\n\n'
        "def displayed_total(total_cents: int) -> int:\n"
        '    """Return a non-negative amount suitable for a preview display."""\n\n'
        "    return max(0, int(total_cents))\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_store.py").write_text(STORE_TEST, encoding="utf-8")


def _semantic_replace(root: Path) -> dict:
    source = (root / FORMATTER_PATH).read_bytes()
    selector = SourceRegionIdentity.from_region(
        canonical_path=FORMATTER_PATH,
        expected_source_version=current_source_version_identity(root / FORMATTER_PATH),
        start_byte=0,
        end_byte=len(source),
        selected_region_sha256=hashlib.sha256(source).hexdigest(),
    ).to_dict()
    return {
        "op": "replace_in_file",
        "path": FORMATTER_PATH,
        "selector": selector,
        "new": (
            "def format_customer_name(value: str) -> str:\n"
            '    """Normalize surrounding and repeated whitespace in a display name."""\n\n'
            "    stripped = str(value).strip()\n"
            "    if not stripped:\n"
            '        return ""\n'
            '    return " ".join(stripped.split()).title()\n'
        ),
    }


def _b_plan(
    root: Path,
    *,
    include_test_target: bool = True,
    materialize_test_target: bool = True,
) -> list[dict]:
    plan = [
        {
            "step_number": 1,
            "description": "Inspect existing test file to understand current coverage and imports",
            "commands": ["cat tests/test_formatting.py"],
            "verification": "python -c \"import pathlib; pathlib.Path('tests/test_formatting.py').exists() or exit(1)\"",
            "rollback": None,
            "expected_files": [FORMATTER_TEST_PATH] if include_test_target else [],
            "ops": [],
        },
        {
            "step_number": 2,
            "description": "Update format_customer_name to title-case words and handle blank input",
            "commands": [],
            "verification": (
                'python -c "from src.greeting.formatting import '
                "format_customer_name; assert format_customer_name('  ada  lovelace  ') "
                "== 'Ada Lovelace'; assert format_customer_name('') == ''; "
                "assert format_customer_name('   ') == ''\""
            ),
            "rollback": None,
            "expected_files": [FORMATTER_PATH],
            "ops": [_semantic_replace(root)],
        },
        {
            "step_number": 3,
            "description": "Run existing tests to ensure regression coverage passes",
            "commands": ["python -m pytest tests/test_formatting.py -v"],
            "verification": "python -m pytest tests/test_formatting.py -v",
            "rollback": None,
            "expected_files": [],
            "ops": [],
        },
    ]
    if include_test_target and materialize_test_target:
        plan[2]["expected_files"] = [FORMATTER_TEST_PATH]
        plan[2]["ops"] = [
            {
                "op": "write_file",
                "path": FORMATTER_TEST_PATH,
                "content": FORMATTER_TEST,
            }
        ]
    return plan


def _c_plan(root: Path, *, include_test_target: bool = False) -> list[dict]:
    steps = [
        {
            "step_number": 1,
            "description": "Inspect existing test and invoice preview files to understand the contract",
            "commands": ["cat tests/test_store.py", "cat src/store/invoice_preview.py"],
            "verification": "python -c \"import pathlib; pathlib.Path('tests/test_store.py').exists() and pathlib.Path('src/store/invoice_preview.py').exists() or exit(1)\"",
            "rollback": None,
            "expected_files": [STORE_TEST_PATH, INVOICE_PATH],
            "ops": [],
        },
        {
            "step_number": 2,
            "description": "Update checkout.py to reject zero/negative totals and update invoice_preview.py if needed",
            "commands": [
                'python -c "from src.store.checkout import submit_order, RecordingGateway; g = RecordingGateway(); submit_order(100, g); exit(0 if g.charges == [100] else 1)"'
            ],
            "verification": 'python -c "from src.store.checkout import submit_order, RecordingGateway; g = RecordingGateway(); submit_order(100, g); exit(0 if g.charges == [100] else 1)"',
            "rollback": None,
            "expected_files": [STORE_PATH],
            "ops": [
                {
                    "op": "write_file",
                    "path": STORE_PATH,
                    "content": STORE_SOURCE.replace(
                        "    return gateway.charge(total_cents)\n",
                        '    if total_cents <= 0:\n        raise ValueError("Order total must be positive")\n'
                        "    return gateway.charge(total_cents)\n",
                    ),
                }
            ],
        },
        {
            "step_number": 3,
            "description": "Run existing tests to verify behavior and coverage",
            "commands": ["python -m pytest tests/test_store.py -v"],
            "verification": "python -m pytest tests/test_store.py -v",
            "rollback": None,
            "expected_files": [],
            "ops": [],
        },
    ]
    if include_test_target:
        steps[2]["expected_files"] = [STORE_TEST_PATH]
        steps[2]["ops"] = [
            {"op": "write_file", "path": STORE_TEST_PATH, "content": STORE_TEST}
        ]
    return steps


def _validate(
    root: Path, plan: list[dict], prompt: str, *, intent_mode: str = "default"
):
    return ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=prompt,
        execution_profile="implementation",
        project_dir=root,
        is_first_ordered_task=True,
        intent_mode=intent_mode,
    )


def test_case_b_valid_semantic_replace_is_not_rejected_by_minimum_evidence(tmp_path):
    _seed_b_workspace(tmp_path)
    plan = _b_plan(tmp_path)

    outcome = _validate(tmp_path, plan, B_PROMPT)

    assert outcome.accepted, outcome.reasons
    contract = outcome.details["task1_bootstrap_contract"]
    assert contract["minimum_implementation_evidence"] is True
    assert contract["expected_test_files"] == [FORMATTER_TEST_PATH]
    assert (
        classify_replace_operation(plan[1]["ops"][0])
        is ReplaceOperationMode.SEMANTIC_REPLACE
    )


def test_case_b_still_requires_primary_and_explicit_test_targets(tmp_path):
    _seed_b_workspace(tmp_path)
    plan = _b_plan(tmp_path, include_test_target=False, materialize_test_target=False)

    outcome = _validate(tmp_path, plan, B_PROMPT)

    assert not outcome.accepted
    contract = outcome.details["task1_bootstrap_contract"]
    assert FORMATTER_PATH in contract["expected_source_files"]
    assert "task1_bootstrap_missing_expected_test_files" in contract["violation_codes"]


def test_case_c_without_regression_target_is_rejected_before_acceptance(tmp_path):
    _seed_c_workspace(tmp_path)

    outcome = _validate(tmp_path, _c_plan(tmp_path), C_PROMPT)

    assert not outcome.accepted
    contract = outcome.details["task1_bootstrap_contract"]
    assert contract["expected_test_reason"] == "explicit_code_test_intent"
    assert "task1_bootstrap_missing_expected_test_files" in contract["violation_codes"]


def test_case_c_accepts_minimum_targets_without_invoice_preview(tmp_path):
    _seed_c_workspace(tmp_path)

    outcome = _validate(tmp_path, _c_plan(tmp_path, include_test_target=True), C_PROMPT)

    assert outcome.accepted, outcome.reasons
    contract = outcome.details["task1_bootstrap_contract"]
    assert contract["expected_test_files"] == [STORE_TEST_PATH]
    assert STORE_PATH in contract["expected_source_files"]
    assert not any(
        operation.get("path") == INVOICE_PATH
        for step in _c_plan(tmp_path, include_test_target=True)
        for operation in step.get("ops", [])
    )


def test_invoice_preview_is_optional_for_case_c_admission(tmp_path):
    _seed_c_workspace(tmp_path)
    plan = _c_plan(tmp_path, include_test_target=True)
    plan[0]["expected_files"] = [STORE_TEST_PATH]
    plan[0]["commands"] = ["cat tests/test_store.py"]

    outcome = _validate(tmp_path, plan, C_PROMPT)

    assert outcome.accepted, outcome.reasons
    assert INVOICE_PATH not in {
        path for step in plan for path in step.get("expected_files", [])
    }


def test_apa_identity_input_remains_plan_bound(tmp_path):
    _seed_b_workspace(tmp_path)
    plan = _b_plan(tmp_path)
    copied = copy.deepcopy(plan)

    assert plan_identity_text(plan) == plan_identity_text(copied)
    assert accepted_plan_identity(plan) == accepted_plan_identity(copied)

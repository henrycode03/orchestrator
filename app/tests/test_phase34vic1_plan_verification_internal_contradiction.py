"""PHASE34-VIC1 — Plan-internal verification contradiction validator contract.

Provider-free replays. The two defect anchors are the exact provider Plans
PHASE34-S2X captured as ``non_thinking-3`` and ``non_thinking-5``: each mutates
``greeter.py`` so ``greet("Ada")`` no longer returns ``"Hello"``, leaves
``test_greeter.py`` asserting ``greet("Ada") == "Hello"``, and then names that
same test file as its verification. Both reached validator ``accepted``.

The guards below pin the conservative boundary: the rule fails closed only on a
contradiction it can actually prove, and stays silent everywhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.orchestration.planning.source_materialization import (
    materialize_planner_source_context,
)
from app.services.orchestration.validation.validator import ValidatorService

pytestmark = pytest.mark.critical_regression

GREETER = (
    '"""Small greeting API."""\n'
    "\n"
    "def greet(name: str) -> str:\n"
    '    """Return the current generic greeting."""\n'
    '    return "Hello"\n'
)
TEST_GREETER = (
    "from greeter import greet\n"
    "\n"
    "\n"
    "def test_greet_keeps_current_behavior():\n"
    '    assert greet("Ada") == "Hello"\n'
)
TASK = (
    "Update the greeting behavior to include the person's name while preserving "
    "the existing public function and current behavior for callers. Update tests "
    "only if necessary and verify the project still passes."
)
OLD_GREET = (
    "def greet(name: str) -> str:\n"
    '    """Return the current generic greeting."""\n'
    '    return "Hello"'
)
NEW_GREET_NAMED = (
    "def greet(name: str) -> str:\n"
    '    """Return a greeting including the person\'s name."""\n'
    '    return f"Hello {name}"'
)
FINDING = "plan_verification_internal_contradiction"


def _workspace(tmp_path: Path, *, test_source: str = TEST_GREETER) -> Path:
    (tmp_path / "greeter.py").write_text(GREETER, encoding="utf-8")
    (tmp_path / "test_greeter.py").write_text(test_source, encoding="utf-8")
    return tmp_path


def _validate(root: Path, plan: list[dict]) -> object:
    materialization = materialize_planner_source_context(
        root,
        task_description=TASK,
        expected_paths=["greeter.py", "test_greeter.py"],
    )
    return ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=TASK,
        execution_profile="implementation",
        project_dir=root,
        source_materialization=materialization,
    )


# The exact verification `_strengthen_weak_expected_file_verifications` attached
# to step 1 during S2X normalization. Anchors must replay the plan the validator
# actually received, not the raw provider text -- the raw text is separately held
# at repair_required by `missing_verification_command`, a different finding.
EXISTENCE_VERIFICATION = (
    'python -c "import pathlib,sys; files=[\\"greeter.py\\"]; '
    'sys.exit(0 if all(pathlib.Path(p).exists() for p in files) else 1)"'
)


def _greeter_step(new_body: str = NEW_GREET_NAMED, old_body: str = OLD_GREET) -> dict:
    return {
        "step_number": 1,
        "description": "Update greeter.py to include the person's name.",
        "commands": [],
        "verification": EXISTENCE_VERIFICATION,
        "rollback": "Revert greeter.py.",
        "expected_files": ["greeter.py"],
        "ops": [
            {
                "op": "replace_in_file",
                "path": "greeter.py",
                "old": old_body,
                "new": new_body,
            }
        ],
    }


def _pytest_step(command: str = "python -m pytest test_greeter.py -v") -> dict:
    return {
        "step_number": 2,
        "description": "Run tests to verify the project is in a valid state.",
        "commands": [command],
        "verification": command,
        "rollback": "Revert changes to greeter.py if tests fail.",
        "expected_files": [],
        "ops": [],
    }


def _anchor_plan() -> list[dict]:
    """The exact S2X non_thinking-3 / non_thinking-5 provider Plan shape."""

    return [_greeter_step(), _pytest_step()]


def _findings(verdict: object) -> list[str]:
    details = getattr(verdict, "details", None) or {}
    return list(details.get("validator_rule_ids") or [])


# --------------------------------------------------------- defect anchors ---


@pytest.mark.parametrize("anchor", ["non_thinking-3", "non_thinking-5"])
def test_s2x_anchor_plan_is_not_accepted(tmp_path, anchor):
    """T1/T2: both S2X anchors must stop reaching `accepted`."""

    verdict = _validate(_workspace(tmp_path), _anchor_plan())
    assert not verdict.accepted, f"{anchor} still accepted: {verdict.reasons}"
    assert FINDING in _findings(verdict), _findings(verdict)


def test_contradiction_finding_carries_debugging_evidence(tmp_path):
    """The finding must name the mutation, the test, the anchor and the reason."""

    verdict = _validate(_workspace(tmp_path), _anchor_plan())
    evidence = (verdict.details or {}).get(FINDING)
    assert isinstance(evidence, dict), verdict.details
    assert evidence["mutated_implementation_path"] == "greeter.py"
    assert evidence["unchanged_test_path"] == "test_greeter.py"
    assert evidence["verification_command"] == "python -m pytest test_greeter.py -v"
    assert evidence["asserted_expression"]
    assert evidence["expected_value"] == "Hello"
    assert evidence["planned_value"] == "Hello Ada"
    assert evidence["contradiction_reason"]
    # Evidence must stay compact -- no source dumps.
    assert len(json.dumps(evidence)) < 1200


def test_contradiction_is_repair_required_not_silent(tmp_path):
    """Severity follows existing convention: repairable, so status is repair_required."""

    verdict = _validate(_workspace(tmp_path), _anchor_plan())
    assert verdict.status == "repair_required", verdict.status


# ------------------------------------------------- false-positive guards ---


def test_guard_1_implementation_and_test_updated_together(tmp_path):
    """§8.1 consistent update must stay clean."""

    plan = [
        _greeter_step(),
        {
            "step_number": 2,
            "description": "Update the test to the new expectation.",
            "commands": [],
            "verification": None,
            "rollback": None,
            "expected_files": ["test_greeter.py"],
            "ops": [
                {
                    "op": "replace_in_file",
                    "path": "test_greeter.py",
                    "old": 'assert greet("Ada") == "Hello"',
                    "new": 'assert greet("Ada") == "Hello Ada"',
                }
            ],
        },
        _pytest_step(),
    ]
    verdict = _validate(_workspace(tmp_path), plan)
    assert FINDING not in _findings(verdict), _findings(verdict)


def test_guard_2_change_not_covered_by_the_assertion(tmp_path):
    """§8.2 mutating a function the assertion never calls must stay clean."""

    plan = [
        {
            "step_number": 1,
            "description": "Add a separate farewell helper.",
            "commands": [],
            "verification": None,
            "rollback": None,
            "expected_files": ["greeter.py"],
            "ops": [
                {
                    "op": "append_file",
                    "path": "greeter.py",
                    "content": (
                        "\n\ndef farewell(name: str) -> str:\n"
                        '    return f"Bye {name}"\n'
                    ),
                }
            ],
        },
        _pytest_step(),
    ]
    verdict = _validate(_workspace(tmp_path), plan)
    assert FINDING not in _findings(verdict), _findings(verdict)


def test_guard_3_verification_targets_a_different_test_file(tmp_path):
    """§8.3 a command naming another test file proves nothing about this one."""

    root = _workspace(tmp_path)
    (root / "test_other.py").write_text(
        "def test_other():\n    assert True\n", encoding="utf-8"
    )
    plan = [_greeter_step(), _pytest_step("python -m pytest test_other.py -v")]
    verdict = _validate(root, plan)
    assert FINDING not in _findings(verdict), _findings(verdict)


def test_guard_4_indirect_assertion_cannot_be_proven(tmp_path):
    """§8.4 a non-literal assertion is not statically decidable."""

    indirect = (
        "from greeter import greet\n"
        "\n"
        "\n"
        "def test_greet_is_a_string():\n"
        '    assert isinstance(greet("Ada"), str)\n'
    )
    verdict = _validate(_workspace(tmp_path, test_source=indirect), _anchor_plan())
    assert FINDING not in _findings(verdict), _findings(verdict)


def test_guard_5_optional_behavior_preserves_the_asserted_default(tmp_path):
    """§8.5 preserving the asserted behaviour behind an opt-in must stay clean."""

    preserving = (
        "def greet(name: str, include_name: bool = False) -> str:\n"
        '    """Return the greeting, optionally including the name."""\n'
        '    return "Hello"'
    )
    plan = [_greeter_step(new_body=preserving), _pytest_step()]
    verdict = _validate(_workspace(tmp_path), plan)
    assert FINDING not in _findings(verdict), _findings(verdict)


def test_guard_6_test_source_unavailable(tmp_path):
    """§8.6 with no materialized test source there is nothing to contradict."""

    (tmp_path / "greeter.py").write_text(GREETER, encoding="utf-8")
    plan = [_greeter_step(), _pytest_step("python -m pytest -v")]
    verdict = _validate(tmp_path, plan)
    assert FINDING not in _findings(verdict), _findings(verdict)


def test_guard_7_broad_pytest_command_is_not_proof(tmp_path):
    """§8.7 a bare `pytest` is deliberately treated as unproven."""

    plan = [_greeter_step(), _pytest_step("python -m pytest")]
    verdict = _validate(_workspace(tmp_path), plan)
    assert FINDING not in _findings(verdict), _findings(verdict)


def test_guard_8_non_literal_implementation_is_not_reducible(tmp_path):
    """§8.8 a body that is not a single literal return must stay clean."""

    branching = (
        "def greet(name: str) -> str:\n"
        '    """Return a greeting."""\n'
        "    if not name:\n"
        '        return "Hello"\n'
        '    return f"Hello {name}"'
    )
    plan = [_greeter_step(new_body=branching), _pytest_step()]
    verdict = _validate(_workspace(tmp_path), plan)
    assert FINDING not in _findings(verdict), _findings(verdict)


def test_guard_9_plan_that_leaves_behaviour_unchanged(tmp_path):
    """A no-op refactor of the same literal must stay clean."""

    same_value = (
        "def greet(name: str) -> str:\n"
        '    """Return the current generic greeting."""\n'
        '    return "Hello"'
    )
    plan = [_greeter_step(new_body=same_value), _pytest_step()]
    verdict = _validate(_workspace(tmp_path), plan)
    assert FINDING not in _findings(verdict), _findings(verdict)

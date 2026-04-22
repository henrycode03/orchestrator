from __future__ import annotations

from types import SimpleNamespace

from app.services.orchestration.completion_flow import (
    _augment_completion_verification_command,
    _classify_completion_verification_failure,
)


def test_missing_jest_binary_is_treated_as_repairable_completion_verification():
    completion_validation = SimpleNamespace(
        profile="implementation",
        details={"expected_core_files": ["src/index.ts", "src/utils/format.test.ts"]},
    )

    verdict = _classify_completion_verification_failure(
        command="pnpm test",
        source="package.json test script via pnpm",
        verification_output=(
            "> demo@1.0.0 test /workspace/demo\n" "> jest\n" "sh: 1: jest: not found\n"
        ),
        completion_validation=completion_validation,
    )

    assert verdict is not None
    assert verdict.repairable is True
    assert verdict.stage == "completion_verification"
    assert "dependencies are missing or not installed" in verdict.reasons[0]
    assert verdict.details["verification_command"] == "pnpm test"
    assert "src/utils/format.test.ts" in verdict.details["expected_core_files"]


def test_real_test_failure_is_not_reclassified_as_missing_dependency():
    completion_validation = SimpleNamespace(
        profile="implementation",
        details={"expected_core_files": ["src/index.ts"]},
    )

    verdict = _classify_completion_verification_failure(
        command="pnpm test",
        source="package.json test script via pnpm",
        verification_output=(
            "FAIL src/index.test.ts\n" "Expected: 2\n" "Received: 1\n"
        ),
        completion_validation=completion_validation,
    )

    assert verdict is None


def test_vitest_completion_verification_excludes_openclaw_snapshots():
    command = _augment_completion_verification_command(
        "pnpm test",
        "vitest run",
    )

    assert command == "pnpm test -- --exclude=.openclaw/**"


def test_module_resolution_failure_is_treated_as_repairable_verification_issue():
    completion_validation = SimpleNamespace(
        profile="implementation",
        details={
            "expected_core_files": ["src/utils/format.ts", "src/utils/format.spec.ts"]
        },
    )

    verdict = _classify_completion_verification_failure(
        command="pnpm test -- --exclude=.openclaw/**",
        source="package.json test script via pnpm",
        verification_output=(
            "FAIL src/utils/format.spec.ts\n"
            "Error: Failed to load url ./format.js in src/utils/format.spec.ts. "
            "Does the file exist?\n"
        ),
        completion_validation=completion_validation,
    )

    assert verdict is not None
    assert verdict.repairable is True
    assert "repairable test/module issue" in verdict.reasons[0]

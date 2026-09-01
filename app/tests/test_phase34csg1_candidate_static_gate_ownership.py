"""PHASE34-CSG1 — project-owned candidate static gate admission."""

from pathlib import Path

import pytest

from app.services.orchestration.phases.completion_deterministic_repair import (
    attempt_deterministic_candidate_repair,
)
from app.services.orchestration.types import CandidateFinding, CandidateValidationResult
from app.services.orchestration.validation.candidate_checks import (
    discover_candidate_static_policy,
    validate_candidate_delta,
)
from app.tests.phase33c4_test_helpers import executor_test_authority


def _run(project_dir: Path, paths: list[str]):
    return validate_candidate_delta(
        project_dir=project_dir,
        change_set={
            "added_files": paths,
            "modified_files": [],
            "deleted_files": [],
        },
        plan=[],
        task_prompt="Validate the candidate change",
        include_static_checks=True,
    )


def _rules(run) -> set[str]:
    return {finding.rule_id for finding in run.findings}


def test_installed_style_tools_are_not_product_gates_without_project_policy(
    tmp_path: Path,
):
    candidate = tmp_path / "module.py"
    candidate.write_text("def answer( ):\n return 42\n", encoding="utf-8")

    run = _run(tmp_path, ["module.py"])

    assert _rules(run) == set()
    assert any("compileall" in command for command in run.commands_run)
    assert not any("black" in command for command in run.commands_run)
    assert not any("flake8" in command for command in run.commands_run)


def test_project_black_configuration_admits_only_black(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.black]\nline-length = 88\n", encoding="utf-8"
    )
    (tmp_path / "module.py").write_text(
        "def answer( ):\n return 42\n", encoding="utf-8"
    )

    run = _run(tmp_path, ["module.py"])

    assert _rules(run) == {"candidate_black_failed"}
    assert any("black" in command for command in run.commands_run)
    assert not any("flake8" in command for command in run.commands_run)


def test_project_flake8_configuration_admits_only_flake8(tmp_path: Path):
    (tmp_path / ".flake8").write_text("[flake8]\n", encoding="utf-8")
    (tmp_path / "module.py").write_text(
        "def answer( ):\n return 42\n", encoding="utf-8"
    )

    run = _run(tmp_path, ["module.py"])

    assert _rules(run) == {"candidate_flake8_failed"}
    assert not any("black" in command for command in run.commands_run)
    assert any("flake8" in command for command in run.commands_run)


@pytest.mark.parametrize("filename", ["setup.cfg", "tox.ini"])
def test_supported_flake8_config_files_admit_the_project_gate(
    tmp_path: Path, filename: str
):
    (tmp_path / filename).write_text("[flake8]\n", encoding="utf-8")

    policy = discover_candidate_static_policy(tmp_path)

    assert policy.flake8_admitted is True
    assert policy.flake8_source == f"{filename}:[flake8]"


def test_project_owned_black_and_flake8_gates_use_their_configs(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.black]\nline-length = 88\n", encoding="utf-8"
    )
    (tmp_path / ".flake8").write_text(
        "[flake8]\nmax-line-length = 88\n", encoding="utf-8"
    )
    (tmp_path / "module.py").write_text(
        "def answer():\n    return 42\n", encoding="utf-8"
    )

    run = _run(tmp_path, ["module.py"])

    assert _rules(run) == set()
    assert any("black" in command for command in run.commands_run)
    assert any("flake8" in command for command in run.commands_run)


def test_project_owned_contradictory_style_configs_leave_policy_failure_visible(
    tmp_path: Path,
):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.black]\nline-length = 88\n", encoding="utf-8"
    )
    (tmp_path / ".flake8").write_text(
        "[flake8]\nmax-line-length = 79\n", encoding="utf-8"
    )
    (tmp_path / "module.py").write_text(
        'VALUE = "' + ("x" * 72) + '"\n', encoding="utf-8"
    )

    run = _run(tmp_path, ["module.py"])

    assert _rules(run) == {"candidate_flake8_failed"}
    assert any("black" in command for command in run.commands_run)
    assert any("flake8" in command for command in run.commands_run)


def test_unconfigured_syntax_error_remains_a_universal_correctness_failure(
    tmp_path: Path,
):
    (tmp_path / "module.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    run = _run(tmp_path, ["module.py"])

    assert _rules(run) == {"candidate_python_compile_failed"}
    assert not any("black" in command for command in run.commands_run)
    assert not any("flake8" in command for command in run.commands_run)


def test_unconfigured_declared_test_failure_remains_a_correctness_failure(
    tmp_path: Path,
):
    (tmp_path / "test_candidate.py").write_text(
        "def test_failure():\n    assert False\n", encoding="utf-8"
    )

    run = _run(tmp_path, ["test_candidate.py"])

    assert _rules(run) == {"focused_pytest_failed"}
    assert not any("black" in command for command in run.commands_run)
    assert not any("flake8" in command for command in run.commands_run)


def test_dcr_refuses_a_black_finding_without_project_black_policy(tmp_path: Path):
    candidate = tmp_path / "module.py"
    original = "def answer( ):\n return 42\n"
    candidate.write_text(original, encoding="utf-8")
    finding = CandidateFinding(
        rule_id="candidate_black_failed",
        source="black",
        category="static",
        severity="error",
        attribution="candidate_introduced",
        repairable=True,
        message="Candidate-scoped black failed",
        evidence={"paths": ["module.py"]},
    )
    authority = executor_test_authority(
        tmp_path,
        [{"op": "write_file", "path": "module.py"}],
        plan=[{"step_number": 1, "verification": "python -m pytest"}],
    )

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=CandidateValidationResult.from_findings(
            profile="implementation", findings=[finding]
        ),
        project_dir=tmp_path,
        accepted_path_authority=authority,
    )

    assert outcome.status == "skipped"
    assert outcome.reason == "black_gate_not_admitted"
    assert candidate.read_text(encoding="utf-8") == original

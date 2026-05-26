from pathlib import Path

from app.services.orchestration.planning.planner import PlannerService
from app.services.project.source_imports import (
    extract_python_test_contract,
    python_test_source_context_from_tests,
    render_python_test_contract_summary,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_phase11a_extractor_maps_test_imports_to_existing_source_files(
    tmp_path: Path,
):
    _write(
        tmp_path / "src" / "small_cli" / "__init__.py",
        "",
    )
    _write(
        tmp_path / "src" / "small_cli" / "cli.py",
        "def build_parser():\n"
        "    return None\n"
        "\n"
        "def main(argv=None):\n"
        "    return 0\n",
    )
    _write(
        tmp_path / "tests" / "test_cli.py",
        "from small_cli.cli import build_parser, main\n"
        "\n"
        "def test_uppercase_option_prints_uppercase_message(capsys):\n"
        "    assert main(['--uppercase', 'hello']) == 0\n"
        "    assert capsys.readouterr().out.strip() == 'HELLO'\n",
    )

    contract = extract_python_test_contract(tmp_path)

    assert contract is not None
    assert "tests/test_cli.py" in contract.test_files
    assert "from small_cli.cli import build_parser, main" in contract.imports
    assert ("src/small_cli/cli.py",) == tuple(
        path for path, _reason in contract.source_targets
    )
    assert "main(['--uppercase', 'hello'])" in contract.public_calls
    assert any("HELLO" in assertion for assertion in contract.assertions)


def test_phase11a_extractor_infers_missing_src_module_from_package_reexport(
    tmp_path: Path,
):
    _write(
        tmp_path / "src" / "import_repair" / "__init__.py",
        "from import_repair.formatters import normalize_greeting\n"
        "\n"
        "__all__ = ['normalize_greeting']\n",
    )
    _write(
        tmp_path / "tests" / "test_formatter.py",
        "from import_repair import normalize_greeting\n"
        "\n"
        "def test_normalize_greeting_trims_and_title_cases_name():\n"
        "    assert normalize_greeting('  ada   lovelace ') == 'Hello, Ada Lovelace!'\n",
    )

    contract = extract_python_test_contract(tmp_path)

    assert contract is not None
    assert ("src/import_repair/__init__.py",) == tuple(
        path for path, _reason in contract.source_targets
    )
    assert ("src/import_repair/formatters.py",) == tuple(
        path for path, _reason in contract.missing_source_targets
    )
    missing_reason = contract.missing_source_targets[0][1]
    assert "import_repair.formatters" in missing_reason
    assert "normalize_greeting" in missing_reason


def test_phase11a_summary_guides_preserve_tests_and_prefer_source_edits(
    tmp_path: Path,
):
    _write(tmp_path / "src" / "pkg" / "__init__.py", "from pkg.feature import run\n")
    _write(
        tmp_path / "tests" / "test_feature.py",
        "from pkg import run\n" "\n" "def test_run():\n" "    assert run('x') == 'X'\n",
    )

    summary = python_test_source_context_from_tests(tmp_path, max_chars=1200)

    assert "## TEST CONTRACT SUMMARY" in summary
    assert "Existing tests are the contract. Preserve them." in summary
    assert "Do not rewrite tests or verifier commands" in summary
    assert "Prefer source edits under src/" in summary
    assert "src/pkg/feature.py" in summary
    assert "run('x') should equal 'X'" in summary
    assert len(summary) <= 1200


def test_phase11a_minimal_planning_prompt_includes_test_contract_summary(
    tmp_path: Path,
):
    _write(tmp_path / "src" / "small_cli" / "__init__.py", "")
    _write(
        tmp_path / "src" / "small_cli" / "cli.py",
        "def build_parser():\n"
        "    return None\n"
        "\n"
        "def main(argv=None):\n"
        "    return 0\n",
    )
    _write(
        tmp_path / "tests" / "test_cli.py",
        "from small_cli.cli import build_parser, main\n"
        "\n"
        "def test_uppercase_option_prints_uppercase_message(capsys):\n"
        "    assert main(['--uppercase', 'hello']) == 0\n"
        "    assert capsys.readouterr().out.strip() == 'HELLO'\n",
    )

    prompt = PlannerService.build_minimal_planning_prompt(
        "Add the --uppercase option to this small Python CLI.",
        tmp_path,
        workspace_has_existing_files=True,
    )

    assert "## TEST CONTRACT SUMMARY" in prompt
    assert "src/small_cli/cli.py" in prompt
    assert "main(['--uppercase', 'hello']) should equal 0" in prompt
    assert "Preserve them." in prompt
    assert len(prompt) < 12000


def test_phase11a_summary_is_target_oriented_without_long_excerpts(tmp_path: Path):
    _write(tmp_path / "src" / "small_cli" / "__init__.py", "")
    _write(
        tmp_path / "src" / "small_cli" / "cli.py",
        "def build_parser():\n    return None\n\ndef main(argv=None):\n    return 0\n",
    )
    _write(
        tmp_path / "tests" / "test_cli.py",
        "from small_cli.cli import build_parser, main\n"
        "\n"
        "def test_uppercase_option_prints_uppercase_message(capsys):\n"
        "    assert main(['--uppercase', 'hello']) == 0\n"
        "    assert capsys.readouterr().out.strip() == 'HELLO'\n",
    )

    summary = python_test_source_context_from_tests(tmp_path, max_chars=900)

    assert "Required source targets:" in summary
    assert "src/small_cli/cli.py" in summary
    assert "Expected behavior:" in summary
    assert "main(['--uppercase', 'hello']) should equal 0" in summary
    assert "printed output should equal 'HELLO'" in summary
    assert "Project imports from tests:" not in summary
    assert "Public calls used by tests:" not in summary
    assert "Existing test files:" not in summary
    assert "def build_parser" not in summary
    assert len(summary) < 700


def test_phase11a_summary_budget_remains_under_cap(tmp_path: Path):
    _write(tmp_path / "src" / "pkg" / "__init__.py", "")
    _write(
        tmp_path / "src" / "pkg" / "feature.py",
        "\n".join(f"def helper_{index}(): return {index}" for index in range(50)),
    )
    assertions = "\n".join(
        f"    assert helper_{index}() == {index}" for index in range(30)
    )
    _write(
        tmp_path / "tests" / "test_feature.py",
        "from pkg.feature import "
        + ", ".join(f"helper_{index}" for index in range(30))
        + "\n\n"
        + "def test_many_helpers():\n"
        + assertions
        + "\n",
    )

    contract = extract_python_test_contract(tmp_path)
    assert contract is not None

    summary = render_python_test_contract_summary(contract, max_chars=700)

    assert len(summary) <= 700
    assert "## TEST CONTRACT SUMMARY" in summary
    assert "src/pkg/feature.py" in summary


def test_phase11a_simplified_prompt_is_smaller_than_verbose_contract(tmp_path: Path):
    _write(tmp_path / "src" / "small_cli" / "__init__.py", "")
    _write(
        tmp_path / "src" / "small_cli" / "cli.py",
        "def build_parser():\n    return None\n\ndef main(argv=None):\n    return 0\n",
    )
    _write(
        tmp_path / "tests" / "test_cli.py",
        "from small_cli.cli import build_parser, main\n"
        "\n"
        "def test_uppercase_option_prints_uppercase_message(capsys):\n"
        "    assert main(['--uppercase', 'hello']) == 0\n"
        "    assert capsys.readouterr().out.strip() == 'HELLO'\n",
    )

    contract = extract_python_test_contract(tmp_path)
    assert contract is not None
    prompt = PlannerService.build_minimal_planning_prompt(
        "Add the --uppercase option to this small Python CLI.",
        tmp_path,
        workspace_has_existing_files=True,
    )

    old_verbose_floor = (
        len("Project imports from tests:\n")
        + sum(len(item) + 2 for item in contract.imports)
        + len("Public calls used by tests:\n")
        + sum(len(item) + 2 for item in contract.public_calls)
        + len("Existing test files:\n")
        + sum(len(item) + 2 for item in contract.test_files)
    )

    assert old_verbose_floor > 100
    assert len(python_test_source_context_from_tests(tmp_path)) < 700
    assert len(prompt) < 7000

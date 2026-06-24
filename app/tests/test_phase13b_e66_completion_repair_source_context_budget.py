"""Phase 13B-E66: Completion repair source context budget increase tests.

Verifies that MAX_SOURCE_CONTENT_PER_FILE_CHARS=2000 and
MAX_SOURCE_CONTENT_TOTAL_CHARS=5000 allow full inclusion of 1130–1300-char
cli.py files so that main() at byte ~883 is visible in the completion repair
prompt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.orchestration.phases.completion_repair_capsule import (
    MAX_SOURCE_CONTENT_PER_FILE_CHARS,
    MAX_SOURCE_CONTENT_TOTAL_CHARS,
    _SOURCE_TRUNCATED_MARKER,
    CompletionRepairCapsule,
    _read_bounded_source_contents,
    build_bounded_completion_repair_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_capsule(tmp_path: Path, **kwargs) -> CompletionRepairCapsule:
    defaults = dict(
        validation_reasons=["pytest failed"],
        relevant_files=["src/medium_cli/cli.py"],
        last_step_summary="Step 3: wrote cli.py - success. Files: src/medium_cli/cli.py.",
        workspace_path=str(tmp_path),
        task_prompt_excerpt="Add summary command to CLI",
    )
    defaults.update(kwargs)
    return CompletionRepairCapsule(**defaults)


def _build_1300_char_cli_py() -> str:
    """Build a synthetic cli.py of exactly 1300 chars with main() at byte 883."""
    preamble = "A" * 883
    main_func = "def main(argv: list[str] | None = None) -> int:\n    return 0\n"
    suffix = "B" * (1300 - 883 - len(main_func))
    content = preamble + main_func + suffix
    return content[:1300]


# ---------------------------------------------------------------------------
# E66 budget constants sanity check
# ---------------------------------------------------------------------------


def test_e66_per_file_cap_is_2000():
    assert MAX_SOURCE_CONTENT_PER_FILE_CHARS == 2000


def test_e66_total_cap_is_5000():
    assert MAX_SOURCE_CONTENT_TOTAL_CHARS == 5000


# ---------------------------------------------------------------------------
# E66: 1300-char cli.py included fully
# ---------------------------------------------------------------------------


def test_e66_1300_char_cli_py_included_fully(tmp_path):
    """A 1300-char cli.py (E65 workspace_cli_bytes) is included without truncation."""
    cli_content = _build_1300_char_cli_py()
    assert len(cli_content) == 1300
    _write(tmp_path / "src/medium_cli/cli.py", cli_content)
    contents = _read_bounded_source_contents(tmp_path, ["src/medium_cli/cli.py"])
    assert "src/medium_cli/cli.py" in contents
    stored = contents["src/medium_cli/cli.py"]
    assert _SOURCE_TRUNCATED_MARKER not in stored
    assert len(stored) == 1300
    assert stored == cli_content


# ---------------------------------------------------------------------------
# E66: content after byte 883 is visible
# ---------------------------------------------------------------------------


def test_e66_content_after_byte_883_is_visible(tmp_path):
    """Content at byte 883 is visible; E64 800-char cap excluded it, E66 2000-char cap includes it."""
    preamble = "A" * 883
    main_func = "def main(argv: list[str] | None = None) -> int:\n    return 0\n"
    cli_content = preamble + main_func
    # Total ~945 chars — within 2000-char cap but beyond the old 800-char cap.
    _write(tmp_path / "src/medium_cli/cli.py", cli_content)
    contents = _read_bounded_source_contents(tmp_path, ["src/medium_cli/cli.py"])
    assert "src/medium_cli/cli.py" in contents
    stored = contents["src/medium_cli/cli.py"]
    assert "def main(argv: list[str] | None = None) -> int:" in stored
    assert _SOURCE_TRUNCATED_MARKER not in stored


def test_e66_1130_char_cli_py_main_visible(tmp_path):
    """1130-char cli.py (E65 M06/M07 workspace size) is fully included."""
    preamble = "A" * 883
    main_func = "def main(argv: list[str] | None = None) -> int:\n    return 0\n"
    # Pad to exactly 1130 chars (883 preamble + 61 main_func + 186 padding).
    padding = "B" * (1130 - len(preamble) - len(main_func))
    cli_content = preamble + main_func + padding
    assert len(cli_content) == 1130
    _write(tmp_path / "src/medium_cli/cli.py", cli_content)
    contents = _read_bounded_source_contents(tmp_path, ["src/medium_cli/cli.py"])
    stored = contents["src/medium_cli/cli.py"]
    assert _SOURCE_TRUNCATED_MARKER not in stored
    assert len(stored) == 1130


# ---------------------------------------------------------------------------
# E66: per-file truncation still occurs above 2000 chars
# ---------------------------------------------------------------------------


def test_e66_per_file_truncation_occurs_above_2000_chars(tmp_path):
    """Files exceeding 2000 chars are still truncated at the per-file cap."""
    long_content = "y" * (MAX_SOURCE_CONTENT_PER_FILE_CHARS + 100)
    _write(tmp_path / "src/big.py", long_content)
    contents = _read_bounded_source_contents(tmp_path, ["src/big.py"])
    assert "src/big.py" in contents
    stored = contents["src/big.py"]
    assert stored.endswith(_SOURCE_TRUNCATED_MARKER)
    assert len(stored) == MAX_SOURCE_CONTENT_PER_FILE_CHARS + len(
        _SOURCE_TRUNCATED_MARKER
    )


def test_e66_2001_char_file_is_truncated(tmp_path):
    """A file of exactly 2001 chars is truncated to 2000 + marker."""
    content_2001 = "x" * 2001
    _write(tmp_path / "src/over.py", content_2001)
    contents = _read_bounded_source_contents(tmp_path, ["src/over.py"])
    stored = contents["src/over.py"]
    assert stored.endswith(_SOURCE_TRUNCATED_MARKER)
    assert len(stored) == 2000 + len(_SOURCE_TRUNCATED_MARKER)


def test_e66_2000_char_file_is_not_truncated(tmp_path):
    """A file of exactly 2000 chars fits within the per-file cap and is not truncated."""
    content_2000 = "x" * 2000
    _write(tmp_path / "src/exact.py", content_2000)
    contents = _read_bounded_source_contents(tmp_path, ["src/exact.py"])
    stored = contents["src/exact.py"]
    assert _SOURCE_TRUNCATED_MARKER not in stored
    assert len(stored) == 2000


# ---------------------------------------------------------------------------
# E66: total cap still applies at 5000 chars
# ---------------------------------------------------------------------------


def test_e66_total_cap_enforced_at_5000_chars(tmp_path):
    """Total source context across all files is bounded at 5000 chars."""
    # Three files of 2000 chars each = 6000 total, exceeds 5000-char cap.
    for i in range(3):
        _write(tmp_path / f"src/f{i}.py", "z" * MAX_SOURCE_CONTENT_PER_FILE_CHARS)
    paths = [f"src/f{i}.py" for i in range(3)]
    contents = _read_bounded_source_contents(tmp_path, paths)
    total_stored = sum(len(v) for v in contents.values())
    assert total_stored <= MAX_SOURCE_CONTENT_TOTAL_CHARS + len(
        _SOURCE_TRUNCATED_MARKER
    )


def test_e66_file_beyond_5000_total_is_skipped(tmp_path):
    """A file that would push the total beyond 5000 chars is skipped entirely."""
    chunk = "y" * (MAX_SOURCE_CONTENT_TOTAL_CHARS // 4)  # 1250 chars
    for i in range(4):
        _write(tmp_path / f"src/f{i}.py", chunk)
    _write(tmp_path / "src/late.py", "z = 1\n")
    paths = [f"src/f{i}.py" for i in range(4)] + ["src/late.py"]
    contents = _read_bounded_source_contents(tmp_path, paths)
    for i in range(4):
        assert f"src/f{i}.py" in contents
    assert "src/late.py" not in contents


# ---------------------------------------------------------------------------
# E66: full 1300-char cli.py visible in completion repair prompt
# ---------------------------------------------------------------------------


def test_e66_prompt_includes_full_1300_char_cli_content(tmp_path):
    """Completion repair prompt includes full content of 1300-char cli.py."""
    cli_content = _build_1300_char_cli_py()
    _write(tmp_path / "src/medium_cli/cli.py", cli_content)
    source_contents = _read_bounded_source_contents(tmp_path, ["src/medium_cli/cli.py"])
    capsule = _make_capsule(
        tmp_path,
        relevant_files=["src/medium_cli/cli.py"],
        source_file_contents=source_contents,
    )
    prompt = build_bounded_completion_repair_prompt(capsule, 4)
    assert "def main(argv: list[str] | None = None) -> int:" in prompt
    assert _SOURCE_TRUNCATED_MARKER not in prompt
    assert "CURRENT FILE CONTENT" in prompt
    assert "--- src/medium_cli/cli.py ---" in prompt


def test_e66_prompt_main_at_byte_883_visible(tmp_path):
    """Prompt contains main() text that starts at byte 883 (E65 failure scenario)."""
    preamble = "A" * 883
    main_sig = "def main(argv: list[str] | None = None) -> int:"
    cli_content = preamble + main_sig + "\n    return 0\n"
    _write(tmp_path / "src/medium_cli/cli.py", cli_content)
    source_contents = _read_bounded_source_contents(tmp_path, ["src/medium_cli/cli.py"])
    capsule = _make_capsule(
        tmp_path,
        relevant_files=["src/medium_cli/cli.py"],
        source_file_contents=source_contents,
    )
    prompt = build_bounded_completion_repair_prompt(capsule, 4)
    assert main_sig in prompt


# ---------------------------------------------------------------------------
# E66: ops_fix schema and rules preserved
# ---------------------------------------------------------------------------


def test_e66_prompt_remains_ops_fix_schema(tmp_path):
    """Prompt still requires ops_fix repair_type after E66 budget increase."""
    capsule = _make_capsule(tmp_path)
    prompt = build_bounded_completion_repair_prompt(capsule, 4)
    assert "ops_fix" in prompt


def test_e66_prompt_excludes_commands_key(tmp_path):
    """Prompt still excludes the 'commands' array after E66 budget increase."""
    capsule = _make_capsule(tmp_path)
    prompt = build_bounded_completion_repair_prompt(capsule, 4)
    assert '"commands": [' not in prompt


def test_e66_rule_12_character_for_character_present(tmp_path):
    """Rule 12 (character-for-character copy) is present after E66 budget increase."""
    capsule = _make_capsule(tmp_path)
    prompt = build_bounded_completion_repair_prompt(capsule, 4)
    assert "character-for-character" in prompt


def test_e66_rule_13_do_not_invent_present(tmp_path):
    """Rule 13 (do not invent/guess old text) is present after E66 budget increase."""
    capsule = _make_capsule(tmp_path)
    prompt = build_bounded_completion_repair_prompt(capsule, 4)
    assert "Do not invent" in prompt or "Do not guess" in prompt

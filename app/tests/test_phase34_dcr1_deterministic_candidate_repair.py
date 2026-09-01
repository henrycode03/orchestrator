"""PHASE34-DCR1 — deterministic candidate repair before LLM escalation.

Guards the one allowlisted repair class (``candidate_black_failed``) and the
false-positive boundary around it: nothing outside the allowlist is repaired
deterministically, nothing outside the accepted mutation authority is touched,
and a formatter that does not succeed never reports success.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.services.orchestration.phases import completion_deterministic_repair as dcr
from app.services.orchestration.phases.completion_deterministic_repair import (
    DETERMINISTIC_REPAIR_RULE_IDS,
    attempt_deterministic_candidate_repair,
    eligible_black_repair_paths,
)
from app.services.orchestration.types import (
    CandidateFinding,
    CandidateValidationResult,
)
from app.services.orchestration.validation.candidate_checks import (
    validate_candidate_delta,
)
from app.tests.phase33c4_test_helpers import executor_test_authority

UNFORMATTED = "def f(x):\n    return x*2\ndef g(y):\n    return y+1\n"
FORMATTED = "def f(x):\n    return x * 2\n\n\ndef g(y):\n    return y + 1\n"
PLAN = [{"step_number": 1, "verification": "python -m pytest"}]


def _authority(project_dir, paths):
    return executor_test_authority(
        project_dir,
        [{"op": "write_file", "path": path} for path in paths],
        plan=PLAN,
    )


def _black_finding(paths):
    return CandidateFinding(
        rule_id="candidate_black_failed",
        source="black",
        category="static",
        severity="error",
        attribution="candidate_introduced",
        repairable=True,
        message="Candidate-scoped black failed",
        evidence={"returncode": 1, "paths": list(paths)},
    )


def _verdict(findings):
    return CandidateValidationResult.from_findings(
        profile="implementation", findings=list(findings)
    )


def _candidate_run(project_dir, paths):
    change_set = {
        "added_files": list(paths),
        "modified_files": [],
        "deleted_files": [],
    }
    return validate_candidate_delta(
        project_dir=project_dir,
        change_set=change_set,
        plan=PLAN,
        task_prompt="task",
        include_static_checks=True,
        observed_scope=tuple(paths),
        verification_scope=tuple(paths),
    )


# --- 1. the proven positive case ------------------------------------------


def test_black_failure_on_authorized_changed_python_file_is_repaired(tmp_path):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    verdict = _verdict([_black_finding(["mod.py"])])

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=verdict,
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    assert outcome.status == "applied"
    assert outcome.paths == ("mod.py",)
    assert outcome.returncode == 0
    assert (tmp_path / "mod.py").read_text() == FORMATTED


def test_repair_clears_the_candidate_black_finding_on_revalidation(tmp_path):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    before = {f.rule_id for f in _candidate_run(tmp_path, ["mod.py"]).findings}
    assert "candidate_black_failed" in before

    attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    after = {f.rule_id for f in _candidate_run(tmp_path, ["mod.py"]).findings}
    assert "candidate_black_failed" not in after


# --- 2. authority boundary -------------------------------------------------


def test_black_failure_on_unauthorized_file_is_not_repaired(tmp_path):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    (tmp_path / "other.py").write_text(UNFORMATTED)
    # Authority covers mod.py only; the finding names an unauthorized path.
    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py", "other.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    assert outcome.status == "skipped"
    assert outcome.reason == "repair_path_outside_accepted_authority"
    assert (tmp_path / "mod.py").read_text() == UNFORMATTED
    assert (tmp_path / "other.py").read_text() == UNFORMATTED


def test_missing_accepted_path_authority_refuses_deterministic_repair(tmp_path):
    (tmp_path / "mod.py").write_text(UNFORMATTED)

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=None,
    )

    assert outcome.status == "skipped"
    assert outcome.reason == "accepted_path_authority_unavailable"
    assert (tmp_path / "mod.py").read_text() == UNFORMATTED


@pytest.mark.parametrize(
    "path_text",
    ["../escape.py", "/etc/passwd.py", ".agent/state.py", "nope.py", "notes.md"],
)
def test_non_product_safe_paths_are_refused(tmp_path, path_text):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    authority = _authority(tmp_path, ["mod.py"])

    paths, reason = eligible_black_repair_paths(
        findings=[_black_finding([path_text])],
        project_dir=tmp_path,
        accepted_path_authority=authority,
    )

    assert paths == ()
    assert reason in {
        "repair_path_not_product_safe",
        "repair_path_outside_accepted_authority",
    }


def test_symlinked_python_path_is_refused(tmp_path):
    outside = tmp_path.parent / "outside_target.py"
    outside.write_text(UNFORMATTED)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "mod.py").symlink_to(outside)

    paths, reason = eligible_black_repair_paths(
        findings=[_black_finding(["mod.py"])],
        project_dir=workspace,
        accepted_path_authority=_authority(workspace, ["mod.py"]),
    )

    assert paths == ()
    assert reason == "repair_path_not_product_safe"
    assert outside.read_text() == UNFORMATTED


def test_deterministic_repair_cannot_mint_path_authority(tmp_path):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    authority = _authority(tmp_path, ["mod.py"])
    grants_before = {str(grant.path) for grant in authority.grants}

    attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=authority,
    )

    assert {str(grant.path) for grant in authority.grants} == grants_before


# --- 3/4. the formatter itself must succeed --------------------------------


def test_unavailable_formatter_does_not_report_success(tmp_path, monkeypatch):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    monkeypatch.setattr(dcr, "_run_command", lambda **_: (1, "No module named black"))

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    assert outcome.status == "failed"
    assert outcome.reason == "deterministic_repair_command_unavailable"
    assert not outcome.applied


def test_timed_out_formatter_does_not_report_success(tmp_path, monkeypatch):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    monkeypatch.setattr(dcr, "_run_command", lambda **_: (None, "timed out"))

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    assert outcome.status == "failed"
    assert outcome.reason == "deterministic_repair_command_unavailable"
    assert not outcome.applied


def test_nonzero_formatter_exit_does_not_report_success(tmp_path, monkeypatch):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    monkeypatch.setattr(dcr, "_run_command", lambda **_: (123, "error: cannot format"))

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    assert outcome.status == "failed"
    assert outcome.reason == "deterministic_repair_command_failed"
    assert not outcome.applied


def test_syntax_error_file_is_not_shortcut_to_completion(tmp_path):
    (tmp_path / "mod.py").write_text("def broken(:\n    pass\n")

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    assert outcome.status == "failed"
    assert not outcome.applied
    findings = {f.rule_id for f in _candidate_run(tmp_path, ["mod.py"]).findings}
    assert "candidate_python_compile_failed" in findings


# --- 5/6. surviving findings still escalate --------------------------------


def test_surviving_flake8_finding_still_reaches_the_llm_repair_path(tmp_path):
    # Undefined name: black formats the file, flake8 still fails.
    (tmp_path / "mod.py").write_text("def f(x):\n    return undefined_name+x\n")

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )
    assert outcome.status == "applied"

    after = _candidate_run(tmp_path, ["mod.py"])
    verdict = _verdict(after.findings)
    assert "candidate_flake8_failed" in {f.rule_id for f in after.findings}
    assert verdict.repairable_findings, "unresolved finding must still escalate"


def test_failing_tests_after_formatting_still_block_completion(tmp_path):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    (tmp_path / "test_mod.py").write_text(
        "from mod import f\n\n\ndef test_f():\n    assert f(2) == 5\n"
    )
    paths = ["mod.py", "test_mod.py"]

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, paths),
    )
    assert outcome.status == "applied"

    verdict = _verdict(_candidate_run(tmp_path, paths).findings)
    assert verdict.status != "accepted"
    assert (
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_mod.py"],
            cwd=tmp_path,
            capture_output=True,
        ).returncode
        != 0
    )


# --- 7/8/9/10. nothing outside the allowlist -------------------------------


@pytest.mark.parametrize(
    "rule_id",
    [
        "candidate_flake8_failed",
        "focused_pytest_failed",
        "candidate_python_compile_failed",
        "candidate_diff_check_failed",
    ],
)
def test_non_allowlisted_findings_are_never_deterministically_repaired(
    tmp_path, rule_id
):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    finding = CandidateFinding(
        rule_id=rule_id,
        source="other",
        category="static",
        severity="error",
        attribution="candidate_introduced",
        repairable=True,
        message="Candidate-scoped check failed",
        evidence={"returncode": 1, "paths": ["mod.py"]},
    )

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([finding]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    assert outcome.status == "skipped"
    assert outcome.reason == "no_deterministic_repairable_finding"
    assert (tmp_path / "mod.py").read_text() == UNFORMATTED


def test_allowlist_is_exactly_one_rule():
    assert DETERMINISTIC_REPAIR_RULE_IDS == frozenset({"candidate_black_failed"})


def test_clean_candidate_never_invokes_the_formatter(tmp_path, monkeypatch):
    (tmp_path / "mod.py").write_text(FORMATTED)
    monkeypatch.setattr(
        dcr,
        "_run_command",
        lambda **_: pytest.fail("formatter must not run without a black finding"),
    )

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, ["mod.py"]),
    )

    assert outcome.status == "skipped"


def test_finding_without_path_evidence_is_refused(tmp_path):
    finding = CandidateFinding(
        rule_id="candidate_black_failed",
        source="black",
        category="static",
        severity="error",
        attribution="candidate_introduced",
        repairable=True,
        message="Candidate-scoped black failed",
        evidence={"returncode": 1},
    )

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([finding]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, []),
    )

    assert outcome.status == "skipped"
    assert outcome.reason == "no_repair_paths_in_finding_evidence"


# --- 9. one attempt per completion cycle -----------------------------------


def test_second_attempt_is_a_no_op_once_the_finding_is_cleared(tmp_path):
    (tmp_path / "mod.py").write_text(UNFORMATTED)
    authority = _authority(tmp_path, ["mod.py"])

    first = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(["mod.py"])]),
        project_dir=tmp_path,
        accepted_path_authority=authority,
    )
    assert first.status == "applied"

    revalidated = _verdict(_candidate_run(tmp_path, ["mod.py"]).findings)
    second = attempt_deterministic_candidate_repair(
        completion_validation=revalidated,
        project_dir=tmp_path,
        accepted_path_authority=authority,
    )
    assert second.status == "skipped"
    assert second.reason == "no_deterministic_repairable_finding"


def test_inaccessible_retained_changeset_is_treated_as_unavailable():
    class InaccessiblePath:
        def is_dir(self):
            raise PermissionError("fixture parent is not traversable")

    class UnreadableDirectoryPath:
        def is_dir(self):
            return True

        def iterdir(self):
            raise PermissionError("fixture entries are not readable")

    assert _retained_changeset_sources(InaccessiblePath()) is None
    assert _retained_changeset_sources(UnreadableDirectoryPath()) is None


# --- exact Task 230 shape ---------------------------------------------------


TASK230_CHANGE_SET = Path(
    "/root/.orchestrator/runtime/control/projects/111/change-sets/319/files"
)


def _retained_changeset_sources(path: Path) -> tuple[Path, ...] | None:
    try:
        if not path.is_dir():
            return None
        return tuple(path.iterdir())
    except OSError:
        return None


TASK230_SOURCES = _retained_changeset_sources(TASK230_CHANGE_SET)


@pytest.mark.skipif(
    TASK230_SOURCES is None, reason="retained Task 230 change set unavailable"
)
def test_task230_retained_changeset_closes_without_llm_repair(tmp_path):
    import shutil

    assert TASK230_SOURCES is not None
    for source in TASK230_SOURCES:
        shutil.copy2(source, tmp_path / source.name)
    paths = ["README.md", "temp_convert.py", "test_temp_convert.py"]
    authority = _authority(tmp_path, paths)

    before = _verdict(_candidate_run(tmp_path, paths).findings)
    assert {f.rule_id for f in before.repairable_findings} == {
        "candidate_black_failed",
        "candidate_flake8_failed",
    }

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=before,
        project_dir=tmp_path,
        accepted_path_authority=authority,
    )
    assert outcome.status == "applied"
    # README.md is not a Python path and is never handed to the formatter.
    assert outcome.paths == ("temp_convert.py", "test_temp_convert.py")

    after = _verdict(_candidate_run(tmp_path, paths).findings)
    assert after.repairable_findings == []
    assert after.status == "accepted"
    assert (
        subprocess.run(
            [sys.executable, "-m", "unittest", "test_temp_convert.py", "-v"],
            cwd=tmp_path,
            capture_output=True,
        ).returncode
        == 0
    )


# --- 13. publication identity must be the post-repair identity -------------


def test_publication_identity_equality_requires_post_repair_revalidation(tmp_path):
    """PUB1 equality holds only when the verdict is recomputed after the repair.

    The persisted change set is built after completion repair, so its digest is
    always post-format. A verdict carried over from before the repair would
    therefore fail the identity handoff — which is why revalidation is
    mandatory rather than optional.
    """
    from app.services.orchestration.validation.candidate_checks import (
        candidate_delta_identity,
    )

    (tmp_path / "mod.py").write_text(UNFORMATTED)
    paths = ["mod.py"]
    change_set = {
        "added_files": paths,
        "modified_files": [],
        "deleted_files": [],
        "target_path": str(tmp_path),
    }

    pre_repair_identity = candidate_delta_identity(change_set, project_dir=tmp_path)

    outcome = attempt_deterministic_candidate_repair(
        completion_validation=_verdict([_black_finding(paths)]),
        project_dir=tmp_path,
        accepted_path_authority=_authority(tmp_path, paths),
    )
    assert outcome.status == "applied"

    revalidated = _verdict(_candidate_run(tmp_path, paths).findings)
    revalidated.candidate_identity = candidate_delta_identity(
        change_set, project_dir=tmp_path
    )
    # The change set the lifecycle persists later, over the same repaired bytes.
    change_set_identity = candidate_delta_identity(change_set)

    assert pre_repair_identity != change_set_identity
    assert revalidated.candidate_identity == change_set_identity
    assert revalidated.accepted

"""Provider-free NPD1 characterization of new-project discovery contracts."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.orchestration.planning.read_only_discovery import (
    DISCOVERY_ADMISSION_REQUIRED,
    DISCOVERY_ADMISSION_SKIPPED,
    assess_discovery_admission,
)
from app.services.orchestration.planning.source_materialization import (
    SOURCE_STATUS_EXISTING,
    SOURCE_STATUS_MISSING,
    SOURCE_STATUS_NEW,
    materialize_planner_source_context,
)
from app.services.orchestration.planning.task_bootstrap_contract import (
    BootstrapTaskType,
    build_task1_bootstrap_contract,
)
from app.services.orchestration.validation.accepted_path_authority import (
    GrantClass,
    build_accepted_path_authority,
)
from app.services.orchestration.validation.path_authority import (
    PathAuthorityError,
    TrustClass,
    classify_trust,
    declare,
)
from app.services.orchestration.validation.validator import ValidatorService


TASK_229_TEXT = (
    "Create a useful local Python capability for converting temperatures between "
    "Celsius and Fahrenheit. Include a clean public conversion API, a simple "
    "command-line interface runnable with Python, automated tests covering normal "
    "values and boundary cases, and a short README with usage. Keep it "
    "deterministic and dependency-light; do not use secrets or external network "
    "services. Run the tests after implementation."
)


def _admission(root: Path, task: str):
    materialization = materialize_planner_source_context(
        root,
        task_description=task,
        supporting_paths=(),
    )
    return (
        assess_discovery_admission(
            prompt=task,
            planner_contract=None,
            materialization=materialization,
        ),
        materialization,
    )


def test_four_project_task_states_are_deterministic(tmp_path: Path):
    # CASE A: the actual R4 wording has creation intent but no authoritative
    # path or pre-discovery create-only contract.
    admission, materialization = _admission(tmp_path, TASK_229_TEXT)
    assert admission == type(admission)(
        DISCOVERY_ADMISSION_REQUIRED, "no_explicit_source_or_creation_path"
    )
    assert materialization.files == ()

    # CASE D: an absent alleged existing file must not become a creation grant.
    admission, materialization = _admission(
        tmp_path,
        "Update app/config.py with the existing settings behavior.",
    )
    assert admission.status == DISCOVERY_ADMISSION_REQUIRED
    assert admission.reason == "expected_source_status_not_grounded"
    record = materialization.file_map()["app/config.py"]
    assert record.status == SOURCE_STATUS_MISSING
    assert record.creation_authorized is False

    # CASE C: a concrete new path with creation wording is already sufficient
    # to reach Planning without existing-source discovery.
    admission, materialization = _admission(
        tmp_path,
        "Create app/new_feature.py with the requested helper.",
    )
    assert admission.status == DISCOVERY_ADMISSION_SKIPPED
    record = materialization.file_map()["app/new_feature.py"]
    assert record.status == SOURCE_STATUS_NEW
    assert record.creation_authorized is True

    # CASE B: a current existing target remains source-grounded without a
    # discovery call when the target is unique and materialized.
    existing = tmp_path / "app" / "services" / "worker.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("def run():\n    return 1\n", encoding="utf-8")
    admission, materialization = _admission(
        tmp_path,
        "Replace `run()` in app/services/worker.py.",
    )
    assert admission.status == DISCOVERY_ADMISSION_SKIPPED
    assert (
        materialization.file_map()["app/services/worker.py"].status
        == SOURCE_STATUS_EXISTING
    )


def test_task_229_has_no_safe_pre_discovery_create_only_fact():
    contract = build_task1_bootstrap_contract(
        plan=[],
        task_prompt=TASK_229_TEXT,
        existing_files=set(),
    )

    assert contract.bootstrap_task_type is BootstrapTaskType.UNKNOWN
    assert contract.expected_source_files == []
    assert contract.classification_evidence["has_source_intent"] is True
    assert contract.classification_evidence["source_paths"] == []


def test_new_file_plan_and_apa_need_creation_authority_not_existing_source(
    tmp_path: Path,
):
    task = "Create app/new_feature.py with the requested helper."
    materialization = materialize_planner_source_context(
        tmp_path,
        task_description=task,
        supporting_paths=(),
    )
    plan = [
        {
            "step_number": 1,
            "description": "Create the helper.",
            "commands": [],
            "verification": "python -m py_compile app/new_feature.py",
            "rollback": "rm -f app/new_feature.py",
            "expected_files": ["app/new_feature.py"],
            "ops": [
                {
                    "op": "write_file",
                    "path": "app/new_feature.py",
                    "content": "def helper():\n    return 1\n",
                }
            ],
        }
    ]

    verdict = ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=task,
        execution_profile="implementation",
        project_dir=tmp_path,
        source_materialization=materialization,
    )
    assert verdict.accepted, verdict.reasons

    authority, undeclarable = build_accepted_path_authority(
        plan=plan,
        source_materialization=materialization,
        creation_requested_paths={"app/new_feature.py"},
    )
    assert undeclarable == ()
    assert [(grant.path.value, grant.grant_class) for grant in authority.grants] == [
        ("app/new_feature.py", GrantClass.CREATION_AUTHORIZED)
    ]
    assert authority.grants[0].baseline_content_hash is None


def test_two_new_file_plan_gets_two_narrow_creation_grants(tmp_path: Path):
    task = "Create app/new_feature.py and app/cli.py as new standalone files."
    materialization = materialize_planner_source_context(
        tmp_path,
        task_description=task,
        supporting_paths=(),
    )
    paths = ["app/new_feature.py", "app/cli.py"]
    plan = [
        {
            "step_number": 1,
            "description": "Create both standalone files.",
            "commands": [],
            "verification": "python -m py_compile app/new_feature.py app/cli.py",
            "rollback": "rm -f app/new_feature.py app/cli.py",
            "expected_files": paths,
            "ops": [
                {
                    "op": "write_file",
                    "path": path,
                    "content": "value = 1\n",
                }
                for path in paths
            ],
        }
    ]

    verdict = ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt=task,
        execution_profile="implementation",
        project_dir=tmp_path,
        source_materialization=materialization,
    )
    assert verdict.accepted, verdict.reasons

    authority, undeclarable = build_accepted_path_authority(
        plan=plan,
        source_materialization=materialization,
        creation_requested_paths=set(paths),
    )
    assert undeclarable == ()
    assert {grant.path.value for grant in authority.grants} == set(paths)
    assert all(
        grant.grant_class is GrantClass.CREATION_AUTHORIZED
        and grant.baseline_content_hash is None
        for grant in authority.grants
    )


def test_bootstrap_ownership_characterization_is_explicit():
    # The current ownership contract excludes the OpenClaw scaffold and both
    # runtime metadata roots from product candidates. AGENTS.md is deliberately
    # a provenance-sensitive exception for legitimate project instructions; the
    # report records that separate caveat rather than silently changing it.
    internal = (
        "BOOTSTRAP.md",
        "HEARTBEAT.md",
        "IDENTITY.md",
        "SOUL.md",
        "TOOLS.md",
        "USER.md",
        "runtime.json",
    )
    for path in internal:
        assert classify_trust(declare(path)) is TrustClass.ORCHESTRATION_INTERNAL

    for path in ("AGENT.md", "AGENTS.md", "MEMORY.md"):
        assert classify_trust(declare(path)) is TrustClass.PRODUCT

    for path in (".agent/events/state.jsonl", ".openclaw/workspace-state.json"):
        try:
            declare(path)
        except PathAuthorityError as exc:
            assert exc.code == "path_protected_root"
        else:  # pragma: no cover - protects the assertion if policy changes
            raise AssertionError(f"protected runtime path was declared: {path}")

"""Provider-free PHASE36-GR2 grounding region-expansion tests.

GR1 proved that ``inspect_file`` returns a deterministic head-of-file slice
bounded by ``MAX_FILE_BYTES``, that its wire shape carries no offset, range,
line, or symbol field, and that the truncation flag and the structural symbol
map both existed on the observation contract without ever reaching the
provider.  A provider that selected exactly the right file could therefore
neither learn that it had seen a fraction of that file nor name the region it
had missed, and the only same-file move available to it returned byte-identical
content.

These tests pin the repaired contract using the two source shapes GR1
reconstructed: the A1 shape, whose target symbol begins at the truncation
boundary, and the ORD3 shape, whose target symbol lies far beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import pytest

from app.services.orchestration.planning.grounding import (
    GroundingAssessmentKind,
    GroundingCoordinator,
    GroundingExecutor,
    GroundingOutcome,
    GroundingProposal,
    GroundingRequestRejection,
    GroundingRunConfig,
    GroundingTaskReference,
    GroundingTerminalReason,
    POST_OBSERVATION_WIRE_EXAMPLES,
    render_grounding_state,
    render_post_observation_prompt,
)
from app.services.orchestration.planning.grounding.provider_adapter import (
    render_terminal_assessment_prompt,
)
from app.services.orchestration.planning.grounding.contracts import (
    MAX_FILE_BYTES,
    MAX_FILE_LINES,
    MAX_STRUCTURAL_SYMBOLS,
    is_substantive_observation,
    parse_grounding_request,
)


# The A1 shape, reproduced byte-exactly.  In the real run the bounded window
# ended at 4,077 bytes, mid-signature, on the ``limit: int = 100,`` line, with
# the ``page``/``per_page`` pagination body beginning two lines later and
# running well past the cut.  The padding below is sized so the window here
# ends the same way: after ``limit`` at 4,090 bytes, with ``page`` at 4,119.
_HEAD_PADDING = "".join(
    f"# bounded head padding line {index:04d} of the A1 source shape\n"
    for index in range(60)
)
_IMPORT_PADDING = "".join(
    f"from app.support.module_{index:03d} import helper_{index:03d}\n"
    for index in range(14)
)
_TAIL_PADDING = "".join(
    f"TAIL_CONSTANT_{index:03d} = 'far tail padding value {index:03d}'\n"
    for index in range(160)
)

A1_SOURCE = (
    _HEAD_PADDING
    + _IMPORT_PADDING
    + "\n\n"
    + "@router.get('/projects')\n"
    + "def get_projects(\n"
    + "    skip: int = 0,\n"
    + "    limit: int = 100,\n"
    # Everything from here lies outside the bounded window.
    + "    page: int | None = None,\n"
    + "    per_page: int = 25,\n"
    + "):\n"
    + "    if page is None:\n"
    + "        return query.offset(skip).limit(limit).all()\n"
    + "    return paginate(query, page, per_page)\n"
    + "\n\n"
    # The ORD3 shape: a second target far beyond the window, as update_project
    # was at lines 461-492 while the window ended at line 132.
    + _TAIL_PADDING
    + "\n"
    + "@router.put('/projects/{project_id}')\n"
    + "def update_project(payload):\n"
    + "    values = payload.model_dump(exclude_unset=True)\n"
    + "    for field, value in values.items():\n"
    + "        setattr(project, field, value)\n"
    + "    return project\n"
)


def _git_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, shell=False)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, shell=False)
    return tmp_path


def _a1_repo(tmp_path: Path) -> Path:
    return _git_repo(tmp_path, {"app/api/projects.py": A1_SOURCE})


def _request(payload: dict[str, object], request_id: str = "request-1"):
    return parse_grounding_request(
        payload, grounding_run_id="grounding-run-1", request_id=request_id
    )


def _execute(root: Path, payload: dict[str, object], request_id: str):
    return GroundingExecutor(root).execute(_request(payload, request_id))


def _inspect(root: Path, path: str = "app/api/projects.py", request_id="inspect-1"):
    return _execute(root, {"action": "inspect_file", "path": path}, request_id)


def _symbols(observation) -> dict[str, tuple[int, int]]:
    return {
        str(item["name"]): (int(item["start_line"]), int(item["end_line"]))
        for item in observation.structural_facts.get("top_level_symbols", ())
    }


@dataclass
class ScriptedProvider:
    steps: list

    def __post_init__(self):
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        return self.steps.pop(0)(context)


def _coordinator(root: Path, provider: ScriptedProvider, *, max_steps: int = 2):
    snapshot = "snapshot-gr2"
    config = GroundingRunConfig(
        grounding_run_id="grounding-run-gr2",
        task_reference=GroundingTaskReference(task_id="task-gr2"),
        workspace_identity=str(root.resolve()),
        snapshot_identity=snapshot,
        max_steps=max_steps,
        max_exploration_provider_requests=2,
        operator_task=(
            "Moving through the project list page by page should show each "
            "project exactly once, with the correct total count."
        ),
    )
    return GroundingCoordinator(
        executor=GroundingExecutor(root, snapshot_identity=snapshot),
        provider=provider,
        config=config,
    )


# --------------------------------------------------------------------------
# T1 — inspect_file remains bounded.  GR2 must not solve region expansion by
# enlarging the window.
# --------------------------------------------------------------------------
def test_t1_inspect_file_window_remains_bounded(tmp_path):
    root = _a1_repo(tmp_path)
    observation = _inspect(root)

    assert len(A1_SOURCE.encode("utf-8")) > MAX_FILE_BYTES
    assert len(observation.bounded_content) <= MAX_FILE_BYTES
    assert len(observation.bounded_content.splitlines()) <= MAX_FILE_LINES
    assert observation.truncated is True

    # Pin the fixture to the A1 shape so it cannot silently drift: the window
    # must stop mid-signature, leaving the pagination body outside it.
    content = observation.bounded_content.decode("utf-8")
    assert content.splitlines()[-1] == "    limit: int = 100,"
    assert "per_page" not in content
    assert _symbols(observation)["get_projects"][1] > len(content.splitlines())


# --------------------------------------------------------------------------
# T2 — truncation reaches provider-visible state.
# --------------------------------------------------------------------------
def test_t2_truncation_is_provider_visible(tmp_path):
    root = _a1_repo(tmp_path)
    captured: list = []

    def inspect(_context):
        return GroundingProposal(
            action_payload={"action": "inspect_file", "path": "app/api/projects.py"}
        )

    def stop(context):
        captured.append(context)
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.INSUFFICIENT,
            reason="recording provider-visible state",
        )

    _coordinator(root, ScriptedProvider([inspect, stop])).run()
    rendered = render_grounding_state(captured[0].state)
    payload = json.loads(rendered.split("\n", 1)[1])
    observation = payload["observation_history"][0]

    assert observation["truncated"] is True
    assert observation["returned_bytes"] == len(
        captured[0].state.observation_history[0].bounded_content
    )
    assert 0 < observation["returned_bytes"] <= MAX_FILE_BYTES
    assert '"truncated":true' in rendered


# --------------------------------------------------------------------------
# T3 — the structural locator map survives truncation.  This is the branch GR1
# found discarding structure at exactly the moment it was needed.
# --------------------------------------------------------------------------
def test_t3_structural_locator_survives_truncated_observation(tmp_path):
    root = _a1_repo(tmp_path)
    observation = _inspect(root)
    symbols = _symbols(observation)

    assert observation.truncated is True
    # The content window is still described as truncated; only navigation
    # metadata was added alongside it.
    assert observation.structural_facts["parse_status"] == "source_truncated"
    assert observation.structural_facts["structure_scope"] == "complete_file"
    assert "get_projects" in symbols
    assert "update_project" in symbols

    returned_lines = len(observation.bounded_content.splitlines())
    # Both targets are named even though neither body was returned in full.
    assert symbols["update_project"][0] > returned_lines


def test_t3b_non_python_truncation_degrades_to_the_previous_marker(tmp_path):
    root = _git_repo(tmp_path, {"frontend/app.tsx": "const x = 1;\n" * 800})
    observation = _inspect(root, "frontend/app.tsx", "inspect-tsx")

    assert observation.truncated is True
    assert observation.structural_facts["parse_status"] == "source_truncated"
    assert "top_level_symbols" not in observation.structural_facts


def test_t3c_structural_symbol_map_is_bounded(tmp_path):
    body = "".join(
        f"def symbol_{index:04d}():\n    return {index}\n" for index in range(300)
    )
    root = _git_repo(tmp_path, {"app/many.py": body})
    observation = _inspect(root, "app/many.py", "inspect-many")
    rendered_symbols = observation.structural_facts["top_level_symbols"]

    assert observation.truncated is True
    assert len(rendered_symbols) == MAX_STRUCTURAL_SYMBOLS
    assert observation.structural_facts["structure_truncated"] is True


# --------------------------------------------------------------------------
# T4 / T8 — the locator map is navigation metadata and never evidence.
# --------------------------------------------------------------------------
def test_t4_structural_metadata_is_not_substantive_evidence(tmp_path):
    root = _a1_repo(tmp_path)
    observation = _inspect(root)

    # Substantive status is decided by action identity, outcome and bounded
    # content.  The symbol map cannot reach that decision.
    assert is_substantive_observation(observation) is True

    stripped = observation.__class__(
        **{
            **{
                field: getattr(observation, field)
                for field in observation.__dataclass_fields__
            },
            "bounded_content": b"",
        }
    )
    assert stripped.structural_facts["top_level_symbols"]
    assert is_substantive_observation(stripped) is False


def test_t8_unseen_region_is_absent_from_the_truncated_observation(tmp_path):
    root = _a1_repo(tmp_path)
    observation = _inspect(root)
    content = observation.bounded_content.decode("utf-8")

    # The far symbol is named by the locator map but its implementation is not
    # carried as evidence by the head observation.
    assert "update_project" in _symbols(observation)
    assert "exclude_unset" not in content
    assert "setattr" not in content


# --------------------------------------------------------------------------
# T5 — the post-observation contract exposes the region action.
# --------------------------------------------------------------------------
def test_t5_post_observation_contract_exposes_resolve_structure(tmp_path):
    root = _a1_repo(tmp_path)
    captured: list = []

    def inspect(_context):
        return GroundingProposal(
            action_payload={"action": "inspect_file", "path": "app/api/projects.py"}
        )

    def stop(context):
        captured.append(context)
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.INSUFFICIENT, reason="capture"
        )

    _coordinator(root, ScriptedProvider([inspect, stop])).run()
    prompt = render_post_observation_prompt(captured[0])

    assert any(
        "resolve_structure" in example for example in POST_OBSERVATION_WIRE_EXAMPLES
    )
    assert '"action":"resolve_structure"' in prompt
    assert '"relation":"symbol_definition"' in prompt
    assert '"relation":"enclosing_symbol"' in prompt
    # The exact locator field sets, restated from the closed action schema.
    assert "symbol_definition locator fields are exactly path, name" in prompt
    assert "enclosing_symbol  locator fields are exactly path, line" in prompt
    assert (
        "mounted_route     locator fields are exactly path, method, decorator_path"
        in prompt
    )
    assert "truncated=true" in prompt
    assert "structural_locators" in prompt


def test_t5b_state_carries_locators_for_the_provider_to_aim(tmp_path):
    root = _a1_repo(tmp_path)
    captured: list = []

    def inspect(_context):
        return GroundingProposal(
            action_payload={"action": "inspect_file", "path": "app/api/projects.py"}
        )

    def stop(context):
        captured.append(context)
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.INSUFFICIENT, reason="capture"
        )

    _coordinator(root, ScriptedProvider([inspect, stop])).run()
    payload = json.loads(render_grounding_state(captured[0].state).split("\n", 1)[1])
    locators = payload["observation_history"][0]["structural_locators"]
    names = {item["name"] for item in locators}

    assert {"get_projects", "update_project"} <= names
    assert len(locators) <= MAX_STRUCTURAL_SYMBOLS
    for item in locators:
        assert set(item) == {"name", "kind", "start_line", "end_line"}


# --------------------------------------------------------------------------
# T6 / T7 — resolve_structure retrieves and promotes the unseen region.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "symbol, marker",
    [("get_projects", "per_page"), ("update_project", "exclude_unset")],
)
def test_t6_resolve_structure_retrieves_region_beyond_window(tmp_path, symbol, marker):
    root = _a1_repo(tmp_path)
    observation = _execute(
        root,
        {
            "action": "resolve_structure",
            "relation": "symbol_definition",
            "locator": {"path": "app/api/projects.py", "name": symbol},
        },
        f"resolve-{symbol}",
    )
    content = observation.bounded_content.decode("utf-8")

    assert observation.outcome is GroundingOutcome.FOUND
    assert observation.truncated is False
    assert marker in content
    assert observation.structural_identity is not None


def test_t7_resolved_region_is_promotable_substantive_evidence(tmp_path):
    root = _a1_repo(tmp_path)
    observation = _execute(
        root,
        {
            "action": "resolve_structure",
            "relation": "symbol_definition",
            "locator": {"path": "app/api/projects.py", "name": "get_projects"},
        },
        "resolve-promotable",
    )

    assert is_substantive_observation(observation) is True


def test_t6b_enclosing_symbol_relation_also_expands_same_file(tmp_path):
    root = _a1_repo(tmp_path)
    definition = _execute(
        root,
        {
            "action": "resolve_structure",
            "relation": "symbol_definition",
            "locator": {"path": "app/api/projects.py", "name": "update_project"},
        },
        "resolve-def",
    )
    inside = definition.structural_identity.start_line + 2
    enclosing = _execute(
        root,
        {
            "action": "resolve_structure",
            "relation": "enclosing_symbol",
            "locator": {"path": "app/api/projects.py", "line": inside},
        },
        "resolve-enclosing",
    )

    assert enclosing.outcome is GroundingOutcome.FOUND
    assert "exclude_unset" in enclosing.bounded_content.decode("utf-8")


# --------------------------------------------------------------------------
# T9 / T10 — sufficiency coverage.
# --------------------------------------------------------------------------
def test_t9_target_inside_truncated_window_can_still_ground(tmp_path):
    """A truncated observation is not disqualified: coverage, not truncation,
    is what matters.  Here the target really is inside the returned region."""

    root = _a1_repo(tmp_path)

    def inspect(_context):
        return GroundingProposal(
            action_payload={"action": "inspect_file", "path": "app/api/projects.py"}
        )

    def sufficient(context):
        observation = context.state.observation_history[0]
        assert observation.truncated is True
        assert b"helper_000" in observation.bounded_content
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.SUFFICIENT,
            cited_observation_ids=(observation.observation_id,),
            cited_source_paths=("app/api/projects.py",),
            rationale="The cited region contains the import being grounded.",
            unresolved_risk=False,
        )

    result = _coordinator(root, ScriptedProvider([inspect, sufficient])).run()

    assert result.terminal_reason is GroundingTerminalReason.SUFFICIENT


def test_t10_a1_shape_expands_to_the_unseen_region_within_two_actions(tmp_path):
    """The A1 sequence, under the unchanged two-action budget.

    Action 1 inspects the correct file and truncates.  Action 2 is the region
    expansion GR1 showed was unreachable, and the resolved region carries the
    implementation the head observation stopped short of.
    """

    root = _a1_repo(tmp_path)

    def inspect(_context):
        return GroundingProposal(
            action_payload={"action": "inspect_file", "path": "app/api/projects.py"}
        )

    def expand(context):
        observation = context.state.observation_history[0]
        assert observation.truncated is True
        # The provider aims at a symbol it can only know about from the map.
        assert "get_projects" in _symbols(observation)
        assert b"per_page" not in observation.bounded_content
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.NEED_MORE_EVIDENCE,
            rationale="The bounded window stopped before the implementation body.",
            action_payload={
                "action": "resolve_structure",
                "relation": "symbol_definition",
                "locator": {"path": "app/api/projects.py", "name": "get_projects"},
            },
        )

    def sufficient(context):
        region = context.state.observation_history[1]
        assert region.action_identity == "resolve_structure"
        assert b"per_page" in region.bounded_content
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.SUFFICIENT,
            cited_observation_ids=(region.observation_id,),
            cited_source_paths=("app/api/projects.py",),
            cited_structural_identities=(region.structural_identity,),
            rationale="The resolved region carries the pagination implementation.",
            unresolved_risk=False,
        )

    provider = ScriptedProvider([inspect, expand, sufficient])
    result = _coordinator(root, provider, max_steps=2).run()

    assert result.terminal_reason is GroundingTerminalReason.SUFFICIENT
    assert result.state_projection.observation_outcomes == (
        GroundingOutcome.FOUND.value,
        GroundingOutcome.FOUND.value,
    )
    # T14: accounting is unchanged by the repair.
    assert result.repository_action_count == 2
    assert result.budget_snapshot.repository_actions == 2
    assert result.budget_snapshot.distinct_files == 1
    assert result.budget_snapshot.positive_regions == 2


def test_t10b_ord3_shape_reaches_a_far_symbol_within_two_actions(tmp_path):
    """The ORD3 shape: the target lies far past the window, where a head-only
    citation previously produced premature sufficiency."""

    root = _a1_repo(tmp_path)

    def inspect(_context):
        return GroundingProposal(
            action_payload={"action": "inspect_file", "path": "app/api/projects.py"}
        )

    def expand(context):
        head = context.state.observation_history[0]
        start, _end = _symbols(head)["update_project"]
        assert start > len(head.bounded_content.splitlines())
        assert b"exclude_unset" not in head.bounded_content
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.NEED_MORE_EVIDENCE,
            rationale="The update implementation lies beyond the returned region.",
            action_payload={
                "action": "resolve_structure",
                "relation": "symbol_definition",
                "locator": {"path": "app/api/projects.py", "name": "update_project"},
            },
        )

    def sufficient(context):
        region = context.state.observation_history[1]
        assert b"exclude_unset" in region.bounded_content
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.SUFFICIENT,
            cited_observation_ids=(region.observation_id,),
            cited_source_paths=("app/api/projects.py",),
            cited_structural_identities=(region.structural_identity,),
            rationale="The resolved region carries the update implementation.",
            unresolved_risk=False,
        )

    result = _coordinator(
        root, ScriptedProvider([inspect, expand, sufficient]), max_steps=2
    ).run()

    assert result.terminal_reason is GroundingTerminalReason.SUFFICIENT


def test_t10c_sufficiency_coverage_rule_is_stated_on_every_sufficiency_turn(tmp_path):
    root = _a1_repo(tmp_path)
    captured: list = []

    def inspect(_context):
        return GroundingProposal(
            action_payload={"action": "inspect_file", "path": "app/api/projects.py"}
        )

    def stop(context):
        captured.append(context)
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.INSUFFICIENT, reason="capture"
        )

    _coordinator(root, ScriptedProvider([inspect, stop])).run()
    marker = "A file being relevant is not coverage"

    assert marker in render_post_observation_prompt(captured[0])
    assert marker in render_terminal_assessment_prompt(captured[0])


# --------------------------------------------------------------------------
# T14b — the durable journal can reconstruct the region-expansion sequence.
# --------------------------------------------------------------------------
def test_t14b_durable_events_reconstruct_truncation_and_expansion(tmp_path):
    root = _a1_repo(tmp_path)
    events: list[tuple[str, dict]] = []

    def inspect(_context):
        return GroundingProposal(
            action_payload={"action": "inspect_file", "path": "app/api/projects.py"}
        )

    def expand(_context):
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.NEED_MORE_EVIDENCE,
            rationale="The bounded window stopped before the implementation.",
            action_payload={
                "action": "resolve_structure",
                "relation": "symbol_definition",
                "locator": {"path": "app/api/projects.py", "name": "get_projects"},
            },
        )

    def sufficient(context):
        region = context.state.observation_history[1]
        return GroundingProposal(
            assessment_kind=GroundingAssessmentKind.SUFFICIENT,
            cited_observation_ids=(region.observation_id,),
            cited_source_paths=("app/api/projects.py",),
            cited_structural_identities=(region.structural_identity,),
            rationale="The resolved region carries the implementation.",
            unresolved_risk=False,
        )

    snapshot = "snapshot-gr2"
    config = GroundingRunConfig(
        grounding_run_id="grounding-run-gr2-events",
        task_reference=GroundingTaskReference(task_id="task-gr2"),
        workspace_identity=str(root.resolve()),
        snapshot_identity=snapshot,
        max_steps=2,
        max_exploration_provider_requests=2,
        operator_task="Ground the project list pagination behavior.",
    )
    GroundingCoordinator(
        executor=GroundingExecutor(root, snapshot_identity=snapshot),
        provider=ScriptedProvider([inspect, expand, sufficient]),
        config=config,
        event_sink=lambda kind, payload: events.append((kind, payload)),
    ).run()

    observations = [
        payload
        for kind, payload in events
        if str(kind).endswith("grounding_observation")
    ]

    assert len(observations) == 2
    assert observations[0]["action"] == "inspect_file"
    assert observations[0]["truncated"] is True
    assert observations[1]["action"] == "resolve_structure"
    assert observations[1]["truncated"] is False
    # GBA1: cumulative accounting still reconciles and is unchanged in shape.
    assert observations[1]["budget"]["repository_actions"] == 2
    assert observations[1]["budget"]["distinct_files"] == 1


# --------------------------------------------------------------------------
# T11 — malformed region actions still fail closed.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {
            "action": "resolve_structure",
            "relation": "symbol_definition",
            "locator": {"path": "app/api/projects.py", "line": 10},
        },
        {
            "action": "resolve_structure",
            "relation": "enclosing_symbol",
            "locator": {"path": "app/api/projects.py", "name": "get_projects"},
        },
        {
            "action": "resolve_structure",
            "relation": "symbol_definition",
            "locator": {"path": "app/api/projects.py", "name": "x", "extra": 1},
        },
        {
            "action": "resolve_structure",
            "relation": "not_a_relation",
            "locator": {"path": "app/api/projects.py", "name": "get_projects"},
        },
        {
            "action": "inspect_file",
            "path": "app/api/projects.py",
            "start_line": 130,
        },
    ],
)
def test_t11_malformed_region_actions_remain_rejected(tmp_path, payload):
    _a1_repo(tmp_path)
    with pytest.raises((GroundingRequestRejection, ValueError)):
        _request(payload, "malformed")


# --------------------------------------------------------------------------
# T12 — search and orientation authority semantics are untouched.
# --------------------------------------------------------------------------
def test_t12_search_remains_non_substantive_and_unchanged(tmp_path):
    root = _git_repo(tmp_path, {"app/api/projects.py": A1_SOURCE})
    observation = _execute(
        root,
        {"action": "search_text", "query": "per_page", "scopes": ["app"]},
        "search-1",
    )

    assert observation.outcome is GroundingOutcome.FOUND
    assert is_substantive_observation(observation) is False
    # Search still consumes no substantive file or region budget.
    assert observation.budget_delta.distinct_files == 0
    assert observation.budget_delta.positive_regions == 0
    assert observation.structural_facts == {"result_order": "path_line"}

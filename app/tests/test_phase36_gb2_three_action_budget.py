"""Provider-free PHASE36-GB2 three-action grounding budget tests.

PHASE36-GB1 adjudicated the typed-grounding action budget after the GR2/GR3
region-expansion repair removed the previously wasted second action.  It found
that two actions are exactly consumed by the minimum correct pair for a
cross-file behavior -- ``inspect_file`` then a same-file ``resolve_structure``
-- leaving nothing for the dependency the behavior actually delegates to, and
that a third action closes the gap.  It also found mechanically that raising
``TYPED_GROUNDING_MAX_STEPS`` alone is inert, because exploration provider
turns upper-bound repository actions and the reserved terminal turn can never
carry an action.

These tests pin the repaired shipped policy.  They drive the real coordinator
and the real executor with a deterministic stub provider, and they read the
limits from ``settings`` rather than parameterizing them, so a regression in
the shipped defaults fails here rather than only in production.

The source fixture reproduces the GR3 shape: an endpoint file whose target
symbol lies beyond the bounded head window and which delegates the graded
behavior to a symbol in a second file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess

import pytest

from app.config import settings
from app.services.orchestration.planning.grounding import (
    GroundingCoordinator,
    GroundingExecutor,
    GroundingLifecycleState,
    GroundingRunConfig,
    GroundingTaskReference,
    GroundingTerminalReason,
)
from app.services.orchestration.planning.grounding.contracts import (
    is_substantive_observation,
)
from app.services.orchestration.planning.grounding.coordinator_contracts import (
    GroundingProviderTurnMode,
)


# The GR3 endpoint shape.  The head window must cut before the pagination
# body, so that reaching ``get_projects`` needs a same-file region expansion
# and reaching ``paginate`` needs a second file.
_HEAD_PADDING = "".join(
    f"# bounded head padding line {index:04d} of the GR3 source shape\n"
    for index in range(60)
)
_IMPORT_PADDING = "".join(
    f"from app.support.module_{index:03d} import helper_{index:03d}\n"
    for index in range(14)
)

ENDPOINT_SOURCE = (
    _HEAD_PADDING
    + "from app.schemas.pagination import paginate\n"
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
    + "    query = db.query(Project).order_by(Project.created_at.desc())\n"
    + "    if page is None:\n"
    + "        return query.offset(skip).limit(limit).all()\n"
    + "    return paginate(query, page, per_page)\n"
)

# The second file.  ``paginate`` carries the offset arithmetic and the total
# computation that decide whether rows are skipped or repeated, and it
# delegates no further inside the product tree.
PAGINATION_SOURCE = (
    '"""Shared pagination abstractions."""\n'
    + "\n"
    + "import math\n"
    + "\n"
    + "\n"
    + "def paginate(query, page, per_page):\n"
    + "    total = query.count()\n"
    + "    offset = (page - 1) * per_page\n"
    + "    items = query.offset(offset).limit(per_page).all()\n"
    + "    total_pages = max(1, math.ceil(total / per_page)) if total > 0 else 1\n"
    + "    return {\n"
    + "        'items': items,\n"
    + "        'page': page,\n"
    + "        'per_page': per_page,\n"
    + "        'total': total,\n"
    + "        'total_pages': total_pages,\n"
    + "    }\n"
)

ENDPOINT_PATH = "app/api/v1/endpoints/projects.py"
PAGINATION_PATH = "app/schemas/pagination.py"

OPERATOR_TASK = (
    "Moving through the project list page by page should show each project "
    "exactly once, with the correct total count, without skipping or "
    "repeating projects."
)


def _git_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, shell=False)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, shell=False)
    return tmp_path


@pytest.fixture
def gr3_repo(tmp_path: Path) -> Path:
    return _git_repo(
        tmp_path,
        {ENDPOINT_PATH: ENDPOINT_SOURCE, PAGINATION_PATH: PAGINATION_SOURCE},
    )


INSPECT_ENDPOINT = {"action": "inspect_file", "path": ENDPOINT_PATH}
RESOLVE_GET_PROJECTS = {
    "action": "resolve_structure",
    "relation": "symbol_definition",
    "locator": {"path": ENDPOINT_PATH, "name": "get_projects"},
}
RESOLVE_PAGINATE = {
    "action": "resolve_structure",
    "relation": "symbol_definition",
    "locator": {"path": PAGINATION_PATH, "name": "paginate"},
}


def _need_more(action: dict):
    def step(_context):
        return {
            "decision": "NEED_MORE_EVIDENCE",
            "next_action": action,
            "rationale": "One bounded evidence action is still required.",
        }

    return step


def _sufficient(_context):
    observations = _context.state.observation_history
    return {
        "decision": "SUFFICIENT",
        "cited_observation_ids": [
            observation.observation_id
            for observation in observations
            if is_substantive_observation(observation)
        ],
        "rationale": "The cited bounded regions ground the paging behavior.",
    }


@dataclass
class ScriptedProvider:
    """Deterministic stub.  Never bypasses budget or contract enforcement."""

    steps: list
    provider_name: str = "gb2_scripted_provider"
    model_name: str = "unbound"
    turn_modes: list = field(default_factory=list)

    def decide(self, context):
        self.turn_modes.append(context.turn_mode)
        if not self.steps:
            raise AssertionError("coordinator requested an unbudgeted provider turn")
        return self.steps.pop(0)(context)


def _coordinator(
    root: Path,
    provider: ScriptedProvider,
    *,
    max_steps: int | None = None,
    max_exploration_provider_requests: int | None = None,
):
    """Build a run from the shipped defaults unless a test overrides them."""

    snapshot = "snapshot-gb2"
    config = GroundingRunConfig(
        grounding_run_id="grounding-run-gb2",
        task_reference=GroundingTaskReference(task_id="task-gb2"),
        workspace_identity=str(root.resolve()),
        snapshot_identity=snapshot,
        max_steps=(
            settings.TYPED_GROUNDING_MAX_STEPS if max_steps is None else max_steps
        ),
        max_exploration_provider_requests=(
            settings.TYPED_GROUNDING_MAX_PROVIDER_REQUESTS
            if max_exploration_provider_requests is None
            else max_exploration_provider_requests
        ),
        operator_task=OPERATOR_TASK,
    )
    return GroundingCoordinator(
        executor=GroundingExecutor(root, snapshot_identity=snapshot),
        provider=provider,
        config=config,
    )


# --------------------------------------------------------------------------
# B1 — the shipped defaults are the paired three-action policy.
# --------------------------------------------------------------------------
def test_b1_shipped_defaults_are_the_paired_three_action_policy():
    assert settings.TYPED_GROUNDING_MAX_STEPS == 3
    assert settings.TYPED_GROUNDING_MAX_PROVIDER_REQUESTS == 3
    # The pair must move together: exploration turns upper-bound repository
    # actions, so an unequal pair silently wastes the larger of the two.
    assert (
        settings.TYPED_GROUNDING_MAX_STEPS
        == settings.TYPED_GROUNDING_MAX_PROVIDER_REQUESTS
    )
    # Enablement is unchanged by the budget repair.
    assert settings.ENABLE_TYPED_GROUNDING_COORDINATOR is False


def test_b1b_reserved_allowances_are_unchanged_and_ceiling_is_exploration_plus_two():
    config = GroundingRunConfig(
        grounding_run_id="grounding-run-gb2",
        task_reference=GroundingTaskReference(task_id="task-gb2"),
        workspace_identity="/tmp/workspace",
        snapshot_identity="snapshot-gb2",
        max_steps=settings.TYPED_GROUNDING_MAX_STEPS,
        max_exploration_provider_requests=(
            settings.TYPED_GROUNDING_MAX_PROVIDER_REQUESTS
        ),
    )

    assert config.max_correction_provider_requests == 1
    assert config.max_terminal_assessment_requests == 1
    assert config.max_total_provider_requests == 5


# --------------------------------------------------------------------------
# B2 / B3 / B13 — the full three-action chain runs under the shipped defaults
# with no local override, and three exploration turns yield three actions.
# --------------------------------------------------------------------------
def test_b2_full_three_action_chain_grounds_the_cross_file_behavior(gr3_repo):
    provider = ScriptedProvider(
        [
            lambda _context: INSPECT_ENDPOINT,
            _need_more(RESOLVE_GET_PROJECTS),
            _need_more(RESOLVE_PAGINATE),
            _sufficient,
        ]
    )
    result = _coordinator(gr3_repo, provider).run()

    assert result.terminal_state is GroundingLifecycleState.SUFFICIENT
    assert result.terminal_reason is GroundingTerminalReason.SUFFICIENT

    budget = result.budget_snapshot
    assert budget.repository_actions == 3
    assert budget.distinct_files == 2
    assert budget.positive_regions == 3

    # Every action was productive: three substantive observations, all cited.
    observations = result.observations
    assert len(observations) == 3
    assert all(is_substantive_observation(item) for item in observations)
    assert len(result.cited_observation_ids) == 3

    # The third action reached the second file and returned an untruncated
    # region carrying the offset arithmetic the behavior turns on.
    third = observations[2]
    assert third.source_paths == (PAGINATION_PATH,)
    assert third.structural_identity is not None
    assert third.structural_identity.symbol_name == "paginate"
    assert third.truncated is False
    assert b"offset = (page - 1) * per_page" in third.bounded_content
    assert b"total = query.count()" in third.bounded_content

    # The first action was truncated, which is why expansion was needed at all.
    assert observations[0].truncated is True


def test_b3_three_exploration_turns_produce_three_repository_actions(gr3_repo):
    provider = ScriptedProvider(
        [
            lambda _context: INSPECT_ENDPOINT,
            _need_more(RESOLVE_GET_PROJECTS),
            _need_more(RESOLVE_PAGINATE),
            _sufficient,
        ]
    )
    result = _coordinator(gr3_repo, provider).run()

    assert provider.turn_modes == [
        GroundingProviderTurnMode.EXPLORATION,
        GroundingProviderTurnMode.EXPLORATION,
        GroundingProviderTurnMode.EXPLORATION,
        GroundingProviderTurnMode.TERMINAL_ASSESSMENT,
    ]

    budget = result.budget_snapshot
    assert budget.exploration_provider_requests == 3
    assert budget.correction_provider_requests == 0
    assert budget.terminal_assessment_requests == 1
    assert budget.repository_actions == 3
    # The partition identity must still hold exactly.
    assert budget.provider_requests == (
        budget.exploration_provider_requests
        + budget.correction_provider_requests
        + budget.terminal_assessment_requests
    )
    assert budget.provider_requests == 4


# --------------------------------------------------------------------------
# B4 — the third repository action must not steal the reserved terminal turn.
# --------------------------------------------------------------------------
def test_b4_third_action_does_not_consume_the_reserved_terminal_turn(gr3_repo):
    provider = ScriptedProvider(
        [
            lambda _context: INSPECT_ENDPOINT,
            _need_more(RESOLVE_GET_PROJECTS),
            _need_more(RESOLVE_PAGINATE),
            _sufficient,
        ]
    )
    result = _coordinator(gr3_repo, provider).run()

    # Exploration was fully spent, and a terminal turn was still available and
    # was taken as a distinct reserved turn after the third action executed.
    assert result.budget_snapshot.exploration_provider_requests == 3
    assert result.budget_snapshot.terminal_assessment_requests == 1
    assert provider.turn_modes[-1] is GroundingProviderTurnMode.TERMINAL_ASSESSMENT
    assert provider.turn_modes.count(GroundingProviderTurnMode.TERMINAL_ASSESSMENT) == 1
    # The terminal turn carried the assessment, not a fourth action.
    assert result.budget_snapshot.repository_actions == 3


# --------------------------------------------------------------------------
# B5 / B6 — the ceiling is a maximum, not a quota.  Raising it must not make
# the coordinator spend actions it does not need.
# --------------------------------------------------------------------------
def test_b5_early_sufficient_after_one_action_still_terminates(gr3_repo):
    provider = ScriptedProvider([lambda _context: INSPECT_ENDPOINT, _sufficient])
    result = _coordinator(gr3_repo, provider).run()

    assert result.terminal_state is GroundingLifecycleState.SUFFICIENT
    assert result.budget_snapshot.repository_actions == 1
    assert result.budget_snapshot.exploration_provider_requests == 2
    # No terminal allowance was needed, so it was never spent.
    assert result.budget_snapshot.terminal_assessment_requests == 0
    assert len(result.cited_observation_ids) == 1
    # The provider was not asked for a turn it did not need.
    assert provider.steps == []


def test_b6_early_sufficient_after_two_actions_still_terminates(gr3_repo):
    provider = ScriptedProvider(
        [
            lambda _context: INSPECT_ENDPOINT,
            _need_more(RESOLVE_GET_PROJECTS),
            _sufficient,
        ]
    )
    result = _coordinator(gr3_repo, provider).run()

    assert result.terminal_state is GroundingLifecycleState.SUFFICIENT
    assert result.budget_snapshot.repository_actions == 2
    assert result.budget_snapshot.distinct_files == 1
    # The third repository action was never executed.
    assert len(result.observations) == 2
    assert all(
        observation.source_paths == (ENDPOINT_PATH,)
        for observation in result.observations
    )
    # This is the ORD3 shape: inspect plus one same-file region is enough.
    assert result.budget_snapshot.terminal_assessment_requests == 0


# --------------------------------------------------------------------------
# B7 — the policy stays parameterized.  An explicit 2/2 run must reproduce the
# old bounded behavior exactly, so nothing now assumes three actions.
# --------------------------------------------------------------------------
def test_b7_explicit_two_action_override_reproduces_the_old_bound(gr3_repo):
    provider = ScriptedProvider(
        [
            lambda _context: INSPECT_ENDPOINT,
            _need_more(RESOLVE_GET_PROJECTS),
            _need_more(RESOLVE_PAGINATE),
            _sufficient,
        ]
    )
    result = _coordinator(
        gr3_repo,
        provider,
        max_steps=2,
        max_exploration_provider_requests=2,
    ).run()

    # The third action is unreachable: exploration is exhausted after two
    # turns and the terminal turn cannot carry an action.
    assert result.budget_snapshot.repository_actions <= 2
    assert result.budget_snapshot.repository_actions == 2
    assert result.budget_snapshot.exploration_provider_requests == 2
    assert provider.turn_modes == [
        GroundingProviderTurnMode.EXPLORATION,
        GroundingProviderTurnMode.EXPLORATION,
        GroundingProviderTurnMode.TERMINAL_ASSESSMENT,
    ]
    assert result.terminal_state is not GroundingLifecycleState.SUFFICIENT


# --------------------------------------------------------------------------
# B15 — the non-Python limitation is carried, not repaired.  A three-action
# budget must not be read as fixing frontend grounding coverage.
# --------------------------------------------------------------------------
def test_b15_non_python_region_expansion_remains_unavailable(tmp_path):
    from app.services.orchestration.planning.grounding.contracts import (
        parse_grounding_request,
    )
    from app.services.orchestration.planning.grounding.contracts import (
        GroundingExecutionError,
    )

    root = _git_repo(
        tmp_path, {"frontend/src/pages/ProjectsList.tsx": "const x = 1;\n" * 800}
    )
    executor = GroundingExecutor(root, snapshot_identity="snapshot-gb2")

    inspected = executor.execute(
        parse_grounding_request(
            {"action": "inspect_file", "path": "frontend/src/pages/ProjectsList.tsx"},
            grounding_run_id="grounding-run-gb2",
            request_id="request-1",
        )
    )
    assert inspected.truncated is True

    # Region expansion is Python-only, so extra budget buys this file nothing.
    with pytest.raises(GroundingExecutionError):
        executor.execute(
            parse_grounding_request(
                {
                    "action": "resolve_structure",
                    "relation": "symbol_definition",
                    "locator": {
                        "path": "frontend/src/pages/ProjectsList.tsx",
                        "name": "ProjectsList",
                    },
                },
                grounding_run_id="grounding-run-gb2",
                request_id="request-2",
            )
        )

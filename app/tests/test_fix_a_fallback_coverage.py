from __future__ import annotations

from app.services.orchestration.phases.planning_support import (
    _PlanningRetryState,
    _SECOND_REPAIR_BLOCKING_POLICIES,
    _get_targeted_second_repair_reason,
    _planning_root_cause_from_issue_key,
    _planning_root_cause_from_immediate_repair_issues,
    _REPLACE_FALLBACK_KEYS,
    _REPLACE_FALLBACK_INCOMPATIBLE_KEYS,
)


def _repaired_state() -> _PlanningRetryState:
    state = _PlanningRetryState()
    state.repair_prompt_used = True
    return state


# ---------------------------------------------------------------------------
# 1. empty_replace_old_text_steps repair eligibility
# ---------------------------------------------------------------------------


def test_empty_replace_has_policy_entry():
    assert "empty_replace_old_text_steps" in _SECOND_REPAIR_BLOCKING_POLICIES


def test_empty_replace_policy_emits_patch_strategy_fallback():
    policy = _SECOND_REPAIR_BLOCKING_POLICIES["empty_replace_old_text_steps"]
    assert policy.semantic_violation_code == "patch_strategy_fallback_required"


def test_empty_replace_policy_shares_stale_replace_cap():
    policy = _SECOND_REPAIR_BLOCKING_POLICIES["empty_replace_old_text_steps"]
    assert policy.cap_attribute == "post_repair_stale_replace_second_repair_used"


def test_empty_replace_sole_issue_triggers_second_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={"empty_replace_old_text_steps": [2]},
    )

    assert reason is not None
    assert reason.issue_key == "empty_replace_old_text_steps"
    assert reason.semantic_violation_code == "patch_strategy_fallback_required"
    assert reason.step_numbers == [2]
    assert reason.cap_used is False


def test_empty_replace_sole_issue_rejection_text_mentions_old_text():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={"empty_replace_old_text_steps": [1, 3]},
    )

    assert reason is not None
    assert "old text" in reason.rejection_text.lower()
    assert "write_file" in reason.rejection_text


def test_empty_replace_sole_issue_respects_cap():
    state = _repaired_state()
    state.post_repair_stale_replace_second_repair_used = True

    reason = _get_targeted_second_repair_reason(
        retry_state=state,
        blocking_repair_issues={"empty_replace_old_text_steps": [1]},
    )

    assert reason is not None
    assert reason.cap_used is True


def test_empty_replace_requires_prior_repair_pass():
    state = _PlanningRetryState()  # repair_prompt_used = False

    reason = _get_targeted_second_repair_reason(
        retry_state=state,
        blocking_repair_issues={"empty_replace_old_text_steps": [1]},
    )

    assert reason is None


# ---------------------------------------------------------------------------
# 2. Stale/empty replace fires with compatible co-issues
# ---------------------------------------------------------------------------


def test_stale_replace_with_weak_verification_triggers_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "weak_verification_steps": [2],
        },
    )

    assert reason is not None
    assert reason.issue_key == "stale_replace_ops_steps"
    assert reason.semantic_violation_code == "patch_strategy_fallback_required"


def test_stale_replace_with_background_process_triggers_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [3],
            "background_process_steps": [1],
        },
    )

    assert reason is not None
    assert reason.issue_key == "stale_replace_ops_steps"


def test_empty_replace_with_weak_verification_triggers_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "empty_replace_old_text_steps": [2],
            "weak_verification_steps": [3],
        },
    )

    assert reason is not None
    assert reason.issue_key == "empty_replace_old_text_steps"
    assert reason.semantic_violation_code == "patch_strategy_fallback_required"


def test_stale_replace_takes_priority_over_empty_replace_when_both_present():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "empty_replace_old_text_steps": [2],
        },
    )

    assert reason is not None
    assert reason.issue_key == "stale_replace_ops_steps"


def test_stale_replace_with_all_compatible_co_issues_triggers_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "empty_replace_old_text_steps": [2],
            "weak_verification_steps": [3],
            "background_process_steps": [4],
        },
    )

    assert reason is not None
    assert reason.issue_key == "stale_replace_ops_steps"


# ---------------------------------------------------------------------------
# 3. Incompatible issue combinations terminalize (return None from replace path)
# ---------------------------------------------------------------------------


def test_stale_replace_with_non_runnable_steps_does_not_trigger_replace_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "non_runnable_steps": [2],
        },
    )

    # non_runnable is incompatible — replace fallback should not fire.
    # Function may still return a non-None for other paths, but must NOT
    # return a replace-fallback reason.
    assert reason is None or reason.issue_key not in (
        "stale_replace_ops_steps",
        "empty_replace_old_text_steps",
    )


def test_stale_replace_with_placeholder_only_steps_does_not_trigger_replace_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "placeholder_only_steps": [2],
        },
    )

    assert reason is None or reason.issue_key not in (
        "stale_replace_ops_steps",
        "empty_replace_old_text_steps",
    )


def test_stale_replace_with_test_assertion_loss_does_not_trigger_replace_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "test_assertion_loss_ops_steps": [2],
        },
    )

    assert reason is None or reason.issue_key not in (
        "stale_replace_ops_steps",
        "empty_replace_old_text_steps",
    )


def test_stale_replace_with_test_deletion_does_not_trigger_replace_repair():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "test_deletion_ops_steps": [2],
        },
    )

    assert reason is None or reason.issue_key not in (
        "stale_replace_ops_steps",
        "empty_replace_old_text_steps",
    )


def test_incompatible_key_set_constants_are_correct():
    assert "non_runnable_steps" in _REPLACE_FALLBACK_INCOMPATIBLE_KEYS
    assert "placeholder_only_steps" in _REPLACE_FALLBACK_INCOMPATIBLE_KEYS
    assert "test_assertion_loss_ops_steps" in _REPLACE_FALLBACK_INCOMPATIBLE_KEYS
    assert "test_deletion_ops_steps" in _REPLACE_FALLBACK_INCOMPATIBLE_KEYS
    # weak_verification and background_process are compatible
    assert "weak_verification_steps" not in _REPLACE_FALLBACK_INCOMPATIBLE_KEYS
    assert "background_process_steps" not in _REPLACE_FALLBACK_INCOMPATIBLE_KEYS


# ---------------------------------------------------------------------------
# 4. Validator-path second repair preserves blocking context
# ---------------------------------------------------------------------------


def test_stale_replace_blocking_issues_consulted_when_plan_verdict_present():
    """Simulates validator-path call: blocking_repair_issues passed alongside plan_verdict.

    With the old code, passing blocking_repair_issues only fired when len == 1.
    After the fix, compatible multi-issue combinations also fire. This test
    verifies that blocking_repair_issues is consulted rather than lost.
    """
    # Simulate: validator-path call with stale replace + weak verification coexisting.
    # Previously this returned None because len > 1 and no plan_verdict path triggered it.
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "weak_verification_steps": [2],
        },
        plan_verdict=None,  # no validator findings — must rely on blocking_repair_issues
    )

    assert reason is not None
    assert reason.issue_key == "stale_replace_ops_steps"
    assert reason.semantic_violation_code == "patch_strategy_fallback_required"


def test_empty_replace_blocking_issues_consulted_when_plan_verdict_present():
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "empty_replace_old_text_steps": [3],
            "weak_verification_steps": [1],
        },
        plan_verdict=None,
    )

    assert reason is not None
    assert reason.issue_key == "empty_replace_old_text_steps"


# ---------------------------------------------------------------------------
# 5. Repair budget is unchanged
# ---------------------------------------------------------------------------


def test_empty_replace_repair_budget_shared_with_stale_replace():
    """Both keys share post_repair_stale_replace_second_repair_used cap."""
    stale_policy = _SECOND_REPAIR_BLOCKING_POLICIES["stale_replace_ops_steps"]
    empty_policy = _SECOND_REPAIR_BLOCKING_POLICIES["empty_replace_old_text_steps"]

    assert stale_policy.cap_attribute == empty_policy.cap_attribute


def test_stale_replace_multi_issue_cap_respected():
    state = _repaired_state()
    state.post_repair_stale_replace_second_repair_used = True

    reason = _get_targeted_second_repair_reason(
        retry_state=state,
        blocking_repair_issues={
            "stale_replace_ops_steps": [1],
            "weak_verification_steps": [2],
        },
    )

    # Cap is exhausted; cap_used must be True.
    assert reason is not None
    assert reason.cap_used is True
    assert reason.cap_attribute == "post_repair_stale_replace_second_repair_used"


def test_empty_replace_multi_issue_cap_respected():
    state = _repaired_state()
    state.post_repair_stale_replace_second_repair_used = True

    reason = _get_targeted_second_repair_reason(
        retry_state=state,
        blocking_repair_issues={
            "empty_replace_old_text_steps": [2],
            "weak_verification_steps": [3],
        },
    )

    assert reason is not None
    assert reason.cap_used is True


# ---------------------------------------------------------------------------
# 6. Root cause mapping updated for empty_replace
# ---------------------------------------------------------------------------


def test_root_cause_from_issue_key_maps_empty_replace_to_stale_replace():
    assert (
        _planning_root_cause_from_issue_key("empty_replace_old_text_steps")
        == "stale_replace"
    )


def test_root_cause_from_issue_key_maps_stale_replace_to_stale_replace():
    assert (
        _planning_root_cause_from_issue_key("stale_replace_ops_steps")
        == "stale_replace"
    )


def test_root_cause_from_immediate_issues_maps_empty_replace_to_stale_replace():
    assert (
        _planning_root_cause_from_immediate_repair_issues(
            {"empty_replace_old_text_steps": [1]}
        )
        == "stale_replace"
    )


def test_root_cause_from_immediate_issues_maps_stale_replace_to_stale_replace():
    assert (
        _planning_root_cause_from_immediate_repair_issues(
            {"stale_replace_ops_steps": [1]}
        )
        == "stale_replace"
    )


# ---------------------------------------------------------------------------
# 7. Regression: existing non-replace mixed classes still terminalize
# ---------------------------------------------------------------------------


def test_weak_verification_plus_background_process_still_terminalize():
    """Existing behavior unchanged: two non-replace blocking issues → None."""
    reason = _get_targeted_second_repair_reason(
        retry_state=_repaired_state(),
        blocking_repair_issues={
            "weak_verification_steps": [1],
            "background_process_steps": [2],
        },
    )

    assert reason is None


def test_replace_fallback_keys_constant_contains_both_keys():
    assert "stale_replace_ops_steps" in _REPLACE_FALLBACK_KEYS
    assert "empty_replace_old_text_steps" in _REPLACE_FALLBACK_KEYS

"""Focused authenticated API tests for Phase 28M operator review."""

from __future__ import annotations

from app.models import PlanningSession, Project
from app.services.planning.operator_review_persistence import OperatorReviewService

from app.tests.test_phase28l_operator_review_domain import _review_fixture


def _api_review_fixture(db_session):
    session, manifest, brief, candidate = _review_fixture(db_session)
    project = db_session.get(Project, session.project_id)
    project.user_id = 1
    session.status = "failed"
    review = OperatorReviewService(db_session).open_review_for_candidate(
        session.id, candidate.id
    )
    db_session.commit()
    return session, manifest, brief, candidate, review


def test_review_read_requires_authentication(api_client):
    response = api_client.get("/api/v1/planning/sessions/1/reviews")
    assert response.status_code == 401


def test_list_and_detail_return_exact_candidate_and_separate_authority(
    authenticated_client, db_session
):
    session, _manifest, brief, candidate, review = _api_review_fixture(db_session)

    listed = authenticated_client.get(f"/api/v1/planning/sessions/{session.id}/reviews")
    assert listed.status_code == 200
    item = listed.json()["items"][0]
    assert item["review_id"] == review.review_id
    assert item["candidate_checkpoint_id"] == candidate.id
    assert "candidate_content" not in item
    assert item["current_accepted_artifact"] is None

    detail = authenticated_client.get(
        f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}"
    )
    assert detail.status_code == 200
    body = detail.json()
    assert body["artifact_authority"] == "review_candidate"
    assert body["candidate_content"] == candidate.content
    assert body["candidate_content_hash"] == brief.content_hash
    assert body["candidate_binding"]["candidate_checkpoint_id"] == candidate.id
    assert body["validation_evidence"]["validation_hash"]
    assert body["event_history"][0]["event_type"] == "review_opened"


def test_approval_uses_exact_binding_is_atomic_and_replays(
    authenticated_client, db_session
):
    session, _manifest, brief, candidate, review = _api_review_fixture(db_session)
    detail = authenticated_client.get(
        f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}"
    ).json()
    action = {
        "binding": detail["candidate_binding"],
        "review_head_sequence": detail["current_event_sequence"],
        "review_head_token": detail["review_head_token"],
        "idempotency_key": "phase28m-approve-1",
        "comment": "The exact canonical candidate is approved unchanged.",
    }

    approved = authenticated_client.post(
        f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}/approve",
        json=action,
    )
    assert approved.status_code == 200, approved.text
    body = approved.json()
    assert body["decision"] == "approve_unchanged"
    assert body["promotion_checkpoint_id"]
    assert body["promotion_content_hash"] == candidate.content_hash
    assert body["current_accepted_artifact"]["artifact_authority"] == "accepted"
    assert body["completion_reevaluation"]["pending"] is True
    assert body["idempotent_replay"] is False

    replay = authenticated_client.post(
        f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}/approve",
        json=action,
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["promotion_checkpoint_id"] == body["promotion_checkpoint_id"]

    db_session.refresh(candidate)
    assert candidate.status == "failed"
    assert candidate.content_hash == brief.content_hash

    session_payload = authenticated_client.get(
        f"/api/v1/planning/sessions/{session.id}"
    ).json()
    assert session_payload["review_state"] == "accepted_after_review"
    assert (
        session_payload["accepted_brief_checkpoint_id"]
        == body["promotion_checkpoint_id"]
    )


def test_stale_head_and_unauthorized_project_are_rejected(
    authenticated_client, db_session
):
    session, _manifest, _brief, _candidate, review = _api_review_fixture(db_session)
    detail = authenticated_client.get(
        f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}"
    ).json()
    stale = {
        "binding": detail["candidate_binding"],
        "review_head_sequence": detail["current_event_sequence"],
        "review_head_token": "0" * 64,
        "idempotency_key": "phase28m-stale-head",
        "comment": "This must not approve with an old head.",
    }
    response = authenticated_client.post(
        f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}/approve",
        json=stale,
    )
    assert response.status_code == 412
    assert response.json()["detail"]["code"] == "review_head_stale"

    project = db_session.get(Project, session.project_id)
    project.user_id = 99
    db_session.commit()
    forbidden = authenticated_client.get(
        f"/api/v1/planning/sessions/{session.id}/reviews"
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["code"] == "review_forbidden"


def test_acknowledgment_is_nonterminal_and_rejection_is_terminal(
    authenticated_client, db_session
):
    session, _manifest, _brief, _candidate, review = _api_review_fixture(db_session)
    detail_url = f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}"
    detail = authenticated_client.get(detail_url).json()
    base = {
        "binding": detail["candidate_binding"],
        "review_head_sequence": detail["current_event_sequence"],
        "review_head_token": detail["review_head_token"],
    }
    acknowledged = authenticated_client.post(
        detail_url + "/acknowledge",
        json={**base, "idempotency_key": "phase28m-ack-1", "comment": "Noted."},
    )
    assert acknowledged.status_code == 200, acknowledged.text
    assert acknowledged.json()["review_state"] == "pending"

    updated = authenticated_client.get(detail_url).json()
    rejected = authenticated_client.post(
        detail_url + "/reject",
        json={
            "binding": updated["candidate_binding"],
            "review_head_sequence": updated["current_event_sequence"],
            "review_head_token": updated["review_head_token"],
            "idempotency_key": "phase28m-reject-1",
            "reason": "A regenerated candidate is required.",
        },
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["review_state"] == "rejected"

    after = authenticated_client.get(detail_url).json()
    approval = authenticated_client.post(
        detail_url + "/approve",
        json={
            "binding": after["candidate_binding"],
            "review_head_sequence": after["current_event_sequence"],
            "review_head_token": after["review_head_token"],
            "idempotency_key": "phase28m-too-late",
            "comment": "This must conflict after rejection.",
        },
    )
    assert approval.status_code == 409
    assert approval.json()["detail"]["code"] == "review_already_decided"


def test_regeneration_and_amendment_are_deferred_and_v1_is_hidden(
    authenticated_client, db_session
):
    session, _manifest, _brief, candidate, review = _api_review_fixture(db_session)
    detail_url = f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}"
    detail = authenticated_client.get(detail_url).json()
    common = {
        "binding": detail["candidate_binding"],
        "review_head_sequence": detail["current_event_sequence"],
        "review_head_token": detail["review_head_token"],
    }
    regeneration = authenticated_client.post(
        detail_url + "/regenerate",
        json={
            **common,
            "idempotency_key": "phase28m-regenerate-1",
            "reason": "The operator requested a fresh candidate.",
            "guidance": "Preserve the exact manifest lineage.",
        },
    )
    assert regeneration.status_code == 200, regeneration.text
    assert regeneration.json()["regeneration"]["provider_invoked"] is False
    assert regeneration.json()["regeneration"]["candidate_created"] is False

    # A fresh fixture avoids the terminal regeneration state for amendment.
    session, _manifest, _brief, candidate, review = _api_review_fixture(db_session)
    detail_url = f"/api/v1/planning/sessions/{session.id}/reviews/{review.review_id}"
    detail = authenticated_client.get(detail_url).json()
    amendment = authenticated_client.post(
        detail_url + "/amend",
        json={
            "binding": detail["candidate_binding"],
            "review_head_sequence": detail["current_event_sequence"],
            "review_head_token": detail["review_head_token"],
            "idempotency_key": "phase28m-amend-1",
            "target_kind": "planning_brief",
            "base_checkpoint_id": candidate.id,
            "base_checkpoint_hash": candidate.content_hash,
            "requested_change_kinds": ["clarify_scope"],
            "target_record_references": ["REQ-001"],
            "instruction": "Clarify the bounded requirement wording.",
            "regeneration_guidance": "Keep all unrelated records unchanged.",
            "reason": "The requirement needs a bounded amendment.",
        },
    )
    assert amendment.status_code == 200, amendment.text
    assert amendment.json()["amendment"]["artifact_amended"] is False
    assert amendment.json()["amendment"]["provider_invoked"] is False

    v1_project = Project(name="v1 review hidden", user_id=1)
    db_session.add(v1_project)
    db_session.flush()
    v1_session = PlanningSession(
        project_id=v1_project.id,
        title="v1",
        prompt="legacy",
        status="failed",
        protocol_version="v1",
    )
    db_session.add(v1_session)
    db_session.commit()
    hidden = authenticated_client.get(
        f"/api/v1/planning/sessions/{v1_session.id}/reviews"
    )
    assert hidden.status_code == 404

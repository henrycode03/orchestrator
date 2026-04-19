from __future__ import annotations

from app.models import Project, Session as SessionModel


def _seed_project_session(db_session):
    project = Project(name="Context Regression Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    session = SessionModel(
        project_id=project.id,
        name="Context Session",
        status="running",
        is_active=True,
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return project, session


def test_context_snapshot_and_state_follow_model_fields(
    authenticated_client, db_session
):
    project, session = _seed_project_session(db_session)

    save_response = authenticated_client.post(
        "/api/v1/context/snapshot",
        json={
            "session_id": session.id,
            "project_id": project.id,
            "current_step": 2,
            "total_steps": 5,
            "plan": [{"step_number": 1, "description": "Inspect current project"}],
        },
    )

    assert save_response.status_code == 200
    assert save_response.json() == {
        "success": True,
        "state_id": save_response.json()["state_id"],
        "current_step": 2,
        "total_steps": 5,
    }

    state_response = authenticated_client.get(f"/api/v1/context/state/{session.id}")
    assert state_response.status_code == 200
    payload = state_response.json()
    assert payload["exists"] is True
    assert payload["session_id"] == session.id
    assert payload["project_id"] == project.id
    assert payload["current_step"] == 2
    assert payload["total_steps"] == 5
    assert "state_version" not in payload
    assert "last_snapshot_at" not in payload


def test_context_conversation_uses_metadata_json_without_created_at_dependency(
    authenticated_client, db_session
):
    _, session = _seed_project_session(db_session)

    add_response = authenticated_client.post(
        "/api/v1/context/conversation",
        json={
            "session_id": session.id,
            "role": "assistant",
            "content": "Architecture inventory complete",
            "metadata": {"source": "regression-test"},
        },
    )

    assert add_response.status_code == 200
    assert add_response.json()["role"] == "assistant"

    history_response = authenticated_client.get(
        f"/api/v1/context/conversation/{session.id}"
    )
    assert history_response.status_code == 200
    payload = history_response.json()
    assert payload["count"] == 1
    assert payload["messages"][0]["metadata"] == {"source": "regression-test"}
    assert "created_at" not in payload["messages"][0]

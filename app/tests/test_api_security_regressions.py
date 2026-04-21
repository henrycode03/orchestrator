from app.models import Project, Session as SessionModel, Task, TaskStatus


def test_session_create_starts_pending_and_inactive(authenticated_client, db_session):
    project = Project(name="Session Security Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    response = authenticated_client.post(
        "/api/v1/sessions",
        json={
            "project_id": project.id,
            "name": "Fresh Session",
            "description": "Security regression coverage",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["is_active"] is False


def test_mobile_connection_secret_refuses_to_return_raw_secret(authenticated_client):
    response = authenticated_client.get("/api/v1/mobile-admin/connection-secret")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key"] is None
    assert payload["header_name"] == "X-OpenClaw-API-Key"
    assert payload["detail"] == "Raw mobile gateway secrets are not returned by the API"


def test_task_update_rejects_unsupported_fields(authenticated_client, db_session):
    project = Project(name="Task Security Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    task = Task(
        project_id=project.id,
        title="Contract Task",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    response = authenticated_client.put(
        f"/api/v1/tasks/{task.id}",
        json={"execution_profile": "debug_only"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported fields: ['execution_profile']"


def test_task_routes_require_authentication(api_client, db_session):
    project = Project(name="Anonymous Access Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    task = Task(
        project_id=project.id,
        title="Protected Task",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    response = api_client.get(f"/api/v1/tasks/{task.id}")

    assert response.status_code == 401


def test_session_routes_require_authentication(api_client, db_session):
    project = Project(name="Anonymous Session Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    session = SessionModel(project_id=project.id, name="Protected Session")
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)

    response = api_client.get(f"/api/v1/sessions/{session.id}")

    assert response.status_code == 401

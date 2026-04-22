"""Interactive planning session orchestration."""

from __future__ import annotations

import json
import re
import subprocess as _subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    Plan,
    PlanningArtifact,
    PlanningMessage,
    PlanningSession,
    Project,
    Task,
)
from app.schemas import PlannerTaskCandidate
from app.services.openclaw_service import OpenClawSessionError, OpenClawSessionService
from app.services.plan_commit_service import PlanCommitService
from app.services.planner_service import PlannerService
from app.services.performance_optimizations import optimize_prompt


class PlanningSessionService:
    """Manage resumable planning conversations and final plan synthesis."""

    ACTIVE_STATUSES = {"active", "waiting_for_input"}
    MAX_QUESTIONS = 2
    SYNTHESIS_TRANSCRIPT_CHAR_BUDGET = 1800
    SYNTHESIS_PROMPT_CHAR_BUDGET = 4200

    def __init__(self, db: Session):
        self.db = db

    def list_sessions(self, project_id: Optional[int] = None) -> list[PlanningSession]:
        query = self.db.query(PlanningSession).join(Project)
        query = query.filter(Project.deleted_at.is_(None))
        if project_id is not None:
            query = query.filter(PlanningSession.project_id == project_id)
        return query.order_by(
            PlanningSession.created_at.desc(), PlanningSession.id.desc()
        ).all()

    def get_session(self, session_id: int) -> PlanningSession:
        session = (
            self.db.query(PlanningSession)
            .join(Project)
            .filter(PlanningSession.id == session_id)
            .filter(Project.deleted_at.is_(None))
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="Planning session not found")
        return session

    def start_session(
        self, project: Project, prompt: str, source_brain: str = "local"
    ) -> PlanningSession:
        existing = (
            self.db.query(PlanningSession)
            .filter(
                PlanningSession.project_id == project.id,
                PlanningSession.status.in_(tuple(self.ACTIVE_STATUSES)),
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail="This project already has an active planning session",
            )

        session = PlanningSession(
            project_id=project.id,
            title=self._generate_title(prompt),
            prompt=prompt.strip(),
            status="active",
            source_brain=source_brain,
        )
        self.db.add(session)
        try:
            self.db.flush()
        except IntegrityError as exc:
            self.db.rollback()
            raise HTTPException(
                status_code=409,
                detail="This project already has an active planning session",
            ) from exc

        self._add_message(session, "user", prompt.strip(), metadata={"kind": "prompt"})
        self._advance_or_finalize(session, project)
        self.db.commit()
        self.db.refresh(session)
        return session

    def respond(self, session_id: int, response: str) -> PlanningSession:
        session = self.get_session(session_id)
        if session.status != "waiting_for_input" or not session.current_prompt_id:
            raise HTTPException(
                status_code=409, detail="Planning session is not waiting for input"
            )

        project = session.project
        self._add_message(
            session,
            "user",
            response.strip(),
            prompt_id=session.current_prompt_id,
            metadata={"kind": "response"},
        )
        session.current_prompt_id = None
        session.status = "active"
        session.updated_at = datetime.now(timezone.utc)
        self._advance_or_finalize(session, project)
        self.db.commit()
        self.db.refresh(session)
        return session

    def cancel(self, session_id: int) -> PlanningSession:
        session = self.get_session(session_id)
        if session.status not in {"completed", "cancelled"}:
            session.status = "cancelled"
            session.current_prompt_id = None
            session.updated_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(session)
        return session

    def commit(
        self,
        session_id: int,
        selected_tasks: Optional[list[PlannerTaskCandidate]] = None,
        planner_markdown: Optional[str] = None,
    ) -> tuple[PlanningSession, Optional[Plan], list[Task]]:
        session = self.get_session(session_id)
        if session.status != "completed":
            raise HTTPException(
                status_code=409,
                detail="Planning session must be completed before commit",
            )

        committed_task_ids = self._load_committed_task_ids(session)
        if committed_task_ids:
            tasks = (
                self.db.query(Task)
                .filter(Task.id.in_(committed_task_ids))
                .order_by(Task.plan_position.asc(), Task.id.asc())
                .all()
            )
            return session, session.finalized_plan, tasks

        effective_markdown = (
            planner_markdown.strip() if planner_markdown is not None else None
        )
        if effective_markdown is None:
            effective_markdown = self._get_artifact_content(session, "planner_markdown")
        if not effective_markdown:
            raise HTTPException(
                status_code=422,
                detail="Planning session is missing final planner markdown",
            )

        task_candidates = (
            PlannerService.parse_markdown(effective_markdown)
            if selected_tasks is None
            else selected_tasks
        )
        included_tasks = [
            task
            for task in task_candidates
            if getattr(task, "include", True) and (task.title or "").strip()
        ]
        if not included_tasks:
            raise HTTPException(
                status_code=422,
                detail="At least one task must be selected for commit",
            )

        plan = session.finalized_plan
        if plan is None:
            title = session.title[:255]
            requirement = (
                self._get_artifact_content(session, "requirements") or session.prompt
            )
            plan = Plan(
                project_id=session.project_id,
                title=title,
                source_brain=session.source_brain,
                requirement=requirement,
                markdown=effective_markdown,
                status="draft",
            )
            self.db.add(plan)
            self.db.flush()
            session.finalized_plan_id = plan.id
        else:
            plan.markdown = effective_markdown

        if planner_markdown is not None:
            self._upsert_artifact(
                session,
                artifact_type="planner_markdown",
                filename="planner.md",
                content=effective_markdown,
            )

        committed_plan, tasks = PlanCommitService(self.db).create_plan_tasks(
            session.project,
            included_tasks,
            plan=plan,
            commit=False,
        )
        session.committed_at = datetime.now(timezone.utc)
        session.committed_task_ids = json.dumps([task.id for task in tasks])
        session.finalized_plan_id = (
            committed_plan.id if committed_plan else session.finalized_plan_id
        )
        self.db.flush()
        self.db.commit()
        self.db.refresh(session)
        if committed_plan:
            self.db.refresh(committed_plan)
        for task in tasks:
            self.db.refresh(task)
        return session, committed_plan, tasks

    def build_session_payload(self, session: PlanningSession) -> dict[str, Any]:
        return {
            "id": session.id,
            "project_id": session.project_id,
            "title": session.title,
            "prompt": session.prompt,
            "status": session.status,
            "source_brain": session.source_brain,
            "current_prompt_id": session.current_prompt_id,
            "finalized_plan_id": session.finalized_plan_id,
            "committed_at": session.committed_at,
            "completed_at": session.completed_at,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "last_error": session.last_error,
            "messages": session.messages,
            "artifacts": session.artifacts,
            "tasks_preview": [
                PlannerTaskCandidate(
                    title=item.title,
                    description=item.description,
                    execution_profile=item.execution_profile,
                    priority=item.priority,
                    plan_position=item.plan_position,
                    estimated_effort=item.estimated_effort,
                )
                for item in PlannerService.parse_markdown(
                    self._get_artifact_content(session, "planner_markdown") or ""
                )
            ],
            "committed_task_ids": self._load_committed_task_ids(session),
        }

    def _advance_or_finalize(self, session: PlanningSession, project: Project) -> None:
        question_count = len([m for m in session.messages if m.role == "assistant"])
        if self._needs_clarification(session) and question_count < self.MAX_QUESTIONS:
            question = self._next_question(session)
            prompt_id = f"prompt-{uuid.uuid4().hex[:12]}"
            session.status = "waiting_for_input"
            session.current_prompt_id = prompt_id
            session.updated_at = datetime.now(timezone.utc)
            self._add_message(
                session,
                "assistant",
                question,
                prompt_id=prompt_id,
                metadata={"kind": "clarifying_question"},
            )
            return

        self._finalize_session(session, project)

    def _finalize_session(self, session: PlanningSession, project: Project) -> None:
        prompt = self._build_synthesis_prompt(session, project)
        try:
            result = self._run_openclaw_with_fallback(prompt)
            artifacts = self._parse_finalization_payload(result)
        except HTTPException:
            raise
        except Exception as exc:
            session.status = "failed"
            session.last_error = str(exc)
            session.current_prompt_id = None
            session.updated_at = datetime.now(timezone.utc)
            return

        planner_markdown = artifacts.get("planner_markdown", "")
        parsed_tasks = PlannerService.parse_markdown(planner_markdown)
        if not planner_markdown or not parsed_tasks:
            session.status = "failed"
            session.last_error = (
                "Planning synthesis did not produce parseable planner markdown"
            )
            session.current_prompt_id = None
            session.updated_at = datetime.now(timezone.utc)
            return

        self.db.query(PlanningArtifact).filter(
            PlanningArtifact.planning_session_id == session.id
        ).delete(synchronize_session=False)

        artifact_specs = {
            "requirements": ("requirements.md", artifacts.get("requirements", "")),
            "design": ("design.md", artifacts.get("design", "")),
            "implementation_plan": (
                "implementation_plan.md",
                artifacts.get("implementation_plan", ""),
            ),
            "planner_markdown": ("planner.md", planner_markdown),
        }
        for artifact_type, (filename, content) in artifact_specs.items():
            self.db.add(
                PlanningArtifact(
                    planning_session_id=session.id,
                    artifact_type=artifact_type,
                    filename=filename,
                    content=content.strip(),
                )
            )

        self._add_message(
            session,
            "assistant",
            "Planning complete. Review the artifacts and task preview, then commit when ready.",
            metadata={"kind": "completion"},
        )
        session.status = "completed"
        session.current_prompt_id = None
        session.completed_at = datetime.now(timezone.utc)
        session.updated_at = datetime.now(timezone.utc)
        session.last_error = None

    def _run_openclaw(self, prompt: str) -> dict[str, Any]:
        """Execute OpenClaw synchronously via subprocess to avoid asyncio threading issues."""
        service = OpenClawSessionService(self.db, session_id=None)
        try:
            cmd = service._resolve_openclaw_command()
        except OpenClawSessionError as exc:
            raise RuntimeError(str(exc))

        planning_key = f"planning-{uuid.uuid4().hex[:12]}"
        full_cmd = [
            *cmd,
            "agent",
            "--local",
            "--session-id",
            planning_key,
            "--message",
            prompt,
            "--json",
            "--timeout",
            "180",
        ]
        try:
            proc = _subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=210,
            )
        except _subprocess.TimeoutExpired:
            raise RuntimeError("Planning synthesis timed out after 180s")
        return service._parse_openclaw_response(proc)

    def _run_openclaw_with_fallback(self, prompt: str) -> dict[str, Any]:
        result = self._run_openclaw(prompt)
        if not OpenClawSessionService._is_context_overflow_result(result):
            return result

        compact_prompt = self._build_compact_synthesis_prompt(prompt)
        if compact_prompt == prompt:
            return result
        return self._run_openclaw(compact_prompt)

    def _parse_finalization_payload(self, result: dict[str, Any]) -> dict[str, str]:
        if result.get("status") == "failed":
            raise RuntimeError(result.get("error") or "Planning synthesis failed")

        output_text = result.get("output", "")
        if isinstance(output_text, str):
            try:
                parsed_output = json.loads(output_text)
                if isinstance(parsed_output, dict) and "payloads" in parsed_output:
                    payloads = parsed_output.get("payloads") or []
                    if payloads and isinstance(payloads[0], dict):
                        output_text = payloads[0].get("text", output_text)
            except json.JSONDecodeError:
                pass

        if not isinstance(output_text, str):
            raise RuntimeError("Planning synthesis returned unsupported output")

        cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", output_text.strip())
        parsed = json.loads(cleaned)
        required_keys = {
            "requirements",
            "design",
            "implementation_plan",
            "planner_markdown",
        }
        if not isinstance(parsed, dict) or not required_keys.issubset(parsed):
            raise RuntimeError("Planning synthesis returned malformed artifact payload")
        return {key: str(parsed.get(key, "")).strip() for key in required_keys}

    def _build_synthesis_prompt(
        self, session: PlanningSession, project: Project
    ) -> str:
        transcript = self._build_condensed_transcript(session)
        project_description = self._trim_text(project.description or "", 280)
        prompt = f"""You are creating implementation-planning artifacts for a software project.

Project: {project.name}
Project description: {project_description or "None provided"}
Planning prompt: {self._trim_text(session.prompt, 600)}

Conversation transcript:
{transcript}

Return JSON only with exactly these keys:
- requirements
- design
- implementation_plan
- planner_markdown

Artifact requirements:
1. requirements must be markdown with goals, scope, constraints, and acceptance criteria.
2. design must be markdown with architecture, interfaces, data flow, and risks.
3. implementation_plan must be markdown with ordered steps and test strategy.
4. planner_markdown must be markdown compatible with an Orchestrator task list using:
   ## Task List
   - [ ] TASK_START: Title | Description | order=1 | P1 | effort=medium | profile=full_lifecycle
5. planner_markdown must contain between 3 and 8 concrete tasks.
6. Prefer relative implementation detail grounded in the prompt and transcript.
7. Do not include prose outside the JSON object.
"""
        return optimize_prompt(
            prompt,
            max_tokens=1400,
            hard_char_limit=self.SYNTHESIS_PROMPT_CHAR_BUDGET,
        )

    def _build_compact_synthesis_prompt(self, prompt: str) -> str:
        compact = optimize_prompt(prompt, max_tokens=700, hard_char_limit=2200)
        compact += "\n\nKeep every artifact concise. Prefer short markdown sections and compact task descriptions."
        return compact

    def _needs_clarification(self, session: PlanningSession) -> bool:
        responses = [m for m in session.messages if m.role == "user"][1:]
        if responses:
            combined = " ".join(message.content for message in responses)
            return len(combined.split()) < 8

        prompt = session.prompt.lower()
        if len(prompt.split()) < 10:
            return True
        strong_detail_markers = (
            "api",
            "database",
            "frontend",
            "backend",
            "auth",
            "mobile",
            "websocket",
            "test",
            "integration",
            "migration",
            "dashboard",
        )
        return sum(marker in prompt for marker in strong_detail_markers) < 2

    def _next_question(self, session: PlanningSession) -> str:
        responses = [m for m in session.messages if m.role == "user"][1:]
        if not responses:
            return (
                "What outcome should this planning session optimize for, and are there any "
                "must-keep constraints around users, integrations, or rollout?"
            )
        return (
            "What acceptance criteria or implementation constraints would make this plan "
            "feel complete enough to execute safely?"
        )

    def _add_message(
        self,
        session: PlanningSession,
        role: str,
        content: str,
        *,
        prompt_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PlanningMessage:
        message = PlanningMessage(
            planning_session_id=session.id,
            role=role,
            prompt_id=prompt_id,
            content=content.strip(),
            metadata_json=metadata,
        )
        self.db.add(message)
        session.messages.append(message)
        return message

    def _generate_title(self, prompt: str) -> str:
        normalized = " ".join((prompt or "").split())
        return normalized[:57] + "..." if len(normalized) > 60 else normalized

    def _build_condensed_transcript(self, session: PlanningSession) -> str:
        rendered: list[str] = []
        total_chars = 0
        for message in session.messages[-6:]:
            speaker = "Planner" if message.role == "assistant" else "User"
            line = f"{speaker}: {self._trim_text(message.content, 260)}"
            if (
                total_chars + len(line) > self.SYNTHESIS_TRANSCRIPT_CHAR_BUDGET
                and rendered
            ):
                break
            rendered.append(line)
            total_chars += len(line)
        if not rendered:
            return "No conversation transcript available."
        return "\n".join(rendered)

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def _get_artifact_content(
        self, session: PlanningSession, artifact_type: str
    ) -> Optional[str]:
        for artifact in session.artifacts:
            if artifact.artifact_type == artifact_type:
                return artifact.content
        return None

    def _upsert_artifact(
        self,
        session: PlanningSession,
        *,
        artifact_type: str,
        filename: str,
        content: str,
    ) -> None:
        existing = next(
            (
                artifact
                for artifact in session.artifacts
                if artifact.artifact_type == artifact_type
            ),
            None,
        )
        if existing:
            existing.filename = filename
            existing.content = content.strip()
            return

        artifact = PlanningArtifact(
            planning_session_id=session.id,
            artifact_type=artifact_type,
            filename=filename,
            content=content.strip(),
        )
        self.db.add(artifact)
        session.artifacts.append(artifact)

    def _load_committed_task_ids(self, session: PlanningSession) -> list[int]:
        if not session.committed_task_ids:
            return []
        try:
            parsed = json.loads(session.committed_task_ids)
        except json.JSONDecodeError:
            return []
        return [
            int(item) for item in parsed if isinstance(item, int) or str(item).isdigit()
        ]

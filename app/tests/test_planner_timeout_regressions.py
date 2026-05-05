import json

from app.services.orchestration.phases.planning_flow import (
    _compress_project_context_for_planning,
    _should_repair_truncated_single_step_plan,
)
from app.services.orchestration.planning.planner import (
    PlannerService,
    PlanningRepairBudgetExceeded,
    PLANNING_REPAIR_PROMPT_MAX_CHARS,
)
from app.services.orchestration.validation.validator import ValidatorService
from app.services.orchestration.policy import (
    MINIMAL_PLANNING_TIMEOUT_SECONDS,
    ORCHESTRATION_TASK_SOFT_TIME_LIMIT_SECONDS,
    ORCHESTRATION_TASK_TIME_LIMIT_SECONDS,
    PLANNING_REPAIR_TIMEOUT_SECONDS,
    ULTRA_MINIMAL_PLANNING_TIMEOUT_SECONDS,
)
from app.tasks.worker import execute_orchestration_task


def test_build_task_with_clean_architecture_does_not_start_minimal_first():
    assert (
        PlannerService.should_start_with_minimal_prompt(
            "Set up frontend (React or Vite) and backend (Node.js or FastAPI) with clean architecture.",
            "",
        )
        is False
    )


def test_true_inspection_task_still_starts_minimal_first():
    assert (
        PlannerService.should_start_with_minimal_prompt(
            "Inspect current project structure and review architecture before changes.",
            "",
        )
        is True
    )


def test_planning_fallback_timeouts_are_relaxed_for_local_models():
    assert MINIMAL_PLANNING_TIMEOUT_SECONDS == 300
    assert PLANNING_REPAIR_TIMEOUT_SECONDS == 300
    assert ULTRA_MINIMAL_PLANNING_TIMEOUT_SECONDS == 240


def test_worker_soft_time_limit_allows_planning_retries_and_execution_headroom():
    assert ORCHESTRATION_TASK_SOFT_TIME_LIMIT_SECONDS == 3300
    assert ORCHESTRATION_TASK_TIME_LIMIT_SECONDS == 3600
    assert execute_orchestration_task.soft_time_limit == 3300
    assert execute_orchestration_task.time_limit == 3600


def test_qwen_local_prompt_profile_enforces_array_only_output():
    profile = PlannerService.select_prompt_profile("local_openclaw", "qwen-local")
    prompt = PlannerService.build_minimal_planning_prompt(
        "Build a hiring platform",
        project_dir=__import__("pathlib").Path("/tmp/project"),
        prompt_profile=profile,
    )

    assert profile == "local_qwen_json_array"
    assert "first non-whitespace character must be `[`" in prompt
    assert "Do not wrap it in an object" in prompt


def test_minimal_planning_prompt_keeps_workflow_rules_for_existing_fullstack_workspace():
    prompt = PlannerService.build_minimal_planning_prompt(
        "Bring the existing frontend and backend to dev-ready state",
        project_dir=__import__("pathlib").Path("/tmp/project"),
        workflow_profile="fullstack_scaffold",
        workflow_phases=[
            "create_frontend_skeleton",
            "create_backend_skeleton",
            "wire_api_config",
            "verify_dev_startup",
        ],
        workspace_has_existing_files=True,
    )

    assert "Workflow profile: fullstack_scaffold" in prompt
    assert "Extend or verify existing files instead of re-scaffolding" in prompt
    assert "Never use parent-directory traversal like `../backend`" in prompt
    assert "`verification` must be a single shell string or null" in prompt
    assert "Do not use background processes" in prompt


def test_large_planning_context_is_compressed_before_first_attempt():
    state = type("State", (), {})()
    state.project_context = "Very long planning context " * 600
    state.plan = []
    state.current_step_index = 0
    state.completed_steps = []
    state.failed_steps = []
    state.debug_attempts = []
    state.changed_files = []

    compressed = _compress_project_context_for_planning(state)

    assert len(compressed) < len(state.project_context)
    assert "Very long planning context" in compressed


def test_planner_allows_write_pseudo_commands_but_flags_background_commands():
    issues = PlannerService.find_immediate_repair_step_issues(
        [
            {
                "step_number": 1,
                "description": "Write file",
                "commands": ["write frontend/src/App.tsx: render root shell"],
            },
            {
                "step_number": 2,
                "description": "Start backend",
                "commands": ["cd backend && npx tsx src/index.ts &"],
            },
        ]
    )

    assert issues == {
        "background_process_steps": [2],
    }


def test_planner_flags_placeholder_only_implementation_and_weak_verification():
    issues = PlannerService.find_immediate_repair_step_issues(
        [
            {
                "step_number": 1,
                "description": "Create the webpage files",
                "commands": [
                    "mkdir -p assets/css assets/js",
                    "touch index.html assets/css/styles.css assets/js/app.js",
                ],
                "verification": "test -f index.html && test -f assets/css/styles.css",
                "rollback": "rm -f index.html assets/css/styles.css assets/js/app.js",
                "expected_files": [
                    "index.html",
                    "assets/css/styles.css",
                    "assets/js/app.js",
                ],
            }
        ]
    )

    assert issues == {
        "placeholder_only_steps": [1],
        "weak_verification_steps": [1],
    }


def test_minimal_planning_prompt_requires_real_content_and_strong_verification():
    prompt = PlannerService.build_minimal_planning_prompt(
        "Build a one-page site",
        project_dir=__import__("pathlib").Path("/tmp/project"),
        workspace_has_existing_files=True,
    )

    assert "materially write or edit file contents" in prompt
    assert "verification must prove behavior or content" in prompt
    assert "inspect -> edit -> verify" in prompt
    assert "prefer `python3 - <<'PY'` heredoc" in prompt
    assert "never emit `python -c` commands" in prompt
    assert (
        "Each step must include: step_number, description, commands, verification, rollback, expected_files"
        in prompt
    )
    assert "`step_number` must be a unique integer" in prompt
    assert "Do not omit keys" in prompt


def test_weak_verification_is_not_treated_as_blocking_immediate_repair_issue():
    plan = [
        {
            "step_number": 1,
            "description": "Build the page shell",
            "commands": [
                "mkdir -p assets/css",
                "printf '<!doctype html>' > index.html",
            ],
            "verification": "test -f index.html",
            "rollback": "rm -f index.html",
            "expected_files": ["index.html"],
        }
    ]

    issues = PlannerService.find_immediate_repair_step_issues(plan)
    blocking_issue_keys = {
        "non_runnable_steps",
        "background_process_steps",
        "placeholder_only_steps",
    }
    blocking = {
        key: value for key, value in issues.items() if key in blocking_issue_keys
    }

    assert issues["weak_verification_steps"] == [1]
    assert blocking == {}


def test_validator_still_warns_on_weak_verification_for_implementation_plan(tmp_path):
    verdict = ValidatorService.validate_plan(
        [
            {
                "step_number": 1,
                "description": "Build the page shell",
                "commands": ["printf '<!doctype html>' > index.html"],
                "verification": "test -f index.html",
                "rollback": "rm -f index.html",
                "expected_files": ["index.html"],
            }
        ],
        output_text="[]",
        task_prompt="Build a one-page site",
        execution_profile="full_lifecycle",
        project_dir=tmp_path,
    )

    assert verdict.warning is True
    assert "weak_verification_steps" in verdict.details


def test_schema_valid_planner_output_passes_validator_without_repair(tmp_path):
    plan = [
        {
            "step_number": 1,
            "description": "Inspect current Python runtime entry points",
            "commands": ['rg -n "FastAPI|create_app|app =" app || true'],
            "verification": "python3 -c \"print('inspect ok')\"",
            "rollback": None,
            "expected_files": [],
        },
        {
            "step_number": 2,
            "description": "Update runtime configuration defaults",
            "commands": ["mkdir -p app && printf 'VALUE = 1\\n' > app/config.py"],
            "verification": "python3 -m py_compile app/config.py",
            "rollback": "rm -f app/config.py",
            "expected_files": ["app/config.py"],
        },
        {
            "step_number": 3,
            "description": "Verify configuration imports cleanly",
            "commands": ["python3 -m py_compile app/config.py"],
            "verification": "python3 -m py_compile app/config.py",
            "rollback": None,
            "expected_files": [],
        },
    ]

    verdict = ValidatorService.validate_plan(
        plan,
        output_text=json.dumps(plan),
        task_prompt="Update runtime configuration",
        execution_profile="full_lifecycle",
        project_dir=tmp_path,
    )

    assert verdict.accepted is True
    assert verdict.repairable is False
    assert verdict.rejected is False


def test_planner_sanitizes_common_local_model_static_site_plan_issues():
    sanitized = PlannerService.sanitize_common_plan_issues(
        [
            {
                "step_number": 1,
                "description": "Create index",
                "commands": [
                    "write index.html: html shell",
                    "file index.html should be a semantic landing page",
                ],
                "verification": "test -f index.html",
                "rollback": "trash index.html",
                "expected_files": ["index.html"],
            },
            {
                "step_number": 2,
                "description": "Final validation: open the page in a local preview to confirm rendering",
                "commands": [
                    "python3 -m http.server 8080 --bind 127.0.0.1 &",
                    "sleep 1 && curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/index.html",
                    "pkill -f 'python3 -m http.server 8080' || true",
                ],
                "verification": "echo ok",
                "rollback": "pkill -f 'python3 -m http.server' || true",
                "expected_files": ["index.html"],
            },
        ]
    )

    assert len(sanitized) == 1
    assert sanitized[0]["commands"] == ["write index.html: html shell"]
    assert sanitized[0]["rollback"] == "rm -f index.html"


def test_planning_repair_prompt_forbids_duplicated_workspace_roots():
    prompt = PlannerService.build_planning_repair_prompt(
        "Build frontend and backend scaffolding",
        malformed_output='[{"step_number":1,"commands":["mkdir -p frontend/src/frontend/src"]}]',
        project_dir=__import__("pathlib").Path("/tmp/project"),
        workflow_profile="fullstack_scaffold",
        workflow_phases=[
            "create_frontend_skeleton",
            "create_backend_skeleton",
            "wire_api_config",
            "verify_dev_startup",
        ],
    )

    assert "frontend/src/frontend/src" in prompt
    assert "backend/src/backend/src" in prompt
    assert "rooted exactly once" in prompt
    assert "Never use parent-directory traversal like `../backend`" not in prompt


def test_planning_repair_prompt_uses_reduced_context_only():
    knowledge_context = type(
        "KnowledgeCtx",
        (),
        {
            "retrieved_items": [
                type(
                    "Ref",
                    (),
                    {
                        "knowledge_type": "format_guide",
                        "title": "First",
                        "content": "alpha" * 200,
                    },
                )(),
                type(
                    "Ref",
                    (),
                    {
                        "knowledge_type": "task_example",
                        "title": "Second",
                        "content": "beta" * 200,
                    },
                )(),
                type(
                    "Ref",
                    (),
                    {
                        "knowledge_type": "debug_case",
                        "title": "Third",
                        "content": "gamma" * 200,
                    },
                )(),
            ]
        },
    )()

    prompt = PlannerService.build_planning_repair_prompt(
        "Massive task context that should not survive into repair prompt",
        malformed_output='{"nonProjectContext":"' + ("x" * 7000) + '"}',
        project_dir=__import__("pathlib").Path("/tmp/project"),
        rejection_reasons=[
            "commands must be an array",
            "verification must be a shell string",
        ],
        workflow_profile="fullstack_scaffold",
        workflow_phases=[
            "create_frontend_skeleton",
            "create_backend_skeleton",
        ],
        workspace_has_existing_files=True,
        knowledge_context=knowledge_context,
    )

    assert "Task:" not in prompt
    assert "Working directory:" not in prompt
    assert "Workflow profile:" not in prompt
    assert "projectContext" not in prompt
    assert "nonProjectContext" not in prompt
    assert prompt.count("[format_guide]") == 1
    assert prompt.count("[task_example]") == 1
    assert "Third" not in prompt
    assert "nonProjectContextChars" not in prompt
    assert "Validation error:" in prompt
    assert "Required JSON schema:" in prompt
    assert len(prompt) < PLANNING_REPAIR_PROMPT_MAX_CHARS
    assert "python3 - <<'PY'" in prompt


def test_validator_rejects_brittle_python_c_with_nested_quotes(tmp_path):
    verdict = ValidatorService.validate_plan(
        [
            {
                "step_number": 1,
                "description": "Check Python version",
                "commands": [
                    "python3 -c \"import sys; print(f'Python {sys.version}')\"",
                ],
                "verification": "test -n ok",
                "rollback": None,
                "expected_files": [],
            }
        ],
        output_text="[]",
        task_prompt="Check runtime",
        execution_profile="full_lifecycle",
        project_dir=tmp_path,
    )

    assert verdict.repairable is True
    assert "brittle" in " ".join(verdict.reasons).lower()


def test_shell_safe_command_guide_recommends_python_heredoc():
    guide = (
        __import__("pathlib")
        .Path("knowledge/seed/format_guides/shell-safe-command.md")
        .read_text()
    )

    assert "prefer heredoc syntax for inline python" in guide.lower()
    assert "python3 - <<'PY'" in guide


def test_planning_repair_still_succeeds_for_small_malformed_output():
    captured = {}

    class Runtime:
        async def execute_task(self, prompt, timeout_seconds=300, **kwargs):
            captured["prompt"] = prompt
            captured["timeout_seconds"] = timeout_seconds
            return {"output": '[{"step_number":1}]'}

    result = PlannerService.repair_output(
        runtime_service=Runtime(),
        task_description="Build a page",
        malformed_output='{"steps":"bad"}',
        project_dir=__import__("pathlib").Path("/tmp/project"),
        timeout_seconds=300,
        logger=__import__("logging").getLogger("test"),
        emit_live=lambda *a, **kw: None,
        reason="json_parse_failed",
        rejection_reasons=["commands must be an array"],
        knowledge_context=None,
        session_id=1,
        task_id=2,
    )

    assert result == {"output": '[{"step_number":1}]'}
    assert "nonProjectContext" not in captured["prompt"]
    assert len(captured["prompt"]) < PLANNING_REPAIR_PROMPT_MAX_CHARS


def test_planning_repair_uses_isolated_one_shot_prompt_when_available():
    captured = {}

    class Runtime:
        async def invoke_prompt(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return {"output": '[{"step_number":1}]'}

    result = PlannerService.repair_output(
        runtime_service=Runtime(),
        task_description="Build a page",
        malformed_output='{"projectContext":"bad","nonProjectContext":"bad"}',
        project_dir=__import__("pathlib").Path("/tmp/project"),
        timeout_seconds=300,
        logger=__import__("logging").getLogger("test"),
        emit_live=lambda *a, **kw: None,
        reason="json_parse_failed",
        rejection_reasons=["commands must be an array"],
        knowledge_context=None,
        session_id=1,
        task_id=2,
    )

    assert result == {"output": '[{"step_number":1}]'}
    assert "projectContext" not in captured["prompt"]
    assert "nonProjectContext" not in captured["prompt"]
    assert captured["kwargs"]["isolate_workspace_context"] is True
    assert captured["kwargs"]["session_prefix"] == "planning-repair"


def test_planning_repair_budget_fails_fast_without_retry():
    runtime = type(
        "Runtime",
        (),
        {
            "execute_task": lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("repair should be skipped before runtime call")
            )
        },
    )()
    oversized_output = '[{"step_number":1,"description":"' + ("x" * 12000) + '"}]'

    from app.services.orchestration import planning as planning_pkg

    original_budget = planning_pkg.planner.REPAIR_PROMPT_MAX_CHARS
    original_alias_budget = planning_pkg.planner.PLANNING_REPAIR_PROMPT_MAX_CHARS
    planning_pkg.planner.REPAIR_PROMPT_MAX_CHARS = 200
    planning_pkg.planner.PLANNING_REPAIR_PROMPT_MAX_CHARS = 200
    try:
        try:
            PlannerService.repair_output(
                runtime_service=runtime,
                task_description="Build a page",
                malformed_output=oversized_output,
                project_dir=__import__("pathlib").Path("/tmp/project"),
                timeout_seconds=300,
                logger=__import__("logging").getLogger("test"),
                emit_live=lambda *a, **kw: None,
                reason="json_parse_failed",
                rejection_reasons=[("commands must be array " + ("z" * 400))] * 4,
                knowledge_context=type(
                    "KnowledgeCtx",
                    (),
                    {
                        "retrieved_items": [
                            type(
                                "Ref",
                                (),
                                {
                                    "knowledge_type": "format_guide",
                                    "title": "Hint",
                                    "content": "y" * 2000,
                                },
                            )(),
                            type(
                                "Ref",
                                (),
                                {
                                    "knowledge_type": "task_example",
                                    "title": "Hint 2",
                                    "content": "q" * 2000,
                                },
                            )(),
                        ]
                    },
                )(),
            )
        except PlanningRepairBudgetExceeded:
            pass
        else:
            raise AssertionError("Expected PlanningRepairBudgetExceeded")
    finally:
        planning_pkg.planner.REPAIR_PROMPT_MAX_CHARS = original_budget
        planning_pkg.planner.PLANNING_REPAIR_PROMPT_MAX_CHARS = original_alias_budget


def test_local_qwen_single_step_plan_is_routed_to_repair():
    assert (
        _should_repair_truncated_single_step_plan(
            prompt_profile="local_qwen_json_array",
            execution_profile="full_lifecycle",
            extracted_plan=[
                {
                    "step_number": 1,
                    "description": "Set up frontend and backend foundations",
                    "commands": ["mkdir -p frontend backend"],
                    "verification": "test -d frontend && test -d backend",
                    "rollback": "rm -rf frontend backend",
                    "expected_files": ["frontend/src/main.tsx", "backend/src/index.ts"],
                }
            ],
        )
        is True
    )


def test_non_qwen_or_non_full_lifecycle_single_step_plan_still_uses_retry_guard():
    single_step_plan = [
        {
            "step_number": 1,
            "description": "Do work",
            "commands": ["echo hi"],
            "verification": "test -n hi",
            "rollback": "true",
            "expected_files": [],
        }
    ]

    assert (
        _should_repair_truncated_single_step_plan(
            prompt_profile="default",
            execution_profile="full_lifecycle",
            extracted_plan=single_step_plan,
        )
        is False
    )
    assert (
        _should_repair_truncated_single_step_plan(
            prompt_profile="local_qwen_json_array",
            execution_profile="review_only",
            extracted_plan=single_step_plan,
        )
        is False
    )


def test_aborted_timeout_metadata_is_not_treated_as_salvageable_plan_output():
    output_text = (
        '{"total":0,"aborted":true,"source":"run","generatedAt":1777555426260}'
    )

    assert PlannerService.looks_salvageable_planning_output(output_text) is False


def test_minimal_prompt_retry_uses_fresh_session_instead_of_task_session():
    captured = {}

    class RuntimeService:
        async def execute_task(self, prompt, timeout_seconds=300, **kwargs):
            captured["reuse_task_session"] = kwargs.get("reuse_task_session")
            return {"status": "failed", "output": "", "error": "Task timed out"}

    PlannerService.retry_with_minimal_prompt(
        runtime_service=RuntimeService(),
        task_description="Build a one-page site",
        project_dir=__import__("pathlib").Path("/tmp/project"),
        timeout_seconds=60,
        logger=__import__("logging").getLogger("test"),
        emit_live=lambda *args, **kwargs: None,
        reason="timeout",
    )

    assert captured["reuse_task_session"] is False

# Scripts

Operational scripts kept here are project-specific helpers. Runtime startup is
handled by `../start.sh`; service logs are written directly to `../logs/`.

## Session And Replay Inspection

- `capture_replay_report.py` - capture semantic replay reports from event journals.
- `capture_task_evidence_bundle.py` - capture a stable per-TaskExecution evidence bundle.
- `inspect_session_state.py` - inspect one session in SQLite.
- `inspect_task_execution_attempts.py` - inspect task execution attempts.
- `inspect_event_journal.py` - inspect orchestration event journals.
- `inspect_runtime_logs.py` - inspect runtime log identity coverage.
- `inspect_checkpoints.py` - inspect checkpoint contents.
- `diagnose_planning_stuck.py` - diagnose planning/session stalls.
- `session_outcome_report.py` - summarize recent session outcomes.
- `workspace_evidence_report.py` - summarize workspace/change-set evidence for recent task executions.
- `planning_contract_report.py` - summarize recent planning contract violations.
- `failure_taxonomy.py` - classify recent failures into reusable failure buckets.
- `phase18e_collect_real_session_validator_evidence.py` - read-only aggregation of Phase 18B/18C validator/recovery telemetry from persisted sessions and their event journals.

## Planning And Knowledge

- `validate_plan_json.py` - validate planner JSON against the deterministic plan contract.
- `planning_floor_check.py` - run an OpenClaw planning-floor diagnostic.
- `ingest_knowledge.py` - ingest knowledge documents into SQLite and Qdrant.
  For the compact laptop Ollama Docker runtime, prefer
  `./wsl-start.sh --ollama --ingest-knowledge`, or:
  `docker compose -f docker-compose.windows.yml exec -T orchestrator python scripts/ingest_knowledge.py --source-dir /app --qdrant-url http://qdrant:6333`.

## Developer Utilities

- `wsl-ollama-start.sh` - compact WSL2 Docker/Ollama startup helper.
- `windows_health_check.ps1` - Windows-side Docker/Ollama/backend health checks.
- `format-python.sh` - format backend Python files with Black.
- `security_check.sh` - scan tracked source-like files for likely secret exposure.
- `orchestrator-mobile-api.sh` - call mobile API endpoints using local env credentials.
- `kill-all.sh` - force-kill local development processes.

## Maintenance

- `check_openai_compatible_endpoint.py` - verify an OpenAI-compatible endpoint is reachable and responding correctly.
- `score_orchestrator_eval_case.py` - score a single orchestrator eval case against expected outcomes.
- `planning_contract_report.py` - summarize recent planning contract violations (also listed under Session And Replay).
- `validate_incremental.py` - initial live-validation harness for Slice J incremental execution (creation-only fast path).
- `validate_incremental_fresh_process.py` - fresh-process re-run harness confirming the `search()` code-fence fix (8-task Python corpus).
- `validate_incremental_20task.py` - 20-task controlled validation window for Slice J post A+E fix (Python/HTML/CSS/JSON).
- `probe_incremental_output.py` - diagnostic probe capturing raw `execute_task` output shapes (no file writes; used for attribution).
- `validate_repo_memory.py` - validate RepoMemory injection against a live project.
- `validate_repo_memory_injection.py` - integration check for RepoMemory block assembly and injection gate.
- `wm_off_runner.py` - WM OFF arm runner for Priority 5 WorkingMemory A/B measurement: creates 3 Python package projects × 6 tasks each, dispatches sequentially, collects debug_repair_attempted and planning repair events per task. Saves raw JSON to `docs/roadmap/reports/maintenance/`.
- `wm_off_recovery.py` - recovery companion to `wm_off_runner.py`: monitors in-flight tasks from a prior runner session, re-dispatches failed/cancelled Task 1s, and collects full event data for the final report.
- `phase18f_seed_real_session_evidence.py` - evidence-generation harness that seeds persisted project/session/task rows and candidate validation events for Phase 18F; no validator/recovery/policy/feature-flag defaults changed.
- `phase18i_machine_a_limited_validation.py` - evidence-only harness running the Candidate Recovery path with Machine A standard runtime inputs; restores feature-flag settings after each controlled session.
- `reflection_replay.py` - Phase 17B-V offline reflection replay tool: replays `ReflectionRetryStrategy` against a synthetic failure corpus (no runtime mutation, no database).

Removed obsolete scripts: old `/tmp` log sync/status/cleanup helpers. Current
startup no longer writes logs to `/tmp`.

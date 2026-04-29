from app.services.error_handler import EnhancedErrorHandler


def test_attempt_json_parsing_recovers_qwen_broken_shell_quotes_and_localhost_links():
    handler = EnhancedErrorHandler()
    broken = """[{"step_number": 2, "description": "create_backend_skeleton", "commands": ["python3 -m venv .venv"], "verification": ".venv/bin/python -c 'from app.main import app; from app.config import Settings; print("backend imports OK")'", "rollback": "rm -rf .venv", "expected_files": ["app/main.py"]}, {"step_number": 3, "description": "wire_api_config", "commands": ["grep -n 'localhost:8080' frontend/vite.config.ts"], "verification": ".venv/bin/python -c 'from app.config import Settings; s=Settings(); assert "[](http://localhost:3000)<http://localhost:3000>" in s.CORS_ORIGINS; print("cors aligned")'", "rollback": null, "expected_files": ["frontend/vite.config.ts"]}]"""

    success, parsed, strategy = handler.attempt_json_parsing(broken, context="planning")

    assert success is True
    assert isinstance(parsed, list)
    assert parsed[0]["step_number"] == 2
    assert 'print("backend imports OK")' in parsed[0]["verification"]
    assert '"http://localhost:3000"' in parsed[1]["verification"]
    assert "Fixed common errors" in strategy or "Found and fixed JSON" in strategy

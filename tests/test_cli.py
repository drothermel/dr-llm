import json

from typer.testing import CliRunner

from llm_pool.cli import app


def test_providers_command_lists_known_providers() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    providers = {item["provider"] for item in payload["providers"]}
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers
    assert "glm" in providers
    assert "minimax" in providers
    assert "claude-code-minimax" in providers
    assert "claude-code-kimi" in providers

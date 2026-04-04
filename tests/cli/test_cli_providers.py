from __future__ import annotations

import json

from typer.testing import CliRunner

from dr_llm.cli import app

runner = CliRunner()


def test_providers_human_readable() -> None:
    result = runner.invoke(app, ["providers"])

    assert result.exit_code == 0
    assert "Providers" in result.stdout
    assert "Available" in result.stdout
    assert "Structured" in result.stdout
    assert "openai" in result.stdout
    assert "anthropic" in result.stdout
    assert "claude-code" in result.stdout
    assert '"providers"' not in result.stdout


def test_providers_json() -> None:
    result = runner.invoke(app, ["providers", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    providers = {item["provider"] for item in payload["providers"]}
    assert "openai" in providers
    assert "anthropic" in providers
    assert "google" in providers
    assert "glm" in providers
    assert "kimi-code" in providers
    assert "minimax" in providers
    assert "claude-code-minimax" in providers
    for item in payload["providers"]:
        assert isinstance(item["available"], bool)
        assert isinstance(item["missing_env_vars"], list | tuple)
        assert isinstance(item["missing_executables"], list | tuple)

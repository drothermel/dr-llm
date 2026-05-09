from __future__ import annotations

import json

from typer.testing import CliRunner

from dr_llm.cli import app

runner = CliRunner()


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
    for item in payload["providers"]:
        assert isinstance(item["available"], bool)
        assert isinstance(item["missing_env_vars"], list | tuple)
        assert isinstance(item["missing_executables"], list | tuple)

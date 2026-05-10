from __future__ import annotations

from dr_llm.llm import ProviderName
import json

from typer.testing import CliRunner

from dr_llm.cli import app

runner = CliRunner()


def test_providers_json() -> None:
    result = runner.invoke(app, ["providers", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    providers = {item["provider"] for item in payload["providers"]}
    assert ProviderName.OPENAI in providers
    assert ProviderName.ANTHROPIC in providers
    assert ProviderName.GOOGLE in providers
    assert ProviderName.GLM in providers
    assert ProviderName.KIMI_CODE in providers
    assert ProviderName.MINIMAX in providers
    for item in payload["providers"]:
        assert isinstance(item["available"], bool)
        assert isinstance(item["missing_env_vars"], list | tuple)
        assert isinstance(item["missing_executables"], list | tuple)

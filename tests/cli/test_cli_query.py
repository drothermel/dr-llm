from __future__ import annotations

from dr_llm.llm import ProviderName
import json
from typing import Any

import pytest
from typer.testing import CliRunner

import dr_llm.cli.query as query_cli
from dr_llm.cli import app
from dr_llm.llm import EffortSpec, TokenUsage, parse_llm_request
from tests.conftest import make_response

runner = CliRunner()


class _FakeProvider:
    mode = "api"

    def __init__(self, name: str) -> None:
        self.name = name

    def build_request(
        self,
        *,
        model: str,
        messages: list[Any],
        max_tokens: int | None = None,
        effort: EffortSpec = EffortSpec.NA,
        reasoning: Any = None,
        temperature: float | None = None,
        top_p: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "provider": self.name,
            "model": model,
            "messages": messages,
            "effort": effort,
            "reasoning": reasoning,
            "metadata": metadata or {},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        return parse_llm_request(payload)

    def generate(self, request: Any) -> Any:
        return make_response(
            text="hi",
            usage=TokenUsage(
                prompt_tokens=1, completion_tokens=2, total_tokens=3
            ),
            provider=ProviderName.OPENAI,
            model="gpt-4.1",
        )


class _FakeRegistry:
    def get(self, name: str) -> _FakeProvider:
        return _FakeProvider(name)

    def close(self) -> None:
        pass


def test_query_emits_response_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_cli, "build_default_registry", _FakeRegistry)

    result = runner.invoke(
        app,
        [
            "query",
            "--provider",
            ProviderName.OPENAI,
            "--model",
            "gpt-4.1",
            "--message",
            "hello",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["text"] == "hi"
    assert payload["usage"]["total_tokens"] == 3


def test_query_rejects_temperature_for_headless_provider() -> None:
    result = runner.invoke(
        app,
        [
            "query",
            "--provider",
            ProviderName.CODEX,
            "--model",
            "gpt-5.4-mini",
            "--message",
            "hi",
            "--temperature",
            "0.5",
        ],
    )

    assert result.exit_code != 0


def test_query_rejects_max_tokens_for_headless_provider() -> None:
    result = runner.invoke(
        app,
        [
            "query",
            "--provider",
            ProviderName.CLAUDE_CODE,
            "--model",
            "claude-sonnet-4-6",
            "--message",
            "hi",
            "--max-tokens",
            "32",
        ],
    )

    assert result.exit_code != 0


def test_query_rejects_temperature_for_kimi_code() -> None:
    result = runner.invoke(
        app,
        [
            "query",
            "--provider",
            ProviderName.KIMI_CODE,
            "--model",
            "kimi-for-coding",
            "--message",
            "hi",
            "--max-tokens",
            "32",
            "--effort",
            "high",
            "--temperature",
            "0.5",
        ],
    )

    assert result.exit_code != 0


def test_query_accepts_max_tokens_for_kimi_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(query_cli, "build_default_registry", _FakeRegistry)

    result = runner.invoke(
        app,
        [
            "query",
            "--provider",
            ProviderName.KIMI_CODE,
            "--model",
            "kimi-for-coding",
            "--message",
            "hi",
            "--max-tokens",
            "32",
            "--effort",
            "high",
        ],
    )

    assert result.exit_code == 0

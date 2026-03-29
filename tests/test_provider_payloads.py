from __future__ import annotations

import json
import subprocess
from typing import Any, cast

import httpx

from dr_llm.providers.api_provider_config import APIProviderConfig
from dr_llm.providers.anthropic import AnthropicAdapter, AnthropicConfig
from dr_llm.providers.google import GoogleAdapter
from dr_llm.providers.headless import (
    ClaudeHeadlessAdapter,
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
    CodexHeadlessAdapter,
)
from dr_llm.generation.models import LlmRequest, Message


def test_anthropic_payload_serializes_plain_messages() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": "end_turn",
            },
        )

    adapter = AnthropicAdapter(
        config=AnthropicConfig(
            api_key="x", base_url="https://api.anthropic.com/v1/messages"
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = adapter.generate(
        LlmRequest(
            provider="anthropic",
            model="claude-test",
            messages=[
                Message(role="system", content="Be concise."),
                Message(role="user", content="find item"),
                Message(role="assistant", content="previous answer"),
            ],
        )
    )

    assert response.text == "done"
    payload = cast(dict[str, Any], captured["payload"])
    assert payload["system"] == "Be concise."
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "find item"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "previous answer"}],
        },
    ]
    assert "tools" not in payload


def test_google_payload_serializes_plain_messages() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "done"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 3,
                },
            },
        )

    adapter = GoogleAdapter(
        config=APIProviderConfig(
            name="google",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GOOGLE_API_KEY",
            api_key="x",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = adapter.generate(
        LlmRequest(
            provider="google",
            model="gemini-test",
            messages=[
                Message(role="system", content="Be concise."),
                Message(role="user", content="find item"),
                Message(role="assistant", content="previous answer"),
            ],
        )
    )

    assert response.text == "done"
    payload = cast(dict[str, Any], captured["payload"])
    headers = cast(dict[str, str], captured["headers"])
    assert headers["x-goog-api-key"] == "x"
    assert "?key=" not in cast(str, captured["url"])
    assert payload["systemInstruction"] == {"parts": [{"text": "Be concise."}]}
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "find item"}]},
        {"role": "model", "parts": [{"text": "previous answer"}]},
    ]
    assert "tools" not in payload


def test_codex_headless_defaults_use_exec_and_minimal_flags(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, Any] = {}

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["command"] = args[0]
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="\n".join(
                [
                    '{"type":"turn.started"}',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}',
                    '{"type":"turn.completed","usage":{"input_tokens":2,"output_tokens":3}}',
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexHeadlessAdapter()
    response = adapter.generate(
        LlmRequest(
            provider="codex",
            model="gpt-5-codex",
            messages=[Message(role="user", content="hello")],
        )
    )

    command = cast(list[str], captured["command"])
    assert command[:3] == ["codex", "exec", "--json"]
    assert "--disable" in command
    assert "web_search_request" in command
    assert "include_plan_tool=false" in command
    assert "project_doc_max_bytes=0" in command
    assert "-m" in command
    assert "gpt-5-codex" in command
    assert command[-1] == "-"
    assert 'model_instructions_file="' in " ".join(command)
    assert str(captured["input"]) == "user: hello"
    assert response.text == "OK"
    assert response.usage.prompt_tokens == 2
    assert response.usage.completion_tokens == 3


def test_claude_headless_defaults_use_empty_system_prompt(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, Any] = {}

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["command"] = args[0]
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "OK",
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                    "total_cost_usd": 0.01,
                },
                ensure_ascii=True,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeHeadlessAdapter()
    response = adapter.generate(
        LlmRequest(
            provider="claude-code",
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="hello")],
        )
    )

    command = cast(list[str], captured["command"])
    assert command[0] == "claude"
    assert "-p" in command
    assert "--output-format" in command
    assert "json" in command
    assert "--system-prompt" in command
    assert "--disable-slash-commands" in command
    assert "--no-session-persistence" in command
    assert "--setting-sources" in command
    assert "--model" in command
    assert "claude-sonnet-4-6" in command
    assert "--tools" not in command
    system_prompt_index = command.index("--system-prompt")
    assert command[system_prompt_index + 1] == ""
    assert str(captured["input"]) == "user: hello"
    assert response.text == "OK"
    assert response.usage.prompt_tokens == 1
    assert response.usage.completion_tokens == 2
    assert response.cost is not None
    assert response.cost.total_cost_usd == 0.01


def test_claude_minimax_preset_maps_api_key_env(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, Any] = {}
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-test-key")

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["env"] = kwargs.get("env")
        captured["command"] = args[0]
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "OK",
                },
                ensure_ascii=True,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeHeadlessMiniMaxAdapter(
        command=["claude", "-p", "--output-format", "json"]
    )
    response = adapter.generate(
        LlmRequest(
            provider="claude-code-minimax",
            model="MiniMax-M2.1",
            messages=[Message(role="user", content="hello")],
        )
    )

    env = cast(dict[str, str], captured["env"])
    assert env["ANTHROPIC_BASE_URL"] == "https://api.minimax.io/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "minimax-test-key"
    assert env["ANTHROPIC_API_KEY"] == "minimax-test-key"
    assert response.text == "OK"


def test_claude_kimi_preset_maps_api_key_env(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, Any] = {}
    monkeypatch.setenv("KIMI_API_KEY", "kimi-test-key")

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["env"] = kwargs.get("env")
        captured["command"] = args[0]
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "OK",
                },
                ensure_ascii=True,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeHeadlessKimiAdapter(
        command=["claude", "-p", "--output-format", "json"]
    )
    response = adapter.generate(
        LlmRequest(
            provider="claude-code-kimi",
            model="kimi-for-coding",
            messages=[Message(role="user", content="hello")],
        )
    )

    env = cast(dict[str, str], captured["env"])
    assert env["ANTHROPIC_BASE_URL"] == "https://api.kimi.com/coding/"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "kimi-test-key"
    assert env["ANTHROPIC_API_KEY"] == "kimi-test-key"
    assert response.text == "OK"

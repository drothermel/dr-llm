from __future__ import annotations

import json
import subprocess
from typing import cast

import pytest
from pydantic import ValidationError

from dr_llm.providers.effort import EffortSpec
from dr_llm.providers.headless.claude import ClaudeHeadlessAdapter
from dr_llm.providers.headless.claude_presets import (
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
)
from dr_llm.providers.headless.codex import CodexHeadlessAdapter
from dr_llm.providers.reasoning import CodexReasoning, ReasoningBudget, ThinkingLevel
from tests.conftest import make_request
from tests.providers.conftest import make_subprocess_mock


def test_codex_command_and_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = "\n".join([
        json.dumps({"type": "turn.started"}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 3}}),
    ])
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexHeadlessAdapter()
    request = make_request(provider="codex", model="gpt-5-codex")
    response = adapter.generate(request)

    command = cast(list[str], captured["command"])
    assert command[:3] == ["codex", "exec", "--json"]
    assert "-m" in command
    assert command[command.index("-m") + 1] == "gpt-5-codex"
    assert captured["input"] == "user: hello"
    assert response.text == "OK"
    assert response.usage.prompt_tokens == 2
    assert response.usage.completion_tokens == 3


def test_codex_command_includes_reasoning_effort_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}}
            ),
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 3}}
            ),
        ]
    )
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexHeadlessAdapter()
    request = make_request(
        provider="codex",
        model="gpt-5.1-codex-mini",
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.HIGH),
    )
    adapter.generate(request)

    command = cast(list[str], captured["command"])
    assert '-c' in command
    assert 'model_reasoning_effort="high"' in command


def test_claude_command_and_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "OK",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "total_cost_usd": 0.01,
    })
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeHeadlessAdapter()
    request = make_request(
        provider="claude-code",
        model="claude-sonnet-4-6",
        effort=EffortSpec.MEDIUM,
    )
    response = adapter.generate(request)

    command = cast(list[str], captured["command"])
    assert command[0] == "claude"
    assert "--model" in command
    assert command[command.index("--model") + 1] == "claude-sonnet-4-6"
    assert "--system-prompt" in command
    assert command[command.index("--system-prompt") + 1] == ""
    assert captured["input"] == "user: hello"
    assert response.text == "OK"
    assert response.cost is not None
    assert response.cost.total_cost_usd == 0.01


def test_claude_command_includes_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "OK",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "total_cost_usd": 0.01,
        }
    )
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeHeadlessAdapter()
    request = make_request(
        provider="claude-code",
        model="claude-sonnet-4-6",
        effort=EffortSpec.HIGH,
    )
    adapter.generate(request)

    command = cast(list[str], captured["command"])
    assert command[command.index("--effort") + 1] == "high"


def test_claude_minimax_preset_maps_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-test-key")
    stdout = json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "OK",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "total_cost_usd": 0.0,
    })
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeHeadlessMiniMaxAdapter(command=["claude", "-p", "--output-format", "json"])
    request = make_request(provider="claude-code-minimax", model="MiniMax-M2.1")
    adapter.generate(request)

    env = cast(dict[str, str], captured["env"])
    assert env["ANTHROPIC_BASE_URL"] == "https://api.minimax.io/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "minimax-test-key"


def test_claude_kimi_preset_maps_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_API_KEY", "kimi-test-key")
    stdout = json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "OK",
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "total_cost_usd": 0.0,
    })
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeHeadlessKimiAdapter(command=["claude", "-p", "--output-format", "json"])
    request = make_request(provider="claude-code-kimi", model="kimi-for-coding")
    adapter.generate(request)

    env = cast(dict[str, str], captured["env"])
    assert env["ANTHROPIC_BASE_URL"] == "https://api.kimi.com/coding/"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "kimi-test-key"


def test_codex_rejects_reasoning_before_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "OK"},
                }
            ),
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 3}}
            ),
        ]
    )
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ValidationError):
        make_request(
            provider="codex",
            model="gpt-5-codex",
            reasoning=ReasoningBudget(tokens=1024),
        )
    assert "command" not in captured

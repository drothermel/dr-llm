from __future__ import annotations

import json
import subprocess
from typing import cast

import pytest
from pydantic import ValidationError

from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.headless.claude import ClaudeHeadlessProvider
from dr_llm.llm.providers.headless.codex import CodexHeadlessProvider
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    ReasoningBudget,
    ThinkingLevel,
)
from tests.conftest import make_request
from tests.llm.providers.conftest import make_subprocess_mock


def test_codex_command_and_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
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
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                }
            ),
        ]
    )
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexHeadlessProvider()
    request = make_request(
        provider="codex",
        model="gpt-5.1-codex-mini",
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.LOW),
    )
    response = adapter.generate(request)

    command = cast(list[str], captured["command"])
    assert command[:3] == ["codex", "exec", "--json"]
    assert "--ephemeral" in command
    assert "include_plan_tool=false" not in command
    assert "-m" in command
    assert command[command.index("-m") + 1] == "gpt-5.1-codex-mini"
    assert captured["input"] == "user: hello"
    assert response.text == "OK"
    assert response.usage.prompt_tokens == 2
    assert response.usage.completion_tokens == 3


def test_codex_uses_cli_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = cast(list[str], args[0])
        captured["command"] = command
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "turn.started"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "agent_message", "text": "OK"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {"input_tokens": 2, "output_tokens": 3},
                        }
                    ),
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexHeadlessProvider()
    request = make_request(
        provider="codex",
        model="gpt-5.1-codex-mini",
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.LOW),
    )

    adapter.generate(request)

    assert captured["timeout"] == 600.0


def test_codex_command_includes_reasoning_effort_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                }
            ),
        ]
    )
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexHeadlessProvider()
    request = make_request(
        provider="codex",
        model="gpt-5.1-codex-mini",
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.XHIGH),
    )
    adapter.generate(request)

    command = cast(list[str], captured["command"])
    assert "-c" in command
    assert 'model_reasoning_effort="xhigh"' in command


def test_claude_command_and_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
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

    adapter = ClaudeHeadlessProvider()
    request = make_request(
        provider="claude-code",
        model="claude-sonnet-4-6",
        effort=EffortSpec.MEDIUM,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    response = adapter.generate(request)

    command = cast(list[str], captured["command"])
    assert command[0] == "claude"
    assert "--model" in command
    assert command[command.index("--model") + 1] == "claude-sonnet-4-6"
    assert "--system-prompt" in command
    assert command[command.index("--system-prompt") + 1] == ""
    assert "--tools" in command
    assert command[command.index("--tools") + 1] == ""
    assert captured["input"] == "user: hello"
    assert response.text == "OK"
    assert response.cost is not None
    assert response.cost.total_cost_usd == 0.01


def test_claude_uses_cli_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = cast(list[str], args[0])
        captured["command"] = command
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "OK",
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = ClaudeHeadlessProvider()
    request = make_request(
        provider="claude-code",
        model="claude-sonnet-4-6",
        effort=EffortSpec.MEDIUM,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )

    adapter.generate(request)

    assert captured["timeout"] == 600.0


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

    adapter = ClaudeHeadlessProvider()
    request = make_request(
        provider="claude-code",
        model="claude-sonnet-4-6",
        effort=EffortSpec.HIGH,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    adapter.generate(request)

    command = cast(list[str], captured["command"])
    assert command[command.index("--effort") + 1] == "high"


def test_codex_rejects_reasoning_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                }
            ),
        ]
    )
    captured, fake_run = make_subprocess_mock(stdout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ValidationError):
        make_request(
            provider="codex",
            model="gpt-5.1-codex-mini",
            reasoning=ReasoningBudget(tokens=1024),
        )
    assert "command" not in captured

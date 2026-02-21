from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_pool.types import Message, ModelToolCall, ReasoningConfig, TokenUsage


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError):
        Message(role="tool", content='{"ok":true}')


def test_non_assistant_message_rejects_tool_calls() -> None:
    with pytest.raises(ValidationError):
        Message(
            role="user",
            content="hello",
            tool_calls=[
                ModelToolCall(tool_call_id="tc_1", name="lookup", arguments={"q": "x"})
            ],
        )


def test_reasoning_config_rejects_effort_with_max_tokens() -> None:
    with pytest.raises(ValidationError):
        ReasoningConfig(effort="high", max_tokens=100)


def test_reasoning_config_effective_enabled() -> None:
    assert ReasoningConfig(effort="low").effective_enabled
    assert not ReasoningConfig(enabled=False).effective_enabled


def test_token_usage_computed_total_and_non_negative_validation() -> None:
    usage = TokenUsage(
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=10,
        reasoning_tokens=1,
    )
    assert usage.computed_total_tokens == 5

    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1)


def test_token_usage_coerces_and_derives_total() -> None:
    usage = TokenUsage.from_raw(
        prompt_tokens="4", completion_tokens="6", total_tokens=None
    )
    assert usage.prompt_tokens == 4
    assert usage.completion_tokens == 6
    assert usage.total_tokens == 10


def test_token_usage_rejects_invalid_integer_like_values() -> None:
    with pytest.raises(ValidationError):
        TokenUsage.from_raw(prompt_tokens="abc")

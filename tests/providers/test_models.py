from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_llm.providers.models import Message
from dr_llm.providers.reasoning import ReasoningConfig
from dr_llm.providers.usage import TokenUsage


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Message(role="tool", content="nope")  # type: ignore[arg-type]


def test_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Message(role="assistant", content="hi", tool_calls=[])  # type: ignore[call-arg]


def test_reasoning_config_rejects_effort_with_max_tokens() -> None:
    with pytest.raises(ValidationError):
        ReasoningConfig(effort="high", max_tokens=100)


def test_reasoning_config_effective_enabled() -> None:
    assert ReasoningConfig(effort="low").effective_enabled
    assert not ReasoningConfig(enabled=False).effective_enabled


def test_token_usage_computed_total() -> None:
    usage = TokenUsage(prompt_tokens=2, completion_tokens=3)
    assert usage.computed_total_tokens == 5


def test_token_usage_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1)


def test_token_usage_coerces_strings() -> None:
    usage = TokenUsage.from_raw(prompt_tokens="4")
    assert usage.prompt_tokens == 4


def test_token_usage_rejects_non_numeric_strings() -> None:
    with pytest.raises(ValidationError):
        TokenUsage.from_raw(prompt_tokens="abc")

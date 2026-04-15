from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    GlmReasoning,
    GoogleReasoning,
    ReasoningBudget,
    ThinkingLevel,
)
from dr_llm.llm.providers.usage import TokenUsage


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Message(role="tool", content="nope")  # type: ignore[arg-type]


def test_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Message(role="assistant", content="hi", tool_calls=[])  # type: ignore[call-arg]


def test_effort_spec_values() -> None:
    assert EffortSpec.NA == "na"
    assert EffortSpec.LOW == "low"
    assert EffortSpec.MEDIUM == "medium"
    assert EffortSpec.HIGH == "high"
    assert EffortSpec.MAX == "max"


def test_reasoning_budget_rejects_non_positive_tokens() -> None:
    with pytest.raises(ValidationError):
        ReasoningBudget(tokens=0)


def test_thinking_level_values() -> None:
    assert ThinkingLevel.NA == "na"
    assert ThinkingLevel.OFF == "off"
    assert ThinkingLevel.BUDGET == "budget"
    assert ThinkingLevel.ADAPTIVE == "adaptive"
    assert ThinkingLevel.MINIMAL == "minimal"
    assert ThinkingLevel.LOW == "low"
    assert ThinkingLevel.MEDIUM == "medium"
    assert ThinkingLevel.HIGH == "high"
    assert ThinkingLevel.XHIGH == "xhigh"


def test_google_reasoning_budget_requires_budget_tokens() -> None:
    with pytest.raises(ValidationError):
        GoogleReasoning(thinking_level=ThinkingLevel.BUDGET)
    with pytest.raises(ValidationError):
        GoogleReasoning(
            thinking_level=ThinkingLevel.LOW,
            budget_tokens=512,
        )


def test_anthropic_adaptive_allows_minimal_shape() -> None:
    reasoning = AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    assert reasoning.thinking_level == ThinkingLevel.ADAPTIVE


def test_anthropic_adaptive_rejects_budget_tokens() -> None:
    with pytest.raises(ValidationError):
        AnthropicReasoning(
            thinking_level=ThinkingLevel.ADAPTIVE,
            budget_tokens=2048,
        )


def test_glm_reasoning_accepts_only_off_and_adaptive() -> None:
    assert (
        GlmReasoning(thinking_level=ThinkingLevel.OFF).thinking_level
        == ThinkingLevel.OFF
    )
    assert (
        GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE).thinking_level
        == ThinkingLevel.ADAPTIVE
    )


def test_glm_reasoning_rejects_na_and_other_levels() -> None:
    for thinking_level in (
        ThinkingLevel.NA,
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
        ThinkingLevel.XHIGH,
        ThinkingLevel.BUDGET,
    ):
        with pytest.raises(ValidationError):
            GlmReasoning(thinking_level=thinking_level)


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

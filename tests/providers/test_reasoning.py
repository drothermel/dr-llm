from __future__ import annotations

import pytest

from dr_llm.errors import HeadlessExecutionError, ProviderSemanticError
from dr_llm.providers.anthropic.reasoning import AnthropicReasoningConfig
from dr_llm.providers.google.reasoning import GoogleReasoningConfig
from dr_llm.providers.headless.reasoning import (
    ClaudeHeadlessReasoningConfig,
    CodexHeadlessReasoningConfig,
)
from dr_llm.providers.openai_compat.reasoning import OpenAICompatReasoningConfig
from dr_llm.providers.reasoning import (
    AnthropicReasoning,
    GoogleReasoning,
    ReasoningBudget,
    ReasoningEffort,
    ReasoningOff,
)


def test_openai_compat_serializes_effort() -> None:
    result = OpenAICompatReasoningConfig.from_base(ReasoningEffort(level="high"))
    assert result.to_payload() == {"effort": "high"}


def test_openai_compat_serializes_off() -> None:
    result = OpenAICompatReasoningConfig.from_base(ReasoningOff())
    assert result.to_payload() == {"effort": "none"}


def test_openai_compat_rejects_provider_specific_shape() -> None:
    with pytest.raises(ProviderSemanticError):
        OpenAICompatReasoningConfig.from_base(GoogleReasoning(thinking_level="low"))


def test_anthropic_rejects_reasoning_effort() -> None:
    with pytest.raises(ProviderSemanticError):
        AnthropicReasoningConfig.from_base(ReasoningEffort(level="medium"))


def test_anthropic_serializes_manual_thinking() -> None:
    result = AnthropicReasoningConfig.from_base(
        AnthropicReasoning(
            budget_tokens=2048,
            thinking_mode="enabled",
            display="omitted",
        )
    )
    assert result.thinking_payload() == {
        "type": "enabled",
        "budget_tokens": 2048,
        "display": "omitted",
    }


def test_google_serializes_budget_and_dynamic() -> None:
    assert GoogleReasoningConfig.from_base(ReasoningBudget(tokens=512)).to_payload() == {
        "thinkingBudget": 512
    }
    assert GoogleReasoningConfig.from_base(GoogleReasoning(dynamic=True)).to_payload() == {
        "thinkingBudget": -1
    }


def test_google_serializes_level() -> None:
    assert GoogleReasoningConfig.from_base(ReasoningEffort(level="low")).to_payload() == {
        "thinkingLevel": "low"
    }


def test_claude_headless_serializes_max_effort() -> None:
    with pytest.raises(HeadlessExecutionError):
        ClaudeHeadlessReasoningConfig.from_base(ReasoningEffort(level="max"))


def test_codex_headless_rejects_reasoning() -> None:
    with pytest.raises(HeadlessExecutionError):
        CodexHeadlessReasoningConfig.from_base(ReasoningEffort(level="high"))

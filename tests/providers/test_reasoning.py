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
    CodexReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    ReasoningBudget,
    ThinkingLevel,
)


def test_openai_compat_rejects_anthropic_reasoning_shape() -> None:
    with pytest.raises(ProviderSemanticError):
        OpenAICompatReasoningConfig.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
        )


def test_openai_compat_rejects_provider_specific_shape() -> None:
    with pytest.raises(ProviderSemanticError):
        OpenAICompatReasoningConfig.from_base(GoogleReasoning(thinking_level="low"))


def test_openai_compat_serializes_thinking_levels() -> None:
    assert (
        OpenAICompatReasoningConfig.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.NA)
        ).to_reasoning_effort()
        is None
    )
    assert (
        OpenAICompatReasoningConfig.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.OFF)
        ).to_reasoning_effort()
        == "none"
    )
    assert (
        OpenAICompatReasoningConfig.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL)
        ).to_reasoning_effort()
        == "minimal"
    )
    assert (
        OpenAICompatReasoningConfig.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.XHIGH)
        ).to_reasoning_effort()
        == "xhigh"
    )


def test_anthropic_rejects_non_anthropic_reasoning_config() -> None:
    with pytest.raises(ProviderSemanticError):
        AnthropicReasoningConfig.from_base(GoogleReasoning(thinking_level="low"))


def test_anthropic_serializes_manual_thinking() -> None:
    result = AnthropicReasoningConfig.from_base(
        AnthropicReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=2048,
            display="omitted",
        )
    )
    assert result.thinking_payload() == {
        "type": "enabled",
        "budget_tokens": 2048,
        "display": "omitted",
    }


def test_anthropic_off_omits_thinking() -> None:
    result = AnthropicReasoningConfig.from_base(
        AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
    )
    assert result.thinking_payload() == {}


def test_claude_headless_accepts_adaptive_and_na() -> None:
    assert (
        ClaudeHeadlessReasoningConfig.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
        ).to_cli_args()
        == []
    )
    assert (
        ClaudeHeadlessReasoningConfig.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        ).to_cli_args()
        == []
    )


def test_google_serializes_budget_and_dynamic() -> None:
    assert GoogleReasoningConfig.from_base(
        ReasoningBudget(tokens=512)
    ).to_payload() == {"thinkingBudget": 512}
    assert GoogleReasoningConfig.from_base(
        GoogleReasoning(dynamic=True)
    ).to_payload() == {"thinkingBudget": -1}


def test_google_serializes_level() -> None:
    assert GoogleReasoningConfig.from_base(
        GoogleReasoning(thinking_level="low")
    ).to_payload() == {"thinkingLevel": "low"}


def test_claude_headless_rejects_reasoning_config() -> None:
    with pytest.raises(HeadlessExecutionError):
        ClaudeHeadlessReasoningConfig.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.BUDGET, budget_tokens=1024)
        )


def test_codex_headless_serializes_reasoning_levels() -> None:
    assert (
        CodexHeadlessReasoningConfig.from_base(
            CodexReasoning(thinking_level=ThinkingLevel.NA)
        ).to_cli_args()
        == []
    )
    assert CodexHeadlessReasoningConfig.from_base(
        CodexReasoning(thinking_level=ThinkingLevel.OFF)
    ).to_cli_args() == ["-c", 'model_reasoning_effort="none"']
    assert CodexHeadlessReasoningConfig.from_base(
        CodexReasoning(thinking_level=ThinkingLevel.HIGH)
    ).to_cli_args() == ["-c", 'model_reasoning_effort="high"']
    assert CodexHeadlessReasoningConfig.from_base(
        CodexReasoning(thinking_level=ThinkingLevel.XHIGH)
    ).to_cli_args() == ["-c", 'model_reasoning_effort="xhigh"']


def test_codex_headless_rejects_non_codex_reasoning() -> None:
    with pytest.raises(HeadlessExecutionError):
        CodexHeadlessReasoningConfig.from_base(ReasoningBudget(tokens=1024))

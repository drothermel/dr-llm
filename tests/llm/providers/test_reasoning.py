from __future__ import annotations

import pytest

from dr_llm.errors import HeadlessExecutionError, ProviderSemanticError
from dr_llm.llm.providers.anthropic.reasoning import AnthropicReasoningConfig
from dr_llm.llm.providers.anthropic.reasoning import KimiCodeReasoningConfig
from dr_llm.llm.providers.anthropic.reasoning import MiniMaxReasoningConfig
from dr_llm.llm.providers.anthropic.reasoning import validate_reasoning_for_kimi_code
from dr_llm.llm.providers.anthropic.reasoning import validate_reasoning_for_minimax
from dr_llm.llm.providers.google.reasoning import GoogleReasoningConfig
from dr_llm.llm.providers.headless.reasoning import (
    ClaudeHeadlessReasoningConfig,
    CodexHeadlessReasoningConfig,
)
from dr_llm.llm.providers.openai_compat.reasoning import OpenAICompatReasoningConfig
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GlmReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
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
        OpenAICompatReasoningConfig.from_base(
            GoogleReasoning(thinking_level=ThinkingLevel.LOW)
        )


def test_openai_compat_serializes_thinking_levels() -> None:
    assert (
        OpenAICompatReasoningConfig.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.NA)
        ).reasoning_effort
        is None
    )
    assert (
        OpenAICompatReasoningConfig.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.OFF)
        ).reasoning_effort
        == "none"
    )
    assert (
        OpenAICompatReasoningConfig.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL)
        ).reasoning_effort
        == "minimal"
    )
    assert OpenAICompatReasoningConfig.from_base(
        GlmReasoning(thinking_level=ThinkingLevel.OFF)
    ).extra_body == {"thinking": {"type": "disabled"}}
    assert OpenAICompatReasoningConfig.from_base(
        GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).extra_body == {"thinking": {"type": "enabled"}}


def test_openrouter_serializes_reasoning_payloads() -> None:
    assert OpenAICompatReasoningConfig.from_base(
        OpenRouterReasoning(enabled=False),
        provider="openrouter",
        model="deepseek/deepseek-chat-v3.1",
    ).extra_body == {"reasoning": {"enabled": False}}
    assert OpenAICompatReasoningConfig.from_base(
        OpenRouterReasoning(effort="low"),
        provider="openrouter",
        model="openai/gpt-oss-20b",
    ).extra_body == {"reasoning": {"effort": "low"}}


def test_anthropic_rejects_non_anthropic_reasoning_config() -> None:
    with pytest.raises(ProviderSemanticError):
        AnthropicReasoningConfig.from_base(
            GoogleReasoning(thinking_level=ThinkingLevel.LOW)
        )


def test_anthropic_serializes_manual_thinking() -> None:
    result = AnthropicReasoningConfig.from_base(
        AnthropicReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=2048,
            display="omitted",
        )
    )
    assert result.thinking == {
        "type": "enabled",
        "budget_tokens": 2048,
        "display": "omitted",
    }


def test_anthropic_off_omits_thinking() -> None:
    result = AnthropicReasoningConfig.from_base(
        AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
    )
    assert result.thinking == {}


def test_kimi_code_serializes_supported_reasoning_controls() -> None:
    assert (
        KimiCodeReasoningConfig.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        ).thinking
        == {}
    )
    assert KimiCodeReasoningConfig.from_base(
        AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
    ).thinking == {"type": "disabled"}
    assert KimiCodeReasoningConfig.from_base(
        AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).thinking == {"type": "adaptive"}
    assert KimiCodeReasoningConfig.from_base(
        AnthropicReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=1024,
        )
    ).thinking == {"type": "enabled", "budget_tokens": 1024}


def test_kimi_code_validation_rejects_unsupported_anthropic_levels() -> None:
    for thinking_level in (
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    ):
        with pytest.raises(
            ValueError,
            match=(
                "kimi-code supports only anthropic thinking levels "
                "'na', 'off', 'adaptive', and 'budget'"
            ),
        ):
            validate_reasoning_for_kimi_code(
                model="kimi-for-coding",
                reasoning=AnthropicReasoning(thinking_level=thinking_level),
            )


def test_minimax_validation_and_serializer_both_require_explicit_na() -> None:
    with pytest.raises(
        ValueError,
        match="reasoning is required for provider='minimax' model='MiniMax-M2.7'",
    ):
        validate_reasoning_for_minimax(model="MiniMax-M2.7", reasoning=None)

    with pytest.raises(
        ProviderSemanticError,
        match="minimax requires explicit AnthropicReasoning\\(thinking_level='na'\\)",
    ):
        MiniMaxReasoningConfig.from_base(None)


def test_claude_headless_accepts_adaptive_and_na() -> None:
    assert (
        ClaudeHeadlessReasoningConfig.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
        ).cli_args
        == []
    )
    assert (
        ClaudeHeadlessReasoningConfig.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        ).cli_args
        == []
    )


def test_google_serializes_budget_family_controls() -> None:
    assert GoogleReasoningConfig.from_base(ReasoningBudget(tokens=512)).payload == {
        "thinkingBudget": 512
    }
    assert GoogleReasoningConfig.from_base(
        GoogleReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).payload == {"thinkingBudget": -1}
    assert GoogleReasoningConfig.from_base(
        GoogleReasoning(thinking_level=ThinkingLevel.OFF)
    ).payload == {"thinkingBudget": 0}
    assert GoogleReasoningConfig.from_base(
        GoogleReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=1024,
            include_thoughts=True,
        )
    ).payload == {"thinkingBudget": 1024, "includeThoughts": True}


def test_google_serializes_level() -> None:
    assert GoogleReasoningConfig.from_base(
        GoogleReasoning(thinking_level=ThinkingLevel.LOW)
    ).payload == {"thinkingLevel": "low"}


def test_claude_headless_rejects_reasoning_config() -> None:
    with pytest.raises(HeadlessExecutionError):
        ClaudeHeadlessReasoningConfig.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.BUDGET, budget_tokens=1024)
        )


def test_codex_headless_serializes_reasoning_levels() -> None:
    assert (
        CodexHeadlessReasoningConfig.from_base(
            CodexReasoning(thinking_level=ThinkingLevel.NA)
        ).cli_args
        == []
    )
    assert CodexHeadlessReasoningConfig.from_base(
        CodexReasoning(thinking_level=ThinkingLevel.OFF)
    ).cli_args == ["-c", 'model_reasoning_effort="none"']
    assert CodexHeadlessReasoningConfig.from_base(
        CodexReasoning(thinking_level=ThinkingLevel.HIGH)
    ).cli_args == ["-c", 'model_reasoning_effort="high"']


def test_codex_headless_rejects_non_codex_reasoning() -> None:
    with pytest.raises(HeadlessExecutionError):
        CodexHeadlessReasoningConfig.from_base(ReasoningBudget(tokens=1024))

from __future__ import annotations

import pytest

from dr_llm.errors import HeadlessExecutionError, ProviderSemanticError
from dr_llm.llm import ProviderName
from dr_llm.llm.providers.impls.anthropic.controls import (
    AnthropicControlMapping,
    _validate_reasoning_for_anthropic,
)
from dr_llm.llm.providers.impls.kimi_code.controls import (
    KimiCodeControlMapping,
)
from dr_llm.llm.providers.impls.kimi_code.controls import (
    _validate_reasoning_for_kimi_code,
)
from dr_llm.llm.providers.impls.minimax.controls import MiniMaxControlMapping
from dr_llm.llm.providers.impls.minimax.controls import (
    _validate_reasoning_for_minimax,
)
from dr_llm.llm.providers.impls.google.controls import GoogleControlMapping
from dr_llm.llm.providers.impls.claude_code.controls import (
    ClaudeHeadlessControlMapping,
    _validate_reasoning_for_claude_code,
)
from dr_llm.llm.providers.impls.codex.controls import (
    CodexHeadlessControlMapping,
)
from dr_llm.llm.providers.impls.glm.controls import GlmControlMapping
from dr_llm.llm.providers.impls.openai.controls import OpenAIControlMapping
from dr_llm.llm.providers.impls.openrouter.controls import (
    OpenRouterControlMapping,
)
from dr_llm.llm.names import ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GlmReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningBudget,
)


def test_openai_rejects_anthropic_reasoning_shape() -> None:
    with pytest.raises(ProviderSemanticError):
        OpenAIControlMapping.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
        )


def test_openai_rejects_provider_specific_shape() -> None:
    with pytest.raises(ProviderSemanticError):
        OpenAIControlMapping.from_base(
            GoogleReasoning(thinking_level=ThinkingLevel.LOW)
        )


def test_provider_controls_serialize_openai_compat_payloads() -> None:
    assert (
        OpenAIControlMapping.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.NA)
        ).reasoning_effort
        is None
    )
    assert (
        OpenAIControlMapping.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.OFF)
        ).reasoning_effort
        == "none"
    )
    assert (
        OpenAIControlMapping.from_base(
            OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL)
        ).reasoning_effort
        == "minimal"
    )
    assert GlmControlMapping.from_base(
        GlmReasoning(thinking_level=ThinkingLevel.OFF)
    ).extra_body == {"thinking": {"type": "disabled"}}
    assert GlmControlMapping.from_base(
        GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).extra_body == {"thinking": {"type": "enabled"}}


def test_openrouter_serializes_reasoning_payloads() -> None:
    assert OpenRouterControlMapping.from_base(
        OpenRouterReasoning(enabled=False),
    ).extra_body == {"reasoning": {"enabled": False}}
    assert OpenRouterControlMapping.from_base(
        OpenRouterReasoning(effort="low"),
    ).extra_body == {"reasoning": {"effort": "low"}}


def test_anthropic_rejects_non_anthropic_reasoning_config() -> None:
    with pytest.raises(ProviderSemanticError):
        AnthropicControlMapping.from_base(
            GoogleReasoning(thinking_level=ThinkingLevel.LOW)
        )


def test_anthropic_serializes_manual_thinking() -> None:
    result = AnthropicControlMapping.from_base(
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
    result = AnthropicControlMapping.from_base(
        AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
    )
    assert result.thinking == {}


def test_kimi_code_serializes_supported_reasoning_settings() -> None:
    assert (
        KimiCodeControlMapping.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        ).thinking
        == {}
    )
    assert KimiCodeControlMapping.from_base(
        AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
    ).thinking == {"type": "disabled"}
    assert KimiCodeControlMapping.from_base(
        AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).thinking == {"type": "adaptive"}
    assert KimiCodeControlMapping.from_base(
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
        ThinkingLevel.XHIGH,
    ):
        with pytest.raises(
            ValueError,
            match=(
                "kimi-code supports only anthropic thinking levels "
                "'na', 'off', 'adaptive', and 'budget'"
            ),
        ):
            _validate_reasoning_for_kimi_code(
                model="kimi-for-coding",
                reasoning=AnthropicReasoning(thinking_level=thinking_level),
            )


def test_kimi_code_validation_rejects_budget_tokens_without_budget_level() -> (
    None
):
    with pytest.raises(ValueError, match=r"budget_tokens.*thinking_level"):
        _validate_reasoning_for_kimi_code(
            model="kimi-for-coding",
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=1024,
            ),
        )


def test_anthropic_validation_rejects_budget_tokens_without_budget_level() -> (
    None
):
    reasoning = AnthropicReasoning.model_construct(
        thinking_level=ThinkingLevel.ADAPTIVE,
        budget_tokens=1024,
    )

    with pytest.raises(ValueError, match=r"budget_tokens.*thinking_level"):
        _validate_reasoning_for_anthropic(
            model="claude-sonnet-4-6",
            reasoning=reasoning,
        )


def test_minimax_validation_and_serializer_both_require_explicit_na() -> None:
    with pytest.raises(
        ValueError,
        match=f"reasoning is required for provider='{ProviderName.MINIMAX}' model='MiniMax-M2.7'",
    ):
        _validate_reasoning_for_minimax(model="MiniMax-M2.7", reasoning=None)

    with pytest.raises(
        ProviderSemanticError,
        match=f"{ProviderName.MINIMAX} requires explicit AnthropicReasoning\\(thinking_level='na'\\)",
    ):
        MiniMaxControlMapping.from_base(None)


def test_minimax_validation_rejects_anthropic_budget_tokens() -> None:
    with pytest.raises(ValueError, match="budget_tokens"):
        _validate_reasoning_for_minimax(
            model="MiniMax-M2.7",
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                budget_tokens=1024,
            ),
        )


def test_claude_headless_accepts_adaptive_and_na() -> None:
    assert (
        ClaudeHeadlessControlMapping.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
        ).cli_args
        == []
    )
    assert (
        ClaudeHeadlessControlMapping.from_base(
            AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        ).cli_args
        == []
    )


def test_claude_headless_validation_rejects_budget_tokens() -> None:
    with pytest.raises(ValueError, match="budget_tokens"):
        _validate_reasoning_for_claude_code(
            model="claude-sonnet-4-6",
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=1024,
            ),
        )


def test_google_serializes_budget_family_controls() -> None:
    assert GoogleControlMapping.from_base(
        ReasoningBudget(tokens=512)
    ).payload == {"thinkingBudget": 512}
    assert GoogleControlMapping.from_base(
        GoogleReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).payload == {"thinkingBudget": -1}
    assert GoogleControlMapping.from_base(
        GoogleReasoning(thinking_level=ThinkingLevel.OFF)
    ).payload == {"thinkingBudget": 0}
    assert GoogleControlMapping.from_base(
        GoogleReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=1024,
            include_thoughts=True,
        )
    ).payload == {"thinkingBudget": 1024, "includeThoughts": True}


def test_google_serializes_level() -> None:
    assert GoogleControlMapping.from_base(
        GoogleReasoning(thinking_level=ThinkingLevel.LOW)
    ).payload == {"thinkingLevel": "low"}


def test_claude_headless_rejects_reasoning_config() -> None:
    with pytest.raises(HeadlessExecutionError):
        ClaudeHeadlessControlMapping.from_base(
            AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET, budget_tokens=1024
            )
        )


def test_codex_headless_serializes_reasoning_levels() -> None:
    assert (
        CodexHeadlessControlMapping.from_base(
            CodexReasoning(thinking_level=ThinkingLevel.NA)
        ).cli_args
        == []
    )
    assert CodexHeadlessControlMapping.from_base(
        CodexReasoning(thinking_level=ThinkingLevel.OFF)
    ).cli_args == ["-c", 'model_reasoning_effort="none"']
    assert CodexHeadlessControlMapping.from_base(
        CodexReasoning(thinking_level=ThinkingLevel.HIGH)
    ).cli_args == ["-c", 'model_reasoning_effort="high"']
    assert CodexHeadlessControlMapping.from_base(
        CodexReasoning(thinking_level=ThinkingLevel.XHIGH)
    ).cli_args == ["-c", 'model_reasoning_effort="xhigh"']


def test_codex_headless_rejects_non_codex_reasoning() -> None:
    with pytest.raises(HeadlessExecutionError):
        CodexHeadlessControlMapping.from_base(ReasoningBudget(tokens=1024))

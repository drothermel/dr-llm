from __future__ import annotations

import pytest

from dr_llm.errors import HeadlessExecutionError, ProviderSemanticError
from dr_llm.llm import ProviderName
from dr_llm.llm.providers.impls.anthropic.controls import (
    _validate_reasoning_for_anthropic,
)
from dr_llm.llm.providers.impls.kimi_code.controls import (
    _validate_reasoning_for_kimi_code,
)
from dr_llm.llm.providers.impls.minimax.controls import (
    _validate_reasoning_for_minimax,
)
from dr_llm.llm.providers.impls.claude_code.controls import (
    _validate_reasoning_for_claude_code,
)
from dr_llm.llm.providers.impls.anthropic.request_controls import (
    AnthropicRequestControls,
)
from dr_llm.llm.providers.impls.claude_code.request_controls import (
    ClaudeCodeRequestControls,
)
from dr_llm.llm.providers.impls.codex.request_controls import (
    CodexRequestControls,
)
from dr_llm.llm.providers.impls.glm.request_controls import GlmRequestControls
from dr_llm.llm.providers.impls.google.request_controls import (
    GoogleRequestControls,
)
from dr_llm.llm.providers.impls.kimi_code.request_controls import (
    KimiCodeRequestControls,
)
from dr_llm.llm.providers.impls.minimax.request_controls import (
    MiniMaxRequestControls,
)
from dr_llm.llm.providers.impls.openai.request_controls import (
    OpenAIRequestControls,
)
from dr_llm.llm.providers.impls.openrouter.request_controls import (
    OpenRouterRequestControls,
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
        OpenAIRequestControls.from_reasoning(
            AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
        )


def test_openai_rejects_provider_specific_shape() -> None:
    with pytest.raises(ProviderSemanticError):
        OpenAIRequestControls.from_reasoning(
            GoogleReasoning(thinking_level=ThinkingLevel.LOW)
        )


def test_provider_controls_serialize_openai_compat_payloads() -> None:
    assert (
        OpenAIRequestControls.from_reasoning(
            OpenAIReasoning(thinking_level=ThinkingLevel.NA)
        ).reasoning_effort
        is None
    )
    assert (
        OpenAIRequestControls.from_reasoning(
            OpenAIReasoning(thinking_level=ThinkingLevel.OFF)
        ).reasoning_effort
        == "none"
    )
    assert (
        OpenAIRequestControls.from_reasoning(
            OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL)
        ).reasoning_effort
        == "minimal"
    )
    assert GlmRequestControls.from_reasoning(
        GlmReasoning(thinking_level=ThinkingLevel.OFF)
    ).extra_body == {"thinking": {"type": "disabled"}}
    assert GlmRequestControls.from_reasoning(
        GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).extra_body == {"thinking": {"type": "enabled"}}


def test_openrouter_serializes_reasoning_payloads() -> None:
    assert OpenRouterRequestControls.from_reasoning(
        OpenRouterReasoning(enabled=False),
    ).extra_body == {"reasoning": {"enabled": False}}
    assert OpenRouterRequestControls.from_reasoning(
        OpenRouterReasoning(effort="low"),
    ).extra_body == {"reasoning": {"effort": "low"}}


def test_anthropic_rejects_non_anthropic_reasoning_config() -> None:
    with pytest.raises(ProviderSemanticError):
        AnthropicRequestControls.from_reasoning(
            GoogleReasoning(thinking_level=ThinkingLevel.LOW)
        )


def test_anthropic_serializes_manual_thinking() -> None:
    result = AnthropicRequestControls.from_reasoning(
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
    result = AnthropicRequestControls.from_reasoning(
        AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
    )
    assert result.thinking == {}


def test_kimi_code_serializes_supported_reasoning_settings() -> None:
    assert (
        KimiCodeRequestControls.from_reasoning(
            AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        ).thinking
        == {}
    )
    assert KimiCodeRequestControls.from_reasoning(
        AnthropicReasoning(thinking_level=ThinkingLevel.OFF)
    ).thinking == {"type": "disabled"}
    assert KimiCodeRequestControls.from_reasoning(
        AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).thinking == {"type": "adaptive"}
    assert KimiCodeRequestControls.from_reasoning(
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
        MiniMaxRequestControls.from_reasoning(None)


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
        ClaudeCodeRequestControls.from_reasoning(
            AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
        ).cli_args
        == []
    )
    assert (
        ClaudeCodeRequestControls.from_reasoning(
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
    assert GoogleRequestControls.from_reasoning(
        ReasoningBudget(tokens=512)
    ).payload == {"thinkingBudget": 512}
    assert GoogleRequestControls.from_reasoning(
        GoogleReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
    ).payload == {"thinkingBudget": -1}
    assert GoogleRequestControls.from_reasoning(
        GoogleReasoning(thinking_level=ThinkingLevel.OFF)
    ).payload == {"thinkingBudget": 0}
    assert GoogleRequestControls.from_reasoning(
        GoogleReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=1024,
            include_thoughts=True,
        )
    ).payload == {"thinkingBudget": 1024, "includeThoughts": True}


def test_google_serializes_level() -> None:
    assert GoogleRequestControls.from_reasoning(
        GoogleReasoning(thinking_level=ThinkingLevel.LOW)
    ).payload == {"thinkingLevel": "low"}


def test_claude_headless_rejects_reasoning_config() -> None:
    with pytest.raises(HeadlessExecutionError):
        ClaudeCodeRequestControls.from_reasoning(
            AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET, budget_tokens=1024
            )
        )


def test_codex_headless_serializes_reasoning_levels() -> None:
    assert (
        CodexRequestControls.from_reasoning(
            CodexReasoning(thinking_level=ThinkingLevel.NA)
        ).cli_args
        == []
    )
    assert CodexRequestControls.from_reasoning(
        CodexReasoning(thinking_level=ThinkingLevel.OFF)
    ).cli_args == ["-c", 'model_reasoning_effort="none"']
    assert CodexRequestControls.from_reasoning(
        CodexReasoning(thinking_level=ThinkingLevel.HIGH)
    ).cli_args == ["-c", 'model_reasoning_effort="high"']
    assert CodexRequestControls.from_reasoning(
        CodexReasoning(thinking_level=ThinkingLevel.XHIGH)
    ).cli_args == ["-c", 'model_reasoning_effort="xhigh"']


def test_codex_headless_rejects_non_codex_reasoning() -> None:
    with pytest.raises(HeadlessExecutionError):
        CodexRequestControls.from_reasoning(ReasoningBudget(tokens=1024))

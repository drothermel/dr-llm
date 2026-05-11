from __future__ import annotations

from collections.abc import Generator

import pytest

from dr_llm.llm import (
    AnthropicReasoning,
    CodexReasoning,
    EffortSpec,
    GlmReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ProviderName,
    ThinkingLevel,
    build_default_registry,
)
from dr_llm.llm.providers.core.controls import ProviderControls
from dr_llm.llm.providers.core.registry import ProviderRegistry


@pytest.fixture
def registry() -> Generator[ProviderRegistry]:
    reg = build_default_registry()
    try:
        yield reg
    finally:
        reg.close()


def controls(
    registry: ProviderRegistry, provider: ProviderName, model: str
) -> ProviderControls:
    return registry.get(provider).controls(model)


def test_openai_controls_prefer_minimal_or_off_defaults(
    registry: ProviderRegistry,
) -> None:
    gpt5 = controls(registry, ProviderName.OPENAI, "gpt-5-mini")
    assert gpt5.supported_thinking_levels == (
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    )
    assert gpt5.default_reasoning == OpenAIReasoning(
        thinking_level=ThinkingLevel.MINIMAL
    )

    gpt54 = controls(registry, ProviderName.OPENAI, "gpt-5.4-mini")
    assert gpt54.default_thinking_level == ThinkingLevel.OFF
    assert gpt54.default_reasoning == OpenAIReasoning(
        thinking_level=ThinkingLevel.OFF
    )


def test_google_controls_cover_budget_and_level_modes(
    registry: ProviderRegistry,
) -> None:
    flash = controls(registry, ProviderName.GOOGLE, "gemini-2.5-flash")
    assert flash.supported_thinking_levels == (
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.OFF,
        ThinkingLevel.BUDGET,
    )
    assert flash.default_reasoning == GoogleReasoning(
        thinking_level=ThinkingLevel.OFF
    )

    gemma = controls(registry, ProviderName.GOOGLE, "gemma-4-26b-a4b-it")
    assert gemma.supported_thinking_levels == (
        ThinkingLevel.MINIMAL,
        ThinkingLevel.HIGH,
    )
    assert gemma.default_reasoning == GoogleReasoning(
        thinking_level=ThinkingLevel.MINIMAL
    )


def test_reasoning_for_budget_level_requires_explicit_tokens(
    registry: ProviderRegistry,
) -> None:
    orchestrator = registry.get(ProviderName.GOOGLE)
    with pytest.raises(ValueError, match="budget thinking requires"):
        orchestrator.controls("gemini-2.5-flash").reasoning_for_thinking_level(
            thinking_level=ThinkingLevel.BUDGET,
        )

    assert orchestrator.controls(
        "gemini-2.5-flash"
    ).reasoning_for_thinking_level(
        thinking_level=ThinkingLevel.BUDGET,
        budget_tokens=128,
    ) == GoogleReasoning(
        thinking_level=ThinkingLevel.BUDGET,
        budget_tokens=128,
    )


def test_codex_and_glm_controls_use_provider_native_specs(
    registry: ProviderRegistry,
) -> None:
    codex = controls(registry, ProviderName.CODEX, "gpt-5.4-mini")
    assert codex.supported_thinking_levels == (
        ThinkingLevel.OFF,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
        ThinkingLevel.XHIGH,
    )
    assert codex.default_reasoning == CodexReasoning(
        thinking_level=ThinkingLevel.OFF
    )

    glm = controls(registry, ProviderName.GLM, "glm-4.5")
    assert glm.supported_thinking_levels == (
        ThinkingLevel.OFF,
        ThinkingLevel.ADAPTIVE,
    )
    assert glm.default_reasoning == GlmReasoning(
        thinking_level=ThinkingLevel.OFF
    )


def test_anthropic_style_controls_cover_effort_and_required_na(
    registry: ProviderRegistry,
) -> None:
    anthropic = controls(
        registry, ProviderName.ANTHROPIC, "claude-sonnet-4-20250514"
    )
    assert anthropic.supported_thinking_levels == (
        ThinkingLevel.OFF,
        ThinkingLevel.BUDGET,
    )
    assert anthropic.default_reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.OFF
    )

    claude = controls(registry, ProviderName.CLAUDE_CODE, "claude-sonnet-4-6")
    assert claude.default_effort == EffortSpec.LOW
    assert claude.default_reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.ADAPTIVE
    )

    minimax = controls(registry, ProviderName.MINIMAX, "MiniMax-M2")
    assert minimax.default_effort == EffortSpec.LOW
    assert minimax.default_reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.NA
    )

    kimi = controls(registry, ProviderName.KIMI_CODE, "kimi-for-coding")
    assert kimi.default_effort == EffortSpec.LOW
    assert kimi.default_reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.OFF
    )


def test_openrouter_controls_follow_curated_policy(
    registry: ProviderRegistry,
) -> None:
    orchestrator = registry.get(ProviderName.OPENROUTER)
    assert (
        orchestrator.controls("deepseek/deepseek-chat").default_reasoning
        is None
    )
    assert orchestrator.controls(
        "deepseek/deepseek-chat-v3.1"
    ).default_reasoning == OpenRouterReasoning(enabled=False)
    assert orchestrator.controls(
        "deepseek/deepseek-r1"
    ).default_reasoning == OpenRouterReasoning(enabled=True)
    assert orchestrator.controls(
        "openai/gpt-oss-20b"
    ).default_reasoning == OpenRouterReasoning(effort="low")
    assert orchestrator.controls(
        "openai/gpt-5-nano"
    ).default_reasoning == OpenRouterReasoning(effort="low")
    assert orchestrator.controls(
        "openai/gpt-5.4-nano"
    ).default_reasoning == OpenRouterReasoning(effort="low")


def test_reasoning_controls_model_collects_all_defaults(
    registry: ProviderRegistry,
) -> None:
    google = controls(registry, ProviderName.GOOGLE, "gemini-2.5-flash")

    assert google.supported_thinking_levels == (
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.OFF,
        ThinkingLevel.BUDGET,
    )
    assert google.default_thinking_level == ThinkingLevel.OFF
    assert google.supported_effort_levels == ()
    assert google.default_effort == EffortSpec.NA
    assert google.default_reasoning == GoogleReasoning(
        thinking_level=ThinkingLevel.OFF
    )

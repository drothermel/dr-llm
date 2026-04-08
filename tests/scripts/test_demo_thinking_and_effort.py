from __future__ import annotations

import demo_thinking_and_effort as script
from _demo_thinking_models import (
    CLAUDE_MODELS,
    CODEX_MODELS,
    GOOGLE_MODELS,
    KIMI_CODE_MODELS,
    MINIMAX_MODELS,
    OPENAI_MODELS,
    OPENROUTER_MODELS,
)

from dr_llm.llm.providers.openrouter.policy import openrouter_allowed_models
from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ThinkingLevel,
)


def test_provider_model_sweeps_match_expected_snapshot() -> None:
    assert CLAUDE_MODELS == [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]
    assert OPENAI_MODELS == [
        "gpt-5.4-mini-2026-03-17",
        "gpt-5-mini-2025-08-07",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4o-mini-2024-07-18",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5-nano-2025-08-07",
        "gpt-4.1-nano-2025-04-14",
    ]
    assert CODEX_MODELS == [
        "gpt-5.1-codex-mini",
    ]
    assert KIMI_CODE_MODELS == [
        "kimi-for-coding",
    ]
    assert MINIMAX_MODELS == [
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ]
    assert GOOGLE_MODELS == [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash-lite",
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemma-3n-e4b-it",
        "gemma-3n-e2b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    ]
    assert OPENROUTER_MODELS == list(openrouter_allowed_models())
    assert script.PHASES == ["models", "thinking", "effort"]


def test_claude_code_uses_model_specific_thinking_and_effort_levels() -> None:
    opus_levels = script.supported_thinking_levels("claude-code", "claude-opus-4-6")
    assert opus_levels == [ThinkingLevel.ADAPTIVE]
    assert (
        script.default_thinking_for_model("claude-code", "claude-opus-4-6")
        == ThinkingLevel.ADAPTIVE
    )
    assert (
        script.default_effort_for_model("claude-code", "claude-opus-4-6")
        == EffortSpec.LOW
    )
    assert script.supported_effort_levels(
        provider="claude-code",
        model="claude-opus-4-6",
    ) == (
        EffortSpec.LOW,
        EffortSpec.MEDIUM,
        EffortSpec.HIGH,
        EffortSpec.MAX,
    )

    haiku_levels = script.supported_thinking_levels(
        "claude-code",
        "claude-haiku-4-5-20251001",
    )
    assert haiku_levels == [ThinkingLevel.NA]
    assert (
        script.default_thinking_for_model("claude-code", "claude-haiku-4-5-20251001")
        == ThinkingLevel.NA
    )
    assert (
        script.default_effort_for_model("claude-code", "claude-haiku-4-5-20251001")
        == EffortSpec.NA
    )
    assert (
        script.supported_effort_levels(
            provider="claude-code",
            model="claude-haiku-4-5-20251001",
        )
        == ()
    )

    adaptive_reasoning = script.reasoning_for_level(
        "claude-code",
        ThinkingLevel.ADAPTIVE,
    )
    assert isinstance(adaptive_reasoning, AnthropicReasoning)
    assert adaptive_reasoning.thinking_level == ThinkingLevel.ADAPTIVE

    explicit_na_reasoning = script.reasoning_for_level(
        "claude-code",
        ThinkingLevel.NA,
        explicit=True,
    )
    assert isinstance(explicit_na_reasoning, AnthropicReasoning)
    assert explicit_na_reasoning.thinking_level == ThinkingLevel.NA
    assert script.reasoning_for_level("claude-code", ThinkingLevel.NA) is None


def test_kimi_code_uses_explicit_reasoning_and_required_effort() -> None:
    kimi_levels = script.supported_thinking_levels("kimi-code", "kimi-for-coding")
    assert kimi_levels == [
        ThinkingLevel.OFF,
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.BUDGET,
    ]
    assert (
        script.default_thinking_for_model("kimi-code", "kimi-for-coding")
        == ThinkingLevel.NA
    )
    assert (
        script.default_effort_for_model("kimi-code", "kimi-for-coding")
        == EffortSpec.LOW
    )
    assert script.supported_effort_levels(
        provider="kimi-code",
        model="kimi-for-coding",
    ) == (
        EffortSpec.LOW,
        EffortSpec.MEDIUM,
        EffortSpec.HIGH,
        EffortSpec.MAX,
    )

    off_reasoning = script.reasoning_for_level("kimi-code", ThinkingLevel.OFF)
    adaptive_reasoning = script.reasoning_for_level("kimi-code", ThinkingLevel.ADAPTIVE)
    budget_reasoning = script.reasoning_for_level("kimi-code", ThinkingLevel.BUDGET)
    assert isinstance(off_reasoning, AnthropicReasoning)
    assert off_reasoning.thinking_level == ThinkingLevel.OFF
    assert isinstance(adaptive_reasoning, AnthropicReasoning)
    assert adaptive_reasoning.thinking_level == ThinkingLevel.ADAPTIVE
    assert isinstance(budget_reasoning, AnthropicReasoning)
    assert budget_reasoning.thinking_level == ThinkingLevel.BUDGET
    assert budget_reasoning.budget_tokens == script.KIMI_CODE_FIXED_BUDGET


def test_minimax_requires_explicit_na_reasoning_and_effort() -> None:
    minimax_levels = script.supported_thinking_levels("minimax", "MiniMax-M2.7")
    assert minimax_levels == [ThinkingLevel.NA]
    assert (
        script.default_thinking_for_model("minimax", "MiniMax-M2.7") == ThinkingLevel.NA
    )
    assert script.default_effort_for_model("minimax", "MiniMax-M2.7") == EffortSpec.LOW
    assert script.supported_effort_levels(
        provider="minimax",
        model="MiniMax-M2.7",
    ) == (
        EffortSpec.LOW,
        EffortSpec.MEDIUM,
        EffortSpec.HIGH,
        EffortSpec.MAX,
    )

    explicit_reasoning = script.reasoning_for_level(
        "minimax",
        ThinkingLevel.NA,
        explicit=True,
    )
    assert isinstance(explicit_reasoning, AnthropicReasoning)
    assert explicit_reasoning.thinking_level == ThinkingLevel.NA
    assert script.reasoning_for_level("minimax", ThinkingLevel.NA) is None
    assert script.requires_explicit_reasoning("minimax") is True


def test_openai_and_codex_use_explicit_thinking_levels_only() -> None:
    openai_levels = script.supported_thinking_levels("openai", "gpt-5-mini-2025-08-07")
    assert openai_levels == [
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    ]
    assert (
        script.default_thinking_for_model("openai", "gpt-5-mini-2025-08-07")
        == ThinkingLevel.MINIMAL
    )

    codex_levels = script.supported_thinking_levels("codex", "gpt-5.1-codex-mini")
    assert codex_levels == [
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    ]
    assert (
        script.default_thinking_for_model("codex", "gpt-5.1-codex-mini")
        == ThinkingLevel.LOW
    )

    openai_reasoning = script.reasoning_for_level("openai", ThinkingLevel.HIGH)
    codex_reasoning = script.reasoning_for_level("codex", ThinkingLevel.MEDIUM)
    assert isinstance(openai_reasoning, OpenAIReasoning)
    assert openai_reasoning.thinking_level == ThinkingLevel.HIGH
    assert isinstance(codex_reasoning, CodexReasoning)
    assert codex_reasoning.thinking_level == ThinkingLevel.MEDIUM


def test_google_uses_budget_or_level_controls_by_model_family() -> None:
    budget_levels = script.supported_thinking_levels("google", "gemini-2.5-flash")
    assert budget_levels == [
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.OFF,
        ThinkingLevel.BUDGET,
    ]
    assert (
        script.default_thinking_for_model("google", "gemini-2.5-flash")
        == ThinkingLevel.OFF
    )

    level_levels = script.supported_thinking_levels("google", "gemini-3-flash-preview")
    assert level_levels == [
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    ]
    assert (
        script.default_thinking_for_model("google", "gemini-3-flash-preview")
        == ThinkingLevel.MINIMAL
    )

    unsupported_levels = script.supported_thinking_levels("google", "gemma-3-1b-it")
    assert unsupported_levels == [ThinkingLevel.NA]
    assert (
        script.default_thinking_for_model("google", "gemma-3-1b-it") == ThinkingLevel.NA
    )

    gemma_4_levels = script.supported_thinking_levels("google", "gemma-4-31b-it")
    assert gemma_4_levels == [
        ThinkingLevel.MINIMAL,
        ThinkingLevel.HIGH,
    ]
    assert (
        script.default_thinking_for_model("google", "gemma-4-31b-it")
        == ThinkingLevel.MINIMAL
    )

    budget_reasoning = script.reasoning_for_level("google", ThinkingLevel.BUDGET)
    level_reasoning = script.reasoning_for_level("google", ThinkingLevel.LOW)
    assert isinstance(budget_reasoning, GoogleReasoning)
    assert budget_reasoning.thinking_level == ThinkingLevel.BUDGET
    assert budget_reasoning.budget_tokens == script.GOOGLE_FIXED_BUDGET
    assert isinstance(level_reasoning, GoogleReasoning)
    assert level_reasoning.thinking_level == ThinkingLevel.LOW


def test_openrouter_uses_minimum_policy_reasoning_in_model_sweep() -> None:
    assert script.supported_thinking_levels(
        "openrouter", "deepseek/deepseek-chat-v3.1"
    ) == [ThinkingLevel.NA]

    hybrid_reasoning = script.default_reasoning_override(
        "openrouter",
        "deepseek/deepseek-chat-v3.1",
    )
    reasoning_only = script.default_reasoning_override(
        "openrouter",
        "deepseek/deepseek-r1",
    )
    unsupported_reasoning = script.default_reasoning_override(
        "openrouter",
        "deepseek/deepseek-chat",
    )
    effort_reasoning = script.default_reasoning_override(
        "openrouter",
        "openai/gpt-oss-20b",
    )

    assert isinstance(hybrid_reasoning, OpenRouterReasoning)
    assert hybrid_reasoning.enabled is False
    assert isinstance(reasoning_only, OpenRouterReasoning)
    assert reasoning_only.enabled is True
    assert unsupported_reasoning is None
    assert isinstance(effort_reasoning, OpenRouterReasoning)
    assert effort_reasoning.effort == "low"

from __future__ import annotations

import runpy
from pathlib import Path

from dr_llm.providers.effort import EffortSpec
from dr_llm.providers.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    ThinkingLevel,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "demo_thinking_and_effort.py"
)
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT_PATH))


def test_provider_model_sweeps_match_expected_snapshot() -> None:
    assert SCRIPT_GLOBALS["CLAUDE_MODELS"] == [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]
    assert SCRIPT_GLOBALS["OPENAI_MODELS"] == [
        "gpt-5.4-mini-2026-03-17",
        "gpt-5-mini-2025-08-07",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4o-mini-2024-07-18",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5-nano-2025-08-07",
        "gpt-4.1-nano-2025-04-14",
    ]
    assert SCRIPT_GLOBALS["CODEX_MODELS"] == [
        "gpt-5.1-codex-mini",
    ]
    assert SCRIPT_GLOBALS["KIMI_CODE_MODELS"] == [
        "kimi-for-coding",
    ]
    assert SCRIPT_GLOBALS["GOOGLE_MODELS"] == [
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
    assert SCRIPT_GLOBALS["PHASES"] == ["models", "thinking", "effort"]


def test_claude_code_uses_model_specific_thinking_and_effort_levels() -> None:
    supported_thinking_levels = SCRIPT_GLOBALS["supported_thinking_levels"]
    default_thinking_for_model = SCRIPT_GLOBALS["default_thinking_for_model"]
    default_effort_for_model = SCRIPT_GLOBALS["default_effort_for_model"]
    reasoning_for_level = SCRIPT_GLOBALS["reasoning_for_level"]

    opus_levels = supported_thinking_levels("claude-code", "claude-opus-4-6")
    assert opus_levels == [ThinkingLevel.ADAPTIVE]
    assert (
        default_thinking_for_model("claude-code", "claude-opus-4-6")
        == ThinkingLevel.ADAPTIVE
    )
    assert default_effort_for_model("claude-code", "claude-opus-4-6") == EffortSpec.LOW
    assert SCRIPT_GLOBALS["supported_effort_levels"](
        provider="claude-code",
        model="claude-opus-4-6",
    ) == (
        EffortSpec.LOW,
        EffortSpec.MEDIUM,
        EffortSpec.HIGH,
        EffortSpec.MAX,
    )

    haiku_levels = supported_thinking_levels(
        "claude-code",
        "claude-haiku-4-5-20251001",
    )
    assert haiku_levels == [ThinkingLevel.NA]
    assert (
        default_thinking_for_model("claude-code", "claude-haiku-4-5-20251001")
        == ThinkingLevel.NA
    )
    assert (
        default_effort_for_model("claude-code", "claude-haiku-4-5-20251001")
        == EffortSpec.NA
    )
    assert SCRIPT_GLOBALS["supported_effort_levels"](
        provider="claude-code",
        model="claude-haiku-4-5-20251001",
    ) == ()

    adaptive_reasoning = reasoning_for_level(
        "claude-code",
        ThinkingLevel.ADAPTIVE,
    )
    assert isinstance(adaptive_reasoning, AnthropicReasoning)
    assert adaptive_reasoning.thinking_level == ThinkingLevel.ADAPTIVE

    explicit_na_reasoning = reasoning_for_level(
        "claude-code",
        ThinkingLevel.NA,
        explicit=True,
    )
    assert isinstance(explicit_na_reasoning, AnthropicReasoning)
    assert explicit_na_reasoning.thinking_level == ThinkingLevel.NA
    assert reasoning_for_level("claude-code", ThinkingLevel.NA) is None


def test_kimi_code_uses_explicit_reasoning_and_required_effort() -> None:
    supported_thinking_levels = SCRIPT_GLOBALS["supported_thinking_levels"]
    default_thinking_for_model = SCRIPT_GLOBALS["default_thinking_for_model"]
    default_effort_for_model = SCRIPT_GLOBALS["default_effort_for_model"]
    reasoning_for_level = SCRIPT_GLOBALS["reasoning_for_level"]

    kimi_levels = supported_thinking_levels("kimi-code", "kimi-for-coding")
    assert kimi_levels == [
        ThinkingLevel.OFF,
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.BUDGET,
    ]
    assert default_thinking_for_model("kimi-code", "kimi-for-coding") == ThinkingLevel.NA
    assert default_effort_for_model("kimi-code", "kimi-for-coding") == EffortSpec.LOW
    assert SCRIPT_GLOBALS["supported_effort_levels"](
        provider="kimi-code",
        model="kimi-for-coding",
    ) == (
        EffortSpec.LOW,
        EffortSpec.MEDIUM,
        EffortSpec.HIGH,
        EffortSpec.MAX,
    )

    off_reasoning = reasoning_for_level("kimi-code", ThinkingLevel.OFF)
    adaptive_reasoning = reasoning_for_level("kimi-code", ThinkingLevel.ADAPTIVE)
    budget_reasoning = reasoning_for_level("kimi-code", ThinkingLevel.BUDGET)
    assert isinstance(off_reasoning, AnthropicReasoning)
    assert off_reasoning.thinking_level == ThinkingLevel.OFF
    assert isinstance(adaptive_reasoning, AnthropicReasoning)
    assert adaptive_reasoning.thinking_level == ThinkingLevel.ADAPTIVE
    assert isinstance(budget_reasoning, AnthropicReasoning)
    assert budget_reasoning.thinking_level == ThinkingLevel.BUDGET
    assert budget_reasoning.budget_tokens == SCRIPT_GLOBALS["KIMI_CODE_FIXED_BUDGET"]


def test_openai_and_codex_use_explicit_thinking_levels_only() -> None:
    supported_thinking_levels = SCRIPT_GLOBALS["supported_thinking_levels"]
    default_thinking_for_model = SCRIPT_GLOBALS["default_thinking_for_model"]
    reasoning_for_level = SCRIPT_GLOBALS["reasoning_for_level"]

    openai_levels = supported_thinking_levels("openai", "gpt-5-mini-2025-08-07")
    assert openai_levels == [
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    ]
    assert (
        default_thinking_for_model("openai", "gpt-5-mini-2025-08-07")
        == ThinkingLevel.MINIMAL
    )

    codex_levels = supported_thinking_levels("codex", "gpt-5.1-codex-mini")
    assert codex_levels == [
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    ]
    assert (
        default_thinking_for_model("codex", "gpt-5.1-codex-mini")
        == ThinkingLevel.LOW
    )

    openai_reasoning = reasoning_for_level("openai", ThinkingLevel.HIGH)
    codex_reasoning = reasoning_for_level("codex", ThinkingLevel.MEDIUM)
    assert isinstance(openai_reasoning, OpenAIReasoning)
    assert openai_reasoning.thinking_level == ThinkingLevel.HIGH
    assert isinstance(codex_reasoning, CodexReasoning)
    assert codex_reasoning.thinking_level == ThinkingLevel.MEDIUM


def test_google_uses_budget_or_level_controls_by_model_family() -> None:
    supported_thinking_levels = SCRIPT_GLOBALS["supported_thinking_levels"]
    default_thinking_for_model = SCRIPT_GLOBALS["default_thinking_for_model"]
    reasoning_for_level = SCRIPT_GLOBALS["reasoning_for_level"]

    budget_levels = supported_thinking_levels("google", "gemini-2.5-flash")
    assert budget_levels == [
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.OFF,
        ThinkingLevel.BUDGET,
    ]
    assert default_thinking_for_model("google", "gemini-2.5-flash") == ThinkingLevel.OFF

    level_levels = supported_thinking_levels("google", "gemini-3-flash-preview")
    assert level_levels == [
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    ]
    assert (
        default_thinking_for_model("google", "gemini-3-flash-preview")
        == ThinkingLevel.MINIMAL
    )

    unsupported_levels = supported_thinking_levels("google", "gemma-3-1b-it")
    assert unsupported_levels == [ThinkingLevel.NA]
    assert default_thinking_for_model("google", "gemma-3-1b-it") == ThinkingLevel.NA

    gemma_4_levels = supported_thinking_levels("google", "gemma-4-31b-it")
    assert gemma_4_levels == [
        ThinkingLevel.MINIMAL,
        ThinkingLevel.HIGH,
    ]
    assert default_thinking_for_model("google", "gemma-4-31b-it") == ThinkingLevel.MINIMAL

    budget_reasoning = reasoning_for_level("google", ThinkingLevel.BUDGET)
    level_reasoning = reasoning_for_level("google", ThinkingLevel.LOW)
    assert isinstance(budget_reasoning, GoogleReasoning)
    assert budget_reasoning.thinking_level == ThinkingLevel.BUDGET
    assert budget_reasoning.budget_tokens == SCRIPT_GLOBALS["GOOGLE_FIXED_BUDGET"]
    assert isinstance(level_reasoning, GoogleReasoning)
    assert level_reasoning.thinking_level == ThinkingLevel.LOW

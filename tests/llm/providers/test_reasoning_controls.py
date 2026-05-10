from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from dr_llm.llm import (
    AnthropicReasoning,
    CodexReasoning,
    EffortSpec,
    GlmReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningControls,
    ThinkingLevel,
    default_effort,
    default_reasoning,
    default_thinking_level,
    reasoning_controls_for_model,
    reasoning_for_thinking_level,
    supported_thinking_levels,
)


def test_openai_controls_prefer_minimal_or_off_defaults() -> None:
    assert supported_thinking_levels(
        provider="openai", model="gpt-5-mini"
    ) == (
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    )
    assert default_reasoning(
        provider="openai", model="gpt-5-mini"
    ) == OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL)

    assert (
        default_thinking_level(provider="openai", model="gpt-5.4-mini")
        == ThinkingLevel.OFF
    )
    assert default_reasoning(
        provider="openai", model="gpt-5.4-mini"
    ) == OpenAIReasoning(thinking_level=ThinkingLevel.OFF)


def test_google_controls_cover_budget_and_level_modes() -> None:
    assert supported_thinking_levels(
        provider="google", model="gemini-2.5-flash"
    ) == (
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.OFF,
        ThinkingLevel.BUDGET,
    )
    assert default_reasoning(
        provider="google", model="gemini-2.5-flash"
    ) == GoogleReasoning(thinking_level=ThinkingLevel.OFF)

    assert supported_thinking_levels(
        provider="google", model="gemma-4-26b-a4b-it"
    ) == (ThinkingLevel.MINIMAL, ThinkingLevel.HIGH)
    assert default_reasoning(
        provider="google", model="gemma-4-26b-a4b-it"
    ) == GoogleReasoning(thinking_level=ThinkingLevel.MINIMAL)


def test_reasoning_for_budget_level_requires_explicit_tokens() -> None:
    with pytest.raises(ValueError, match="budget thinking requires"):
        reasoning_for_thinking_level(
            provider="google",
            model="gemini-2.5-flash",
            thinking_level=ThinkingLevel.BUDGET,
        )

    assert reasoning_for_thinking_level(
        provider="google",
        model="gemini-2.5-flash",
        thinking_level=ThinkingLevel.BUDGET,
        budget_tokens=128,
    ) == GoogleReasoning(
        thinking_level=ThinkingLevel.BUDGET,
        budget_tokens=128,
    )


def test_codex_and_glm_controls_use_provider_native_specs() -> None:
    assert supported_thinking_levels(
        provider="codex", model="gpt-5.4-mini"
    ) == (
        ThinkingLevel.OFF,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
        ThinkingLevel.XHIGH,
    )
    assert default_reasoning(
        provider="codex", model="gpt-5.4-mini"
    ) == CodexReasoning(thinking_level=ThinkingLevel.OFF)

    assert supported_thinking_levels(provider="glm", model="glm-4.5") == (
        ThinkingLevel.OFF,
        ThinkingLevel.ADAPTIVE,
    )
    assert default_reasoning(provider="glm", model="glm-4.5") == GlmReasoning(
        thinking_level=ThinkingLevel.OFF
    )


def test_anthropic_style_controls_cover_effort_and_required_na() -> None:
    assert supported_thinking_levels(
        provider="anthropic", model="claude-sonnet-4-20250514"
    ) == (ThinkingLevel.OFF, ThinkingLevel.BUDGET)
    assert default_reasoning(
        provider="anthropic", model="claude-sonnet-4-20250514"
    ) == AnthropicReasoning(thinking_level=ThinkingLevel.OFF)

    assert (
        default_effort(provider="claude-code", model="claude-sonnet-4-6")
        == EffortSpec.LOW
    )
    assert default_reasoning(
        provider="claude-code", model="claude-sonnet-4-6"
    ) == AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)

    assert (
        default_effort(provider="minimax", model="MiniMax-M2")
        == EffortSpec.LOW
    )
    assert default_reasoning(
        provider="minimax", model="MiniMax-M2"
    ) == AnthropicReasoning(thinking_level=ThinkingLevel.NA)

    assert (
        default_effort(provider="kimi-code", model="kimi-for-coding")
        == EffortSpec.LOW
    )
    assert default_reasoning(
        provider="kimi-code", model="kimi-for-coding"
    ) == AnthropicReasoning(thinking_level=ThinkingLevel.OFF)


def test_openrouter_controls_follow_curated_policy() -> None:
    assert (
        default_reasoning(
            provider="openrouter", model="deepseek/deepseek-chat"
        )
        is None
    )
    assert default_reasoning(
        provider="openrouter", model="deepseek/deepseek-chat-v3.1"
    ) == OpenRouterReasoning(enabled=False)
    assert default_reasoning(
        provider="openrouter", model="deepseek/deepseek-r1"
    ) == OpenRouterReasoning(enabled=True)
    assert default_reasoning(
        provider="openrouter", model="openai/gpt-oss-20b"
    ) == OpenRouterReasoning(effort="low")


def test_reasoning_controls_model_collects_all_defaults() -> None:
    controls = reasoning_controls_for_model(
        provider="google", model="gemini-2.5-flash"
    )

    assert isinstance(controls, ReasoningControls)
    assert controls.supported_thinking_levels == (
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.OFF,
        ThinkingLevel.BUDGET,
    )
    assert controls.default_thinking_level == ThinkingLevel.OFF
    assert controls.supported_effort_levels == ()
    assert controls.default_effort == EffortSpec.NA
    assert controls.default_reasoning == GoogleReasoning(
        thinking_level=ThinkingLevel.OFF
    )


def test_demo_pool_provider_args_are_built_from_defaults() -> None:
    module = _load_demo_pool_providers()

    assert module.provider_extra_args("google", "gemini-2.5-flash") == [
        "--reasoning-json",
        '{"kind":"google","thinking_level":"off"}',
    ]
    assert module.provider_extra_args("minimax", "MiniMax-M2") == [
        "--reasoning-json",
        '{"kind":"anthropic","thinking_level":"na"}',
        "--effort",
        "low",
    ]


def _load_demo_pool_providers() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3] / "scripts/demo-pool-providers.py"
    )
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("demo_pool_providers", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

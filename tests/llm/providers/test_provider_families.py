from __future__ import annotations

from dr_llm.llm.names import (
    ControlMode,
    EffortSpec,
    OpenRouterEffortLevel,
    ThinkingLevel,
)
from dr_llm.llm.providers.impls.anthropic.families import AnthropicFamilies
from dr_llm.llm.providers.impls.claude_code.families import ClaudeCodeFamilies
from dr_llm.llm.providers.impls.codex.families import CodexFamilies
from dr_llm.llm.providers.impls.google.families import GoogleFamilies
from dr_llm.llm.providers.impls.kimi_code.families import KimiCodeFamilies
from dr_llm.llm.providers.impls.openai.families import OpenAIFamilies
from dr_llm.llm.providers.impls.openrouter.families import (
    OpenRouterControlRequestStyle,
    OpenRouterFamilies,
    OpenRouterModelPolicy,
)


def test_openai_families_match_snapshots_and_prefixed_models() -> None:
    families = OpenAIFamilies()

    assert families.supports_off_thinking("openai/gpt-5.2-20260201")
    assert families.supports_minimal_thinking("gpt-5-mini")
    assert families.supports_sampling_with_reasoning_off("gpt-5.4")
    assert not families.supports_sampling_with_reasoning_off("gpt-5.3")
    assert families.supports_gpt_oss_thinking("gpt-oss-20b")
    assert families.supported_thinking_levels("gpt-oss-20b") == (
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    )
    assert families.default_thinking_level("gpt-oss-20b") == ThinkingLevel.LOW


def test_codex_families_separate_public_and_codex_model_behavior() -> None:
    families = CodexFamilies()

    assert families.supports_off_thinking("gpt-5.2")
    assert families.supports_minimal_thinking("gpt-5")
    assert families.supports_configurable_thinking("gpt-5.3-codex-spark")
    assert not families.supports_off_thinking("gpt-5.1-codex-mini")


def test_anthropic_families_expose_budget_and_effort_capabilities() -> None:
    families = AnthropicFamilies()

    assert (
        families.control_mode("claude-opus-4-6")
        == ControlMode.ANTHROPIC_EFFORT
    )
    assert (
        families.control_mode("claude-opus-4-5-20251101")
        == ControlMode.ANTHROPIC_EFFORT_AND_BUDGET
    )
    assert families.budget_min_for_model("claude-sonnet-4-5") == 1024
    assert families.budget_max_for_model("claude-sonnet-4-5") == 128000
    assert EffortSpec.MAX in families.supported_effort_levels(
        "claude-opus-4-6"
    )


def test_google_families_carry_budget_and_level_metadata() -> None:
    families = GoogleFamilies()

    assert (
        families.control_mode("gemini-2.5-flash-lite")
        == ControlMode.GOOGLE_BUDGET
    )
    assert families.min_budget_tokens("gemini-2.5-flash-lite") == 512
    assert families.max_budget_tokens("gemini-2.5-pro") == 32768
    assert families.supported_thinking_levels("gemini-3") == (
        ThinkingLevel.MINIMAL,
        ThinkingLevel.LOW,
        ThinkingLevel.MEDIUM,
        ThinkingLevel.HIGH,
    )
    assert families.supported_thinking_levels("gemma-4") == (
        ThinkingLevel.MINIMAL,
        ThinkingLevel.HIGH,
    )


def test_kimi_code_families_expose_fixed_budget_and_effort_defaults() -> None:
    families = KimiCodeFamilies()

    assert families.control_mode("kimi-for-coding") == (
        ControlMode.KIMI_CODE_EFFORT_AND_BUDGET
    )
    assert families.budget_min_for_model("kimi-for-coding") == 1024
    assert families.budget_max_for_model("kimi-for-coding") == 128000
    assert families.default_effort("kimi-for-coding") == EffortSpec.LOW


def test_claude_code_families_compose_anthropic_capabilities() -> None:
    families = ClaudeCodeFamilies()

    assert families.is_supported_model("claude-sonnet-4-6")
    assert families.supports_adaptive_thinking("claude-sonnet-4-6")
    assert families.default_thinking_level("claude-sonnet-4-6") == (
        ThinkingLevel.ADAPTIVE
    )
    assert EffortSpec.MAX in families.supported_effort_levels(
        "claude-opus-4-6"
    )


def test_openrouter_families_use_exact_policy_lookup() -> None:
    families = OpenRouterFamilies(
        policies={
            "provider/model": OpenRouterModelPolicy(
                model="provider/model",
                request_style=OpenRouterControlRequestStyle.EFFORT,
                supports_disable=False,
                allowed_efforts=(
                    OpenRouterEffortLevel.LOW,
                    OpenRouterEffortLevel.MEDIUM,
                    OpenRouterEffortLevel.HIGH,
                ),
            )
        }
    )

    assert families.allowed_models() == ("provider/model",)
    assert (
        families.control_mode("provider/model")
        == ControlMode.OPENROUTER_EFFORT
    )
    assert (
        families.control_mode("provider/model-snapshot")
        == ControlMode.UNSUPPORTED
    )

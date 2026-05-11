from __future__ import annotations

from collections.abc import Generator

import pytest

from dr_llm.llm import (
    AnthropicReasoning,
    EffortSpec,
    GoogleReasoning,
    OpenAIReasoning,
    ProviderName,
    ThinkingLevel,
    build_default_registry,
)
from dr_llm.llm.providers.core.registry import ProviderRegistry


@pytest.fixture
def registry() -> Generator[ProviderRegistry]:
    reg = build_default_registry()
    try:
        yield reg
    finally:
        reg.close()


def test_request_defaults_expose_openai_sampling_and_reasoning_policy(
    registry: ProviderRegistry,
) -> None:
    defaults = registry.get(ProviderName.OPENAI).request_defaults("gpt-5-mini")

    assert defaults.provider == ProviderName.OPENAI
    assert defaults.mode == "api"
    assert defaults.effort == EffortSpec.NA
    assert defaults.reasoning == OpenAIReasoning(
        thinking_level=ThinkingLevel.MINIMAL
    )
    assert defaults.supports_temperature
    assert defaults.temperature is None
    assert defaults.supports_top_p
    assert defaults.top_p is None


def test_request_defaults_expose_google_sampling_defaults(
    registry: ProviderRegistry,
) -> None:
    defaults = registry.get(ProviderName.GOOGLE).request_defaults(
        "gemini-2.5-flash"
    )

    assert defaults.reasoning == GoogleReasoning(
        thinking_level=ThinkingLevel.OFF
    )
    assert defaults.supports_temperature
    assert defaults.temperature == 1.0
    assert defaults.supports_top_p
    assert defaults.top_p == 0.95


def test_request_defaults_expose_required_max_tokens(
    registry: ProviderRegistry,
) -> None:
    anthropic = registry.get(ProviderName.ANTHROPIC).request_defaults(
        "claude-sonnet-4-20250514"
    )
    kimi = registry.get(ProviderName.KIMI_CODE).request_defaults(
        "kimi-for-coding"
    )

    assert anthropic.max_tokens_required
    assert anthropic.max_tokens == 4096
    assert anthropic.supports_temperature
    assert anthropic.temperature == 1.0
    assert anthropic.supports_top_p
    assert anthropic.top_p == 0.95

    assert kimi.max_tokens_required
    assert kimi.max_tokens == 16384
    assert kimi.effort == EffortSpec.LOW
    assert kimi.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.OFF
    )
    assert not kimi.supports_temperature
    assert not kimi.supports_top_p


def test_request_defaults_expose_headless_and_minimax_defaults(
    registry: ProviderRegistry,
) -> None:
    codex = registry.get(ProviderName.CODEX).request_defaults("gpt-5.4-mini")
    claude = registry.get(ProviderName.CLAUDE_CODE).request_defaults(
        "claude-sonnet-4-6"
    )
    minimax = registry.get(ProviderName.MINIMAX).request_defaults("MiniMax-M2")

    assert codex.mode == "headless"
    assert not codex.supports_temperature
    assert claude.mode == "headless"
    assert not claude.supports_top_p

    assert minimax.effort == EffortSpec.LOW
    assert minimax.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.NA
    )
    assert minimax.supports_temperature
    assert minimax.temperature == 1.0

from __future__ import annotations

from collections.abc import Generator

import pytest
from pydantic import ValidationError

from dr_llm.llm import (
    AnthropicReasoning,
    CallMode,
    EffortSpec,
    GoogleReasoning,
    OpenAIReasoning,
    ProviderName,
    SamplingControls,
    ThinkingLevel,
    build_default_registry,
)
from dr_llm.llm.providers.core.request_defaults import ProviderRequestDefaults
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
    assert defaults.sampling_supported
    assert defaults.sampling is None


def test_request_defaults_expose_google_sampling_defaults(
    registry: ProviderRegistry,
) -> None:
    defaults = registry.get(ProviderName.GOOGLE).request_defaults(
        "gemini-2.5-flash"
    )

    assert defaults.reasoning == GoogleReasoning(
        thinking_level=ThinkingLevel.OFF
    )
    assert defaults.sampling_supported
    assert defaults.sampling == SamplingControls(temperature=1.0, top_p=0.95)


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
    assert anthropic.sampling_supported
    assert anthropic.sampling == SamplingControls(temperature=1.0, top_p=0.95)

    assert kimi.max_tokens_required
    assert kimi.max_tokens == 16384
    assert kimi.effort == EffortSpec.LOW
    assert kimi.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.OFF
    )
    assert not kimi.sampling_supported
    assert kimi.sampling is None


def test_request_defaults_expose_headless_and_minimax_defaults(
    registry: ProviderRegistry,
) -> None:
    codex = registry.get(ProviderName.CODEX).request_defaults("gpt-5.4-mini")
    claude = registry.get(ProviderName.CLAUDE_CODE).request_defaults(
        "claude-sonnet-4-6"
    )
    minimax = registry.get(ProviderName.MINIMAX).request_defaults("MiniMax-M2")

    assert codex.mode == CallMode.headless
    assert not codex.sampling_supported
    assert claude.mode == CallMode.headless
    assert not claude.sampling_supported

    assert minimax.effort == EffortSpec.LOW
    assert minimax.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.NA
    )
    assert minimax.sampling_supported
    assert minimax.sampling == SamplingControls(temperature=1.0, top_p=0.95)


def test_request_defaults_reject_inconsistent_max_token_state() -> None:
    with pytest.raises(ValidationError, match="max_tokens_required"):
        ProviderRequestDefaults(
            provider="test",
            model="test-model",
            mode=CallMode.api,
            max_tokens_required=True,
        )


def test_request_defaults_reject_unsupported_sampling_defaults() -> None:
    with pytest.raises(ValidationError, match="sampling_supported"):
        ProviderRequestDefaults(
            provider="test",
            model="test-model",
            mode=CallMode.api,
            sampling=SamplingControls(temperature=0.7),
        )


def test_build_request_applies_provider_defaults(
    registry: ProviderRegistry,
) -> None:
    request = registry.get(ProviderName.KIMI_CODE).build_request(
        model="kimi-for-coding",
        messages=[],
    )

    assert request.provider == ProviderName.KIMI_CODE
    assert request.mode == CallMode.api
    assert request.max_tokens == 16384
    assert request.effort == EffortSpec.LOW
    assert request.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.OFF
    )


def test_build_request_rejects_unsupported_sampling_control(
    registry: ProviderRegistry,
) -> None:
    with pytest.raises(ValueError, match="sampling is not supported"):
        registry.get(ProviderName.KIMI_CODE).build_request(
            model="kimi-for-coding",
            messages=[],
            sampling=SamplingControls(temperature=0.2),
        )


def test_build_request_treats_empty_sampling_control_as_unset(
    registry: ProviderRegistry,
) -> None:
    request = registry.get(ProviderName.KIMI_CODE).build_request(
        model="kimi-for-coding",
        messages=[],
        sampling=SamplingControls(),
    )

    assert request.sampling is None

from __future__ import annotations

from typing import Any, Protocol, cast

import pytest
from pydantic import ValidationError

from dr_llm.llm import (
    AnthropicBudgetConfig,
    AnthropicEffortAndBudgetConfig,
    AnthropicEffortConfig,
    AnthropicLegacyConfig,
    ClaudeCodeAdaptiveConfig,
    ClaudeCodeEffortConfig,
    ClaudeCodeLegacyConfig,
    CodexGpt5CodexConfig,
    CodexGpt5Config,
    CodexGpt51Config,
    CodexGpt52Config,
    CodexGpt54Config,
    CodexLegacyConfig,
    EffortSpec,
    GlmLegacyConfig,
    GlmThinkingConfig,
    GoogleBudgetConfig,
    GoogleLegacyConfig,
    GoogleLevelConfig,
    KimiCodeConfig,
    LlmConfig,
    LlmRequest,
    Message,
    MiniMaxConfig,
    OpenAIGpt5Config,
    OpenAIGpt51Config,
    OpenAIGpt52Config,
    OpenAIGpt53Config,
    OpenAIGpt54Config,
    OpenAIGptOssConfig,
    OpenAILegacyConfig,
    OpenRouterEffortConfig,
    OpenRouterEffortLevel,
    OpenRouterNoControlConfig,
    OpenRouterToggleConfig,
    ProviderName,
    SamplingControls,
    ThinkingLevel,
    build_default_registry,
    build_request_from_config,
    parse_llm_config,
    parse_llm_request,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GlmReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
)
from dr_llm.llm.response import CallMode


class SupportsToLlmConfig(Protocol):
    def to_llm_config(self) -> LlmConfig: ...


def test_normalized_config_construction() -> None:
    sampling = SamplingControls(temperature=0.7, top_p=0.9)
    config = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        max_tokens=1024,
        sampling=sampling,
    )

    assert config.provider == ProviderName.OPENAI
    assert config.model == "gpt-4.1-mini"
    assert config.max_tokens == 1024
    assert config.sampling == sampling
    assert config.effort == EffortSpec.NA
    assert config.reasoning is None


def test_old_flat_sampling_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        parse_llm_config(
            {
                "provider": ProviderName.OPENAI,
                "model": "gpt-4.1-mini",
                "mode": CallMode.api,
                "temperature": 0.7,
            }
        )

    with pytest.raises(ValidationError, match="top_p"):
        parse_llm_request(
            {
                "provider": ProviderName.OPENAI,
                "model": "gpt-4.1-mini",
                "mode": CallMode.api,
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 0.9,
            }
        )


def test_orchestrator_build_request_applies_normalized_config() -> None:
    config = OpenAILegacyConfig(
        model="gpt-4.1-mini",
        max_tokens=100,
        sampling=SamplingControls(temperature=0.5),
    ).to_llm_config()
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]

    registry = build_default_registry()
    try:
        request = build_request_from_config(
            registry.get(config.provider),
            config,
            messages,
        )
    finally:
        registry.close()

    assert isinstance(request, LlmRequest)
    assert request.provider == ProviderName.OPENAI
    assert request.model == "gpt-4.1-mini"
    assert request.messages == messages
    assert request.sampling == SamplingControls(temperature=0.5)
    assert request.max_tokens == 100
    assert request.metadata == {}


@pytest.mark.parametrize(
    ("config", "expected_reasoning"),
    [
        (
            OpenAILegacyConfig(model="gpt-4.1-mini"),
            None,
        ),
        (
            OpenAIGpt5Config(
                model="gpt-5-mini",
                thinking_level=ThinkingLevel.MINIMAL,
            ),
            OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL),
        ),
        (
            OpenAIGpt51Config(
                model="gpt-5.1-mini",
                thinking_level=ThinkingLevel.OFF,
            ),
            OpenAIReasoning(thinking_level=ThinkingLevel.OFF),
        ),
        (
            OpenAIGpt52Config(
                model="gpt-5.2-mini",
                thinking_level=ThinkingLevel.OFF,
                sampling=SamplingControls(temperature=0.7, top_p=0.95),
            ),
            OpenAIReasoning(thinking_level=ThinkingLevel.OFF),
        ),
        (
            OpenAIGpt53Config(
                model="gpt-5.3-mini",
                thinking_level=ThinkingLevel.HIGH,
            ),
            OpenAIReasoning(thinking_level=ThinkingLevel.HIGH),
        ),
        (
            OpenAIGpt54Config(
                model="gpt-5.4-mini",
                thinking_level=ThinkingLevel.OFF,
                sampling=SamplingControls(temperature=0.7),
            ),
            OpenAIReasoning(thinking_level=ThinkingLevel.OFF),
        ),
        (
            OpenAIGptOssConfig(
                model="gpt-oss-20b",
                thinking_level=ThinkingLevel.MEDIUM,
            ),
            OpenAIReasoning(thinking_level=ThinkingLevel.MEDIUM),
        ),
    ],
)
def test_openai_authoring_configs_to_llm_config(
    config: SupportsToLlmConfig, expected_reasoning: object
) -> None:
    llm_config = config.to_llm_config()

    assert llm_config.provider == ProviderName.OPENAI
    assert llm_config.mode == CallMode.api
    assert llm_config.reasoning == expected_reasoning


@pytest.mark.parametrize(
    "config",
    [
        AnthropicLegacyConfig(model="claude-3-5-sonnet-latest"),
        AnthropicBudgetConfig(
            model="claude-sonnet-4-20250514",
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=1024,
        ),
        AnthropicEffortConfig(
            model="claude-sonnet-4-6",
            effort=EffortSpec.MEDIUM,
            thinking_level=ThinkingLevel.ADAPTIVE,
        ),
        AnthropicEffortAndBudgetConfig(
            model="claude-opus-4-5-20251101",
            effort=EffortSpec.HIGH,
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=2048,
        ),
    ],
)
def test_anthropic_authoring_configs_to_llm_config(
    config: SupportsToLlmConfig,
) -> None:
    llm_config = config.to_llm_config()

    assert llm_config.provider == ProviderName.ANTHROPIC
    assert llm_config.mode == CallMode.api
    assert llm_config.max_tokens == 4096


@pytest.mark.parametrize(
    ("config", "expected_reasoning"),
    [
        (GoogleLegacyConfig(model="gemini-2.0-flash"), None),
        (
            GoogleBudgetConfig(
                model="gemini-2.5-flash",
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=1,
            ),
            GoogleReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=1,
            ),
        ),
        (
            GoogleLevelConfig(
                model="gemini-3-flash-preview",
                thinking_level=ThinkingLevel.LOW,
                include_thoughts=True,
            ),
            GoogleReasoning(
                thinking_level=ThinkingLevel.LOW,
                include_thoughts=True,
            ),
        ),
    ],
)
def test_google_authoring_configs_to_llm_config(
    config: SupportsToLlmConfig, expected_reasoning: object
) -> None:
    llm_config = config.to_llm_config()

    assert llm_config.provider == ProviderName.GOOGLE
    assert llm_config.mode == CallMode.api
    assert llm_config.reasoning == expected_reasoning


@pytest.mark.parametrize(
    ("config", "expected_reasoning"),
    [
        (GlmLegacyConfig(model="glm-4-air"), None),
        (
            GlmThinkingConfig(
                model="glm-4.5",
                thinking_level=ThinkingLevel.OFF,
            ),
            GlmReasoning(thinking_level=ThinkingLevel.OFF),
        ),
    ],
)
def test_glm_authoring_configs_to_llm_config(
    config: SupportsToLlmConfig, expected_reasoning: object
) -> None:
    llm_config = config.to_llm_config()

    assert llm_config.provider == ProviderName.GLM
    assert llm_config.reasoning == expected_reasoning


@pytest.mark.parametrize(
    ("config", "expected_reasoning"),
    [
        (
            OpenRouterNoControlConfig(model="deepseek/deepseek-chat"),
            None,
        ),
        (
            OpenRouterToggleConfig(
                model="deepseek/deepseek-chat-v3.1",
                reasoning_enabled=False,
            ),
            OpenRouterReasoning(enabled=False),
        ),
        (
            OpenRouterEffortConfig(
                model="openai/gpt-oss-120b",
                effort=OpenRouterEffortLevel.MEDIUM,
            ),
            OpenRouterReasoning(effort=OpenRouterEffortLevel.MEDIUM),
        ),
    ],
)
def test_openrouter_authoring_configs_to_llm_config(
    config: SupportsToLlmConfig, expected_reasoning: object
) -> None:
    llm_config = config.to_llm_config()

    assert llm_config.provider == ProviderName.OPENROUTER
    assert llm_config.reasoning == expected_reasoning


@pytest.mark.parametrize(
    ("config", "expected_reasoning"),
    [
        (CodexLegacyConfig(model="gpt-4.1"), None),
        (
            CodexGpt5Config(
                model="gpt-5",
                thinking_level=ThinkingLevel.MINIMAL,
            ),
            CodexReasoning(thinking_level=ThinkingLevel.MINIMAL),
        ),
        (
            CodexGpt51Config(
                model="gpt-5.1",
                thinking_level=ThinkingLevel.OFF,
            ),
            CodexReasoning(thinking_level=ThinkingLevel.OFF),
        ),
        (
            CodexGpt52Config(
                model="gpt-5.2",
                thinking_level=ThinkingLevel.HIGH,
            ),
            CodexReasoning(thinking_level=ThinkingLevel.HIGH),
        ),
        (
            CodexGpt54Config(
                model="gpt-5.4-mini",
                thinking_level=ThinkingLevel.OFF,
            ),
            CodexReasoning(thinking_level=ThinkingLevel.OFF),
        ),
        (
            CodexGpt5CodexConfig(
                model="gpt-5.3-codex",
                thinking_level=ThinkingLevel.XHIGH,
            ),
            CodexReasoning(thinking_level=ThinkingLevel.XHIGH),
        ),
    ],
)
def test_codex_authoring_configs_to_llm_config(
    config: SupportsToLlmConfig, expected_reasoning: object
) -> None:
    llm_config = config.to_llm_config()

    assert llm_config.provider == ProviderName.CODEX
    assert llm_config.mode == CallMode.headless
    assert llm_config.reasoning == expected_reasoning


@pytest.mark.parametrize(
    ("config", "expected_provider"),
    [
        (
            ClaudeCodeLegacyConfig(model="claude-3-5-sonnet-latest"),
            ProviderName.CLAUDE_CODE,
        ),
        (
            ClaudeCodeAdaptiveConfig(model="claude-sonnet-4-6"),
            ProviderName.CLAUDE_CODE,
        ),
        (
            ClaudeCodeEffortConfig(
                model="claude-opus-4-5-20251101",
                effort=EffortSpec.HIGH,
            ),
            ProviderName.CLAUDE_CODE,
        ),
        (
            KimiCodeConfig(
                model="kimi-for-coding",
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=1024,
            ),
            ProviderName.KIMI_CODE,
        ),
        (
            MiniMaxConfig(model="MiniMax-M2.7", effort=EffortSpec.MEDIUM),
            ProviderName.MINIMAX,
        ),
    ],
)
def test_headless_and_narrow_authoring_configs_to_llm_config(
    config: SupportsToLlmConfig, expected_provider: ProviderName
) -> None:
    llm_config = config.to_llm_config()

    assert llm_config.provider == expected_provider


def test_anthropic_authoring_config_uses_family_capabilities_for_snapshots() -> (
    None
):
    config = AnthropicEffortAndBudgetConfig(
        model="claude-opus-4-5-20261201",
        effort=EffortSpec.HIGH,
        thinking_level=ThinkingLevel.BUDGET,
        budget_tokens=2048,
    ).to_llm_config()

    assert config.effort == EffortSpec.HIGH
    assert config.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.BUDGET,
        budget_tokens=2048,
    )


def test_claude_code_authoring_configs_use_anthropic_family_capabilities() -> (
    None
):
    adaptive_config = ClaudeCodeAdaptiveConfig(
        model="claude-sonnet-4-6-20261201"
    ).to_llm_config()
    effort_config = ClaudeCodeEffortConfig(
        model="claude-opus-4-5-20261201",
        effort=EffortSpec.HIGH,
    ).to_llm_config()

    assert adaptive_config.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.ADAPTIVE
    )
    assert effort_config.effort == EffortSpec.HIGH


@pytest.mark.parametrize(
    ("config_factory", "match_pattern"),
    [
        (
            lambda: OpenAIGpt5Config(model="gpt-5.2-mini"),
            "OpenAIGpt5Config only supports",
        ),
        (
            lambda: OpenAIGpt52Config(
                model="gpt-5.2-mini",
                thinking_level=ThinkingLevel.HIGH,
                sampling=SamplingControls(temperature=0.7),
            ),
            "custom sampling requires thinking_level='off'",
        ),
        (
            lambda: OpenAIGptOssConfig(
                model="gpt-oss-20b",
                thinking_level=cast(Any, ThinkingLevel.OFF),
            ),
            "Input should be",
        ),
        (
            lambda: GoogleBudgetConfig(
                model="gemini-2.5-flash-lite",
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=1,
            ),
            "budget_tokens must be between",
        ),
        (
            lambda: OpenRouterToggleConfig(
                model="deepseek/deepseek-r1",
                reasoning_enabled=False,
            ),
            "reasoning cannot be disabled",
        ),
        (
            lambda: KimiCodeConfig(model="kimi-k2"),
            "KimiCodeConfig only supports",
        ),
        (
            lambda: MiniMaxConfig(model="not-minimax"),
            "MiniMaxConfig only supports",
        ),
    ],
)
def test_authoring_configs_reject_invalid_family_or_controls(
    config_factory, match_pattern: str
) -> None:
    with pytest.raises(ValueError, match=match_pattern):
        config_factory()


def test_provider_reasoning_shape_can_parse_before_validation() -> None:
    config = LlmConfig(
        provider=ProviderName.GOOGLE,
        model="gemini-2.5-flash",
        mode=CallMode.api,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
    )

    assert config.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.OFF
    )


def test_build_config_rejects_provider_specific_reasoning_mismatch() -> None:
    registry = build_default_registry()
    try:
        with pytest.raises(ValueError, match="openai reasoning"):
            registry.get(ProviderName.OPENAI).build_config(
                model="gpt-5-mini",
                reasoning=GoogleReasoning(thinking_level=ThinkingLevel.OFF),
            )
    finally:
        registry.close()


def test_openai_gpt_oss_rejects_no_thinking_level() -> None:
    registry = build_default_registry()
    try:
        with pytest.raises(ValueError, match="thinking_level='off'"):
            registry.get(ProviderName.OPENAI).build_config(
                model="gpt-oss-20b",
                thinking_level=ThinkingLevel.OFF,
            )
    finally:
        registry.close()


def test_openai_config_does_not_store_empty_sampling_default() -> None:
    config = OpenAIGpt5Config(
        model="gpt-5-mini",
        thinking_level=ThinkingLevel.MINIMAL,
    ).to_llm_config()

    assert config.sampling is None
    assert "sampling" not in config.model_dump(mode="json", exclude_none=True)


def test_build_request_from_config_rejects_provider_mismatch() -> None:
    config = OpenAILegacyConfig(model="gpt-4.1-mini").to_llm_config()
    messages = [Message(role="user", content="Hello")]

    registry = build_default_registry()
    try:
        with pytest.raises(ValueError, match="does not match"):
            build_request_from_config(
                registry.get(ProviderName.GOOGLE),
                config,
                messages,
            )
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("provider", "config", "match_pattern"),
    [
        (
            ProviderName.CODEX,
            CodexGpt54Config(
                model="gpt-5.4-mini",
                thinking_level=ThinkingLevel.OFF,
            )
            .to_llm_config()
            .model_copy(update={"max_tokens": 100}),
            "max_tokens is not supported",
        ),
        (
            ProviderName.CLAUDE_CODE,
            ClaudeCodeAdaptiveConfig(model="claude-sonnet-4-6")
            .to_llm_config()
            .model_copy(update={"max_tokens": 100}),
            "max_tokens is not supported",
        ),
        (
            ProviderName.CODEX,
            CodexGpt54Config(
                model="gpt-5.4-mini",
                thinking_level=ThinkingLevel.OFF,
            )
            .to_llm_config()
            .model_copy(
                update={"sampling": SamplingControls(temperature=0.2)}
            ),
            "sampling is not supported",
        ),
        (
            ProviderName.CLAUDE_CODE,
            ClaudeCodeAdaptiveConfig(model="claude-sonnet-4-6")
            .to_llm_config()
            .model_copy(update={"sampling": SamplingControls(top_p=0.9)}),
            "sampling is not supported",
        ),
        (
            ProviderName.KIMI_CODE,
            KimiCodeConfig(model="kimi-for-coding")
            .to_llm_config()
            .model_copy(
                update={"sampling": SamplingControls(temperature=0.2)}
            ),
            "sampling is not supported",
        ),
    ],
)
def test_build_request_from_config_rejects_unsupported_stored_controls(
    provider: ProviderName, config: LlmConfig, match_pattern: str
) -> None:
    messages = [Message(role="user", content="Hello")]

    registry = build_default_registry()
    try:
        with pytest.raises(ValueError, match=match_pattern):
            build_request_from_config(
                registry.get(provider),
                config,
                messages,
            )
    finally:
        registry.close()


def test_build_request_from_config_accepts_supported_stored_sampling() -> None:
    config = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        sampling=SamplingControls(temperature=0.4, top_p=0.8),
    )
    messages = [Message(role="user", content="Hello")]

    registry = build_default_registry()
    try:
        request = build_request_from_config(
            registry.get(ProviderName.OPENAI),
            config,
            messages,
        )
    finally:
        registry.close()

    assert request.sampling == SamplingControls(temperature=0.4, top_p=0.8)

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_llm.llm import (
    AnthropicReasoning,
    EffortSpec,
    GlmReasoning,
    GoogleReasoning,
    HeadlessLlmConfig,
    HeadlessLlmRequest,
    KimiCodeLlmConfig,
    KimiCodeLlmRequest,
    Message,
    OpenAILlmRequest,
    OpenAIReasoning,
    ProviderName,
    ReasoningBudget,
    ThinkingLevel,
    build_default_registry,
    build_request_from_config,
    parse_llm_config,
    parse_llm_request,
)


def LlmConfig(**kwargs: object):
    return parse_llm_config(kwargs)


def LlmRequest(**kwargs: object):
    return parse_llm_request(kwargs)


def test_basic_construction() -> None:
    config = LlmConfig(provider=ProviderName.OPENAI, model="gpt-4.1-mini")

    assert config.provider == ProviderName.OPENAI
    assert config.model == "gpt-4.1-mini"
    assert config.temperature is None
    assert config.top_p is None
    assert config.max_tokens is None
    assert config.effort == EffortSpec.NA
    assert config.reasoning is None


def test_construction_with_all_shape_fields() -> None:
    reasoning = GoogleReasoning(thinking_level=ThinkingLevel.LOW)
    config = LlmConfig(
        provider=ProviderName.GOOGLE,
        model="gemini-3-flash-preview",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        reasoning=reasoning,
    )

    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.max_tokens == 1024
    assert config.reasoning is not None
    assert config.reasoning.kind == ProviderName.GOOGLE


def test_headless_config_rejects_sampling_fields() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        HeadlessLlmConfig.model_validate(
            {
                "provider": ProviderName.CODEX,
                "model": "gpt-5.4-mini",
                "temperature": 0.2,
            }
        )

    with pytest.raises(ValidationError, match="top_p"):
        parse_llm_config(
            {
                "provider": ProviderName.CLAUDE_CODE,
                "model": "claude-sonnet-4-6",
                "top_p": 0.5,
            }
        )


def test_headless_request_rejects_sampling_and_max_tokens() -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        parse_llm_request(
            {
                "provider": ProviderName.CODEX,
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
            }
        )

    with pytest.raises(ValidationError, match="temperature"):
        HeadlessLlmRequest.model_validate(
            {
                "provider": ProviderName.CLAUDE_CODE,
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.2,
            }
        )


def test_kimi_code_config_rejects_sampling_fields() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        KimiCodeLlmConfig.model_validate(
            {
                "provider": ProviderName.KIMI_CODE,
                "model": "kimi-for-coding",
                "max_tokens": 256,
                "effort": EffortSpec.HIGH,
                "temperature": 0.2,
            }
        )

    with pytest.raises(ValidationError, match="top_p"):
        parse_llm_config(
            {
                "provider": ProviderName.KIMI_CODE,
                "model": "kimi-for-coding",
                "max_tokens": 256,
                "effort": "high",
                "top_p": 0.5,
            }
        )


def test_kimi_code_request_rejects_sampling_fields() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        KimiCodeLlmRequest.model_validate(
            {
                "provider": ProviderName.KIMI_CODE,
                "model": "kimi-for-coding",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 256,
                "effort": EffortSpec.HIGH,
                "temperature": 0.2,
            }
        )

    with pytest.raises(ValidationError, match="top_p"):
        parse_llm_request(
            {
                "provider": ProviderName.KIMI_CODE,
                "model": "kimi-for-coding",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 256,
                "effort": "high",
                "top_p": 0.5,
            }
        )


def test_orchestrator_build_request_applies_config_values() -> None:
    config = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        temperature=0.5,
        max_tokens=100,
    )
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

    assert request.provider == ProviderName.OPENAI
    assert isinstance(request, OpenAILlmRequest)
    assert request.model == "gpt-4.1-mini"
    assert request.messages == messages
    assert request.temperature == 0.5
    assert request.max_tokens == 100
    assert request.top_p is None
    assert request.effort == EffortSpec.NA
    assert request.reasoning is None
    assert request.metadata == {}


def test_orchestrator_build_request_uses_provider_reasoning_defaults() -> None:
    config = LlmConfig(
        provider=ProviderName.GOOGLE,
        model="gemini-2.5-flash",
    )
    messages = [Message(role="user", content="Think about this")]

    registry = build_default_registry()
    try:
        request = build_request_from_config(
            registry.get(config.provider),
            config,
            messages,
        )
    finally:
        registry.close()

    assert request.reasoning is not None
    assert request.reasoning == GoogleReasoning(
        thinking_level=ThinkingLevel.OFF
    )


def test_semantic_validation_is_orchestrator_owned() -> None:
    request = LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        messages=[Message(role="user", content="Hello")],
        effort=EffortSpec.MEDIUM,
    )
    registry = build_default_registry()
    try:
        with pytest.raises(ValueError, match="effort is not supported"):
            registry.get(ProviderName.OPENAI).validate_request(request)
    finally:
        registry.close()


def test_anthropic_orchestrator_requires_max_tokens() -> None:
    request = LlmRequest(
        provider=ProviderName.ANTHROPIC,
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="Hello")],
        effort=EffortSpec.MEDIUM,
    )
    registry = build_default_registry()
    try:
        with pytest.raises(ValueError, match="max_tokens is required"):
            registry.get(ProviderName.ANTHROPIC).validate_request(request)
    finally:
        registry.close()


def test_openai_sampling_semantics_are_orchestrator_owned() -> None:
    request = LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-5.4",
        messages=[Message(role="user", content="Hello")],
        temperature=0.2,
        reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.LOW),
    )
    registry = build_default_registry()
    try:
        with pytest.raises(ValueError, match="thinking_level='off'"):
            registry.get(ProviderName.OPENAI).validate_request(request)
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("provider", "model", "request_fields", "error_match"),
    [
        (
            ProviderName.GOOGLE,
            "gemini-2.5-flash",
            {"reasoning": AnthropicReasoning(thinking_level=ThinkingLevel.NA)},
            "google reasoning is not supported",
        ),
        (
            ProviderName.KIMI_CODE,
            "kimi-for-coding",
            {
                "max_tokens": 2048,
                "effort": EffortSpec.HIGH,
                "reasoning": ReasoningBudget(tokens=1024),
            },
            "kimi-code requires anthropic reasoning configs",
        ),
        (
            ProviderName.MINIMAX,
            "MiniMax-M2.7",
            {"effort": EffortSpec.LOW},
            "reasoning is required",
        ),
        (
            ProviderName.CLAUDE_CODE,
            "claude-sonnet-4-6",
            {
                "effort": EffortSpec.MEDIUM,
                "reasoning": AnthropicReasoning(
                    thinking_level=ThinkingLevel.OFF
                ),
            },
            "only supports anthropic thinking_level='adaptive'",
        ),
        (
            ProviderName.CODEX,
            "gpt-5.4-mini",
            {"reasoning": AnthropicReasoning(thinking_level=ThinkingLevel.NA)},
            "reasoning is not supported",
        ),
        (
            ProviderName.GLM,
            "glm-4.5",
            {},
            "reasoning is required",
        ),
        (
            ProviderName.OPENROUTER,
            "openai/gpt-4.1",
            {},
            "not in the curated allowlist",
        ),
    ],
)
def test_provider_semantic_validation_is_orchestrator_owned(
    provider: ProviderName,
    model: str,
    request_fields: dict[str, object],
    error_match: str,
) -> None:
    request = LlmRequest(
        provider=provider,
        model=model,
        messages=[Message(role="user", content="Hello")],
        **request_fields,
    )
    registry = build_default_registry()
    try:
        with pytest.raises(ValueError, match=error_match):
            registry.get(provider).validate_request(request)
    finally:
        registry.close()


def test_glm_orchestrator_accepts_native_reasoning() -> None:
    request = LlmRequest(
        provider=ProviderName.GLM,
        model="glm-4.5",
        messages=[Message(role="user", content="Hello")],
        reasoning=GlmReasoning(thinking_level=ThinkingLevel.OFF),
    )
    registry = build_default_registry()
    try:
        assert registry.get(ProviderName.GLM).validate_request(request) == []
    finally:
        registry.close()


def test_provider_reasoning_shape_can_parse_before_validation() -> None:
    config = LlmConfig(
        provider=ProviderName.GOOGLE,
        model="gemini-2.5-flash",
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.NA),
    )

    assert config.reasoning == AnthropicReasoning(
        thinking_level=ThinkingLevel.NA
    )


def test_build_request_from_config_rejects_provider_mismatch() -> None:
    config = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
    )
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

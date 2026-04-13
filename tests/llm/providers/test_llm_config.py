from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.config import HeadlessLlmConfig, parse_llm_config
from dr_llm.llm.request import HeadlessLlmRequest, parse_llm_request
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GlmReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningBudget,
    ThinkingLevel,
)


def LlmConfig(**kwargs: object):
    return parse_llm_config(kwargs)


def LlmRequest(**kwargs: object):
    return parse_llm_request(kwargs)


def test_basic_construction() -> None:
    config = LlmConfig(provider="openai", model="gpt-4.1-mini")

    assert config.provider == "openai"
    assert config.model == "gpt-4.1-mini"
    assert config.temperature == 1.0
    assert config.top_p == 0.95
    assert config.max_tokens is None
    assert config.effort == EffortSpec.NA
    assert config.reasoning is None


def test_construction_with_all_fields() -> None:
    reasoning = GoogleReasoning(thinking_level=ThinkingLevel.LOW)
    config = LlmConfig(
        provider="google",
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
    assert config.reasoning.kind == "google"


def test_frozen() -> None:
    config = LlmConfig(provider="openai", model="gpt-4.1-mini")

    with pytest.raises(ValidationError):
        config.provider = "anthropic"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(provider="openai", model="gpt-4.1-mini", extra_field="nope")


def test_headless_config_rejects_sampling_fields() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        HeadlessLlmConfig(
            provider="codex",
            model="gpt-5.4-mini",
            temperature=0.2,  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError, match="top_p"):
        parse_llm_config(
            {
                "provider": "claude-code",
                "model": "claude-sonnet-4-6",
                "top_p": 0.5,
            }
        )


def test_headless_request_rejects_sampling_and_max_tokens() -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        parse_llm_request(
            {
                "provider": "codex",
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
            }
        )

    with pytest.raises(ValidationError, match="temperature"):
        HeadlessLlmRequest(
            provider="claude-code",
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="hi")],
            temperature=0.2,  # type: ignore[call-arg]
        )


def test_to_request() -> None:
    config = LlmConfig(
        provider="openai",
        model="gpt-4.1-mini",
        temperature=0.5,
        max_tokens=100,
    )
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]

    request = config.to_request(messages)

    assert request.provider == "openai"
    assert request.model == "gpt-4.1-mini"
    assert request.messages == messages
    assert request.temperature == 0.5
    assert request.max_tokens == 100
    assert request.top_p == 0.95
    assert request.effort == EffortSpec.NA
    assert request.reasoning is None
    assert request.metadata == {}


def test_to_request_with_reasoning() -> None:
    config = LlmConfig(
        provider="google",
        model="gemini-2.5-flash",
        reasoning=ReasoningBudget(tokens=512),
    )
    messages = [Message(role="user", content="Think about this")]

    request = config.to_request(messages)

    assert request.reasoning is not None
    assert request.reasoning.kind == "budget"


def test_to_request_with_effort() -> None:
    config = LlmConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_tokens=256,
        effort=EffortSpec.MEDIUM,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
    )
    messages = [Message(role="user", content="Think about this")]

    request = config.to_request(messages)

    assert request.effort == EffortSpec.MEDIUM


def test_rejects_non_na_effort_for_unsupported_provider() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-4.1-mini",
            effort=EffortSpec.MEDIUM,
        )


def test_rejects_na_effort_for_supported_anthropic_model() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_tokens=256,
            effort=EffortSpec.NA,
        )


def test_rejects_non_na_effort_for_unsupported_anthropic_model() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            effort=EffortSpec.MEDIUM,
        )


def test_rejects_na_effort_for_supported_claude_code_model() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="claude-code",
            model="claude-sonnet-4-6",
            effort=EffortSpec.NA,
        )


def test_rejects_non_na_effort_for_unsupported_claude_code_model() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="claude-code",
            model="claude-haiku-4-5-20251001",
            effort=EffortSpec.MEDIUM,
        )


def test_claude_code_accepts_max_effort() -> None:
    LlmConfig(
        provider="claude-code",
        model="claude-sonnet-4-6",
        effort=EffortSpec.MAX,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )


def test_minimax_requires_effort_and_reasoning() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="minimax",
            model="MiniMax-M2.7",
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="minimax",
            model="MiniMax-M2.7",
            effort=EffortSpec.LOW,
        )


def test_minimax_accepts_all_effort_levels() -> None:
    for effort in (
        EffortSpec.LOW,
        EffortSpec.MEDIUM,
        EffortSpec.HIGH,
        EffortSpec.MAX,
    ):
        LlmConfig(
            provider="minimax",
            model="MiniMax-M2.7",
            effort=effort,
            reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.NA),
        )


def test_minimax_accepts_optional_max_tokens_and_explicit_na_reasoning() -> None:
    LlmConfig(
        provider="minimax",
        model="MiniMax-M2.7",
        effort=EffortSpec.LOW,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.NA),
    )
    LlmConfig(
        provider="minimax",
        model="MiniMax-M2.7",
        max_tokens=4096,
        effort=EffortSpec.LOW,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.NA),
    )


def test_minimax_rejects_explicit_thinking_controls() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="minimax",
            model="MiniMax-M2.7",
            effort=EffortSpec.LOW,
            reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="minimax",
            model="MiniMax-M2.7",
            effort=EffortSpec.LOW,
            reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="minimax",
            model="MiniMax-M2.7",
            effort=EffortSpec.LOW,
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=2048,
            ),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="minimax",
            model="MiniMax-M2.7",
            effort=EffortSpec.LOW,
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                display="omitted",
            ),
        )


def test_kimi_code_requires_effort_and_max_tokens() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="kimi-code",
            model="kimi-for-coding",
            max_tokens=256,
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="kimi-code",
            model="kimi-for-coding",
            effort=EffortSpec.HIGH,
        )


def test_kimi_code_accepts_off_adaptive_and_budget_with_effort() -> None:
    for reasoning in (
        None,
        AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
        AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
        AnthropicReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=1024,
        ),
    ):
        LlmConfig(
            provider="kimi-code",
            model="kimi-for-coding",
            max_tokens=2048,
            effort=EffortSpec.HIGH,
            reasoning=reasoning,
        )


def test_kimi_code_rejects_display_and_top_level_budget() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="kimi-code",
            model="kimi-for-coding",
            max_tokens=2048,
            effort=EffortSpec.HIGH,
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=1024,
                display="omitted",
            ),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="kimi-code",
            model="kimi-for-coding",
            max_tokens=2048,
            effort=EffortSpec.HIGH,
            reasoning=ReasoningBudget(tokens=1024),
        )


def test_llm_request_rejects_na_effort_for_supported_anthropic_model() -> None:
    with pytest.raises(ValidationError):
        LlmRequest(
            provider="anthropic",
            model="claude-opus-4-6",
            messages=[Message(role="user", content="Hello")],
            max_tokens=256,
            effort=EffortSpec.NA,
        )


def test_anthropic_config_requires_max_tokens() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            effort=EffortSpec.MEDIUM,
        )


def test_anthropic_request_requires_max_tokens() -> None:
    with pytest.raises(ValidationError):
        LlmRequest(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hello")],
            effort=EffortSpec.MEDIUM,
        )


def test_glm_request_requires_explicit_reasoning() -> None:
    with pytest.raises(ValidationError):
        LlmRequest(
            provider="glm",
            model="glm-4.5",
            messages=[Message(role="user", content="Hello")],
        )


def test_model_dump_roundtrip() -> None:
    config = LlmConfig(
        provider="google",
        model="gemini-3-flash-preview",
        temperature=0.7,
        reasoning=GoogleReasoning(thinking_level=ThinkingLevel.MINIMAL),
    )
    dumped = config.model_dump()
    restored = LlmConfig(**dumped)

    assert restored == config


def test_rejects_reasoning_for_unsupported_model() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-4.1-mini",
            reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.LOW),
        )


def test_supported_models_require_explicit_reasoning() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5-mini",
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="codex",
            model="gpt-5.1-codex-mini",
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="glm",
            model="glm-4.5",
        )


def test_rejects_provider_specific_reasoning_on_wrong_provider() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5-mini",
            reasoning=GoogleReasoning(thinking_level=ThinkingLevel.LOW),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5-mini",
            reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.NA),
        )


def test_openai_gpt5_family_accepts_provider_shaped_reasoning() -> None:
    LlmConfig(
        provider="openai",
        model="gpt-5-mini",
        reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL),
    )

    LlmConfig(
        provider="openai",
        model="gpt-5.2",
        reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.OFF),
    )
    LlmConfig(
        provider="openrouter",
        model="openai/gpt-oss-20b",
        reasoning=OpenRouterReasoning(effort="medium"),
    )


def test_openrouter_requires_allowlisted_models() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openrouter",
            model="openai/gpt-4.1",
        )


def test_openrouter_validates_manual_reasoning_modes() -> None:
    LlmConfig(
        provider="openrouter",
        model="deepseek/deepseek-chat-v3.1",
        reasoning=OpenRouterReasoning(enabled=False),
    )
    LlmConfig(
        provider="openrouter",
        model="deepseek/deepseek-r1",
        reasoning=OpenRouterReasoning(enabled=True),
    )
    LlmConfig(
        provider="openrouter",
        model="deepseek/deepseek-chat",
    )

    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openrouter",
            model="deepseek/deepseek-chat-v3.1",
            reasoning=OpenRouterReasoning(effort="low"),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openrouter",
            model="deepseek/deepseek-r1",
            reasoning=OpenRouterReasoning(enabled=False),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openrouter",
            model="deepseek/deepseek-chat",
            reasoning=OpenRouterReasoning(enabled=True),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openrouter",
            model="openai/gpt-oss-20b",
            reasoning=OpenRouterReasoning(enabled=True),
        )


def test_openrouter_supported_models_require_explicit_reasoning() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openrouter",
            model="deepseek/deepseek-chat-v3.1",
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openrouter",
            model="openai/gpt-oss-20b",
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openrouter",
            model="deepseek/deepseek-chat-v3.1",
            reasoning=ReasoningBudget(tokens=1024),
        )


def test_openai_gpt5_rejects_off_before_51() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5-mini",
            reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.OFF),
        )


def test_openai_51_plus_rejects_minimal() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5.1",
            reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5.1",
            reasoning=ReasoningBudget(tokens=1024),
        )


def test_codex_accepts_provider_shaped_reasoning() -> None:
    LlmConfig(
        provider="codex",
        model="gpt-5.1-codex-mini",
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.LOW),
    )
    LlmConfig(
        provider="codex",
        model="gpt-5.4",
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.OFF),
    )
    LlmConfig(
        provider="codex",
        model="gpt-5.1-codex-mini",
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.XHIGH),
    )


def test_codex_rejects_unsupported_model_specific_thinking_levels() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="codex",
            model="gpt-5.1-codex-mini",
            reasoning=CodexReasoning(thinking_level=ThinkingLevel.MINIMAL),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="codex",
            model="gpt-5.2-codex",
            reasoning=CodexReasoning(thinking_level=ThinkingLevel.OFF),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="codex",
            model="gpt-5.1-codex-mini",
            reasoning=CodexReasoning(thinking_level=ThinkingLevel.OFF),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="codex",
            model="gpt-5-codex",
            reasoning=CodexReasoning(thinking_level=ThinkingLevel.MINIMAL),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="codex",
            model="gpt-5.4",
            reasoning=CodexReasoning(thinking_level=ThinkingLevel.NA),
        )


def test_openai_and_codex_reject_top_level_effort() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5-mini",
            effort=EffortSpec.HIGH,
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="codex",
            model="gpt-5.1-codex-mini",
            effort=EffortSpec.HIGH,
        )


def test_glm_requires_explicit_reasoning_and_rejects_effort() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="glm",
            model="glm-4.5",
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="glm",
            model="glm-4.5",
            effort=EffortSpec.HIGH,
            reasoning=GlmReasoning(thinking_level=ThinkingLevel.OFF),
        )


def test_google_supported_models_require_explicit_reasoning() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-2.5-flash",
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-3-flash-preview",
        )


def test_google_budget_models_accept_only_budget_family_controls() -> None:
    LlmConfig(
        provider="google",
        model="gemini-2.5-flash",
        reasoning=GoogleReasoning(thinking_level=ThinkingLevel.OFF),
    )
    LlmConfig(
        provider="google",
        model="gemini-2.5-flash",
        reasoning=GoogleReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    LlmConfig(
        provider="google",
        model="gemini-2.5-flash",
        reasoning=GoogleReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=1024,
        ),
    )
    LlmConfig(
        provider="google",
        model="gemini-2.5-flash",
        reasoning=ReasoningBudget(tokens=1024),
    )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-2.5-flash",
            reasoning=GoogleReasoning(thinking_level=ThinkingLevel.NA),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-2.5-flash",
            reasoning=GoogleReasoning(thinking_level=ThinkingLevel.MINIMAL),
        )


def test_google_level_models_accept_only_level_controls() -> None:
    LlmConfig(
        provider="google",
        model="gemini-3-flash-preview",
        reasoning=GoogleReasoning(thinking_level=ThinkingLevel.MINIMAL),
    )
    LlmConfig(
        provider="google",
        model="gemma-4-31b-it",
        reasoning=GoogleReasoning(thinking_level=ThinkingLevel.HIGH),
    )
    LlmConfig(
        provider="google",
        model="gemma-4-31b-it",
        reasoning=GoogleReasoning(thinking_level=ThinkingLevel.MINIMAL),
    )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-3-flash-preview",
            reasoning=GoogleReasoning(thinking_level=ThinkingLevel.OFF),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-3-flash-preview",
            reasoning=GoogleReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=1024,
            ),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemma-4-31b-it",
            reasoning=GoogleReasoning(thinking_level=ThinkingLevel.LOW),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemma-4-31b-it",
            reasoning=GoogleReasoning(thinking_level=ThinkingLevel.MEDIUM),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-3-flash-preview",
            reasoning=ReasoningBudget(tokens=1024),
        )


def test_google_unsupported_models_allow_omission_and_reject_explicit_reasoning() -> None:
    LlmConfig(
        provider="google",
        model="gemini-2.0-flash-lite-001",
    )
    LlmConfig(
        provider="google",
        model="gemma-3-1b-it",
    )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-2.0-flash-lite-001",
            reasoning=GoogleReasoning(thinking_level=ThinkingLevel.MINIMAL),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemma-3-1b-it",
            reasoning=GoogleReasoning(thinking_level=ThinkingLevel.MINIMAL),
        )


def test_glm_accepts_explicit_off_and_adaptive() -> None:
    LlmConfig(
        provider="glm",
        model="glm-4.5",
        reasoning=GlmReasoning(thinking_level=ThinkingLevel.OFF),
    )
    LlmConfig(
        provider="glm",
        model="glm-4.5",
        reasoning=GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="glm",
            model="glm-4.5",
            reasoning=ReasoningBudget(tokens=1024),
        )


def test_glm_rejects_unsupported_thinking_levels_and_wrong_kinds() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="glm",
            model="glm-4.5",
            reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.LOW),
        )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="glm",
            model="glm-4.5",
            reasoning=GlmReasoning(thinking_level=ThinkingLevel.HIGH),
        )


def test_allows_combining_effort_with_anthropic_off() -> None:
    LlmConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_tokens=256,
        effort=EffortSpec.HIGH,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
    )


def test_anthropic_adaptive_requires_model_support_not_effort() -> None:
    LlmConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_tokens=256,
        effort=EffortSpec.MEDIUM,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="anthropic",
            model="claude-opus-4-5",
            max_tokens=256,
            reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
        )


def test_anthropic_budget_requires_budget_supported_model() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_tokens=4096,
            effort=EffortSpec.MEDIUM,
            reasoning=AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=2048,
            ),
        )


def test_claude_code_accepts_adaptive_only_for_46_models() -> None:
    LlmConfig(
        provider="claude-code",
        model="claude-sonnet-4-6",
        effort=EffortSpec.MEDIUM,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="claude-code",
            model="claude-sonnet-4-6",
            effort=EffortSpec.MEDIUM,
            reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
        )


def test_claude_code_haiku_accepts_na_only() -> None:
    LlmConfig(
        provider="claude-code",
        model="claude-haiku-4-5-20251001",
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.NA),
    )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="claude-code",
            model="claude-haiku-4-5-20251001",
            reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
        )

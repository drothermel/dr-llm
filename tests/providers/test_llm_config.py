from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_llm.providers.effort import EffortSpec
from dr_llm.providers.llm_config import LlmConfig
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.models import Message
from dr_llm.providers.reasoning import (
    AnthropicReasoning,
    GoogleReasoning,
    ReasoningBudget,
    ReasoningOff,
)


def test_basic_construction() -> None:
    config = LlmConfig(provider="openai", model="gpt-4.1-mini")

    assert config.provider == "openai"
    assert config.model == "gpt-4.1-mini"
    assert config.temperature is None
    assert config.top_p is None
    assert config.max_tokens is None
    assert config.effort == EffortSpec.NA
    assert config.reasoning is None


def test_construction_with_all_fields() -> None:
    reasoning = GoogleReasoning(thinking_level="low")
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
        LlmConfig(provider="openai", model="gpt-4.1-mini", extra_field="nope")  # type: ignore[call-arg]


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
    assert request.top_p is None
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
        effort=EffortSpec.MEDIUM,
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
            effort=EffortSpec.NA,
        )


def test_rejects_non_na_effort_for_unsupported_anthropic_model() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
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


def test_llm_request_rejects_na_effort_for_supported_anthropic_model() -> None:
    with pytest.raises(ValidationError):
        LlmRequest(
            provider="anthropic",
            model="claude-opus-4-6",
            messages=[Message(role="user", content="Hello")],
            effort=EffortSpec.NA,
        )


def test_model_dump_roundtrip() -> None:
    config = LlmConfig(
        provider="google",
        model="gemini-3-flash-preview",
        temperature=0.7,
        reasoning=GoogleReasoning(thinking_level="minimal"),
    )
    dumped = config.model_dump()
    restored = LlmConfig(**dumped)

    assert restored == config


def test_rejects_reasoning_for_unsupported_model() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-4.1-mini",
            reasoning=ReasoningBudget(tokens=1024),
        )


def test_rejects_provider_specific_reasoning_on_wrong_provider() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5-mini",
            reasoning=GoogleReasoning(thinking_level="low"),
        )


def test_openai_gpt5_family_allows_top_level_effort() -> None:
    LlmConfig(
        provider="openai",
        model="gpt-5-mini",
        effort=EffortSpec.HIGH,
    )

    LlmConfig(
        provider="openai",
        model="gpt-5.2",
        effort=EffortSpec.LOW,
    )
    LlmConfig(
        provider="openrouter",
        model="openai/gpt-5.1",
        effort=EffortSpec.MEDIUM,
    )


def test_rejects_combining_effort_with_reasoning_off() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5-mini",
            effort=EffortSpec.HIGH,
            reasoning=ReasoningOff(),
        )


def test_anthropic_adaptive_requires_model_support_not_effort() -> None:
    LlmConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        effort=EffortSpec.MEDIUM,
        reasoning=AnthropicReasoning(thinking_mode="adaptive"),
    )
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="anthropic",
            model="claude-opus-4-5",
            reasoning=AnthropicReasoning(thinking_mode="adaptive"),
        )


def test_google_flash_budget_zero_is_not_manual_budget() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="google",
            model="gemini-2.5-flash",
            reasoning=GoogleReasoning(thinking_budget=0),
        )
    LlmConfig(
        provider="google",
        model="gemini-2.5-flash",
        reasoning=ReasoningOff(),
    )

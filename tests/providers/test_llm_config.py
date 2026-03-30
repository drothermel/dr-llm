from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_llm.providers.llm_config import LlmConfig
from dr_llm.providers.models import Message
from dr_llm.providers.reasoning import (
    GoogleReasoning,
    ReasoningBudget,
    ReasoningEffort,
)


def test_basic_construction() -> None:
    config = LlmConfig(provider="openai", model="gpt-4.1-mini")

    assert config.provider == "openai"
    assert config.model == "gpt-4.1-mini"
    assert config.temperature is None
    assert config.top_p is None
    assert config.max_tokens is None
    assert config.reasoning is None


def test_construction_with_all_fields() -> None:
    reasoning = ReasoningEffort(level="high")
    config = LlmConfig(
        provider="openai",
        model="gpt-5-mini",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        reasoning=reasoning,
    )

    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.max_tokens == 1024
    assert config.reasoning is not None
    assert config.reasoning.kind == "effort"


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
            reasoning=ReasoningEffort(level="high"),
        )


def test_rejects_provider_specific_reasoning_on_wrong_provider() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(
            provider="openai",
            model="gpt-5-mini",
            reasoning=GoogleReasoning(thinking_level="low"),
        )

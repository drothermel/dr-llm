from __future__ import annotations

from dr_llm.llm.config import (
    ApiLlmConfig,
    HeadlessLlmConfig,
    KimiCodeLlmConfig,
    OpenAILlmConfig,
)
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    GoogleReasoning,
    OpenAIReasoning,
    ThinkingLevel,
)
from dr_llm.llm.request import (
    ApiLlmRequest,
    HeadlessLlmRequest,
    KimiCodeLlmRequest,
    OpenAILlmRequest,
)


def build_api_shapes() -> tuple[ApiLlmConfig, ApiLlmRequest]:
    config = ApiLlmConfig(
        provider="openrouter",
        model="deepseek/deepseek-chat",
        temperature=0.8,
        top_p=0.9,
        max_tokens=128,
    )
    request = ApiLlmRequest(
        provider="google",
        model="gemini-3-flash-preview",
        messages=[Message(role="user", content="hello")],
        temperature=0.7,
        top_p=0.8,
        max_tokens=256,
        reasoning=GoogleReasoning(thinking_level=ThinkingLevel.HIGH),
    )
    return config, request


def build_openai_shapes() -> tuple[OpenAILlmConfig, OpenAILlmRequest]:
    config = OpenAILlmConfig(
        provider="openai",
        model="gpt-4.1-mini",
    )
    request = OpenAILlmRequest(
        provider="openai",
        model="gpt-5.4",
        messages=[Message(role="user", content="hello")],
        temperature=0.7,
        top_p=0.8,
        reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.OFF),
    )
    return config, request


def build_headless_shapes() -> tuple[HeadlessLlmConfig, HeadlessLlmRequest]:
    config = HeadlessLlmConfig(
        provider="claude-code",
        model="claude-sonnet-4-6",
        effort=EffortSpec.HIGH,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    request = HeadlessLlmRequest(
        provider="codex",
        model="gpt-5.4-mini",
        messages=[Message(role="user", content="hello")],
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.LOW),
    )
    return config, request


def build_kimi_shapes() -> tuple[KimiCodeLlmConfig, KimiCodeLlmRequest]:
    config = KimiCodeLlmConfig(
        provider="kimi-code",
        model="kimi-for-coding",
        max_tokens=256,
        effort=EffortSpec.HIGH,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    request = KimiCodeLlmRequest(
        provider="kimi-code",
        model="kimi-for-coding",
        messages=[Message(role="user", content="hello")],
        max_tokens=256,
        effort=EffortSpec.HIGH,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    return config, request


def test_build_shapes_exist() -> None:
    api_config, api_request = build_api_shapes()
    openai_config, openai_request = build_openai_shapes()
    headless_config, headless_request = build_headless_shapes()
    kimi_config, kimi_request = build_kimi_shapes()

    assert api_config.provider == "openrouter"
    assert api_request.provider == "google"
    assert openai_config.provider == "openai"
    assert openai_request.provider == "openai"
    assert headless_config.provider == "claude-code"
    assert headless_request.provider == "codex"
    assert kimi_config.provider == "kimi-code"
    assert kimi_request.provider == "kimi-code"

from __future__ import annotations

from dr_llm.llm.config import ApiLlmConfig, HeadlessLlmConfig
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import AnthropicReasoning, ThinkingLevel
from dr_llm.llm.request import ApiLlmRequest, HeadlessLlmRequest


def build_api_shapes() -> tuple[ApiLlmConfig, ApiLlmRequest]:
    config = ApiLlmConfig(
        provider="openai",
        model="gpt-4.1-mini",
        temperature=0.8,
        top_p=0.9,
        max_tokens=128,
    )
    request = ApiLlmRequest(
        provider="google",
        model="gemini-2.5-flash",
        messages=[Message(role="user", content="hello")],
        temperature=0.7,
        top_p=0.8,
        max_tokens=256,
    )
    return config, request


def build_headless_shapes() -> tuple[HeadlessLlmConfig, HeadlessLlmRequest]:
    config = HeadlessLlmConfig(
        provider="claude-code",
        model="claude-sonnet-4-6",
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    request = HeadlessLlmRequest(
        provider="codex",
        model="gpt-5.4-mini",
        messages=[Message(role="user", content="hello")],
    )
    return config, request

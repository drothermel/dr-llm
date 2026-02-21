from __future__ import annotations

import httpx
import pytest

from llm_pool.errors import ProviderTransportError
from llm_pool.providers.anthropic import AnthropicAdapter, AnthropicConfig
from llm_pool.providers.google import GoogleAdapter, GoogleConfig
from llm_pool.providers.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from llm_pool.types import LlmRequest, Message


def _invalid_json_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(status_code=200, text="{")

    return httpx.MockTransport(handler)


def test_openai_compat_invalid_json_is_transport_error() -> None:
    request = LlmRequest(
        provider="openrouter",
        model="openai/gpt-4o-mini",
        messages=[Message(role="user", content="hi")],
    )
    with OpenAICompatAdapter(
        name="openrouter",
        config=OpenAICompatConfig(base_url="https://openrouter.ai/api/v1", api_key="x"),
        client=httpx.Client(transport=_invalid_json_transport()),
    ) as adapter:
        with pytest.raises(ProviderTransportError, match="invalid JSON response"):
            adapter.generate(request)


def test_anthropic_invalid_json_is_transport_error() -> None:
    request = LlmRequest(
        provider="anthropic",
        model="claude-test",
        messages=[Message(role="user", content="hi")],
    )
    with AnthropicAdapter(
        config=AnthropicConfig(
            api_key="x", base_url="https://api.anthropic.com/v1/messages"
        ),
        client=httpx.Client(transport=_invalid_json_transport()),
    ) as adapter:
        with pytest.raises(ProviderTransportError, match="invalid JSON response"):
            adapter.generate(request)


def test_google_invalid_json_is_transport_error() -> None:
    adapter = GoogleAdapter(
        config=GoogleConfig(
            api_key="x", base_url="https://generativelanguage.googleapis.com/v1beta"
        ),
        client=httpx.Client(transport=_invalid_json_transport()),
    )
    request = LlmRequest(
        provider="google",
        model="gemini-test",
        messages=[Message(role="user", content="hi")],
    )
    with pytest.raises(ProviderTransportError, match="invalid JSON response"):
        adapter.generate(request)

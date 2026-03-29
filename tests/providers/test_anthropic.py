from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from dr_llm.errors import ProviderTransportError
from dr_llm.providers.anthropic.adapter import AnthropicAdapter
from dr_llm.providers.anthropic.config import AnthropicConfig
from dr_llm.providers.models import Message
from tests.conftest import make_request
from tests.providers.conftest import make_http_client

_MOCK_RESPONSE: dict[str, Any] = {
    "content": [{"type": "text", "text": "done"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "stop_reason": "end_turn",
}


def _make_config() -> AnthropicConfig:
    return AnthropicConfig(api_key="x", base_url="https://api.anthropic.com/v1/messages")


def test_payload_serializes_messages() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = AnthropicAdapter(config=_make_config(), client=client)

    request = make_request(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
        ],
    )
    result = adapter.generate(request)

    assert result.text == "done"

    payload = captured["payload"]
    assert payload["system"] == "You are helpful."
    assert "tools" not in payload

    messages = payload["messages"]
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
    ]


def test_invalid_json_raises_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, text="{")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = AnthropicAdapter(config=_make_config(), client=client)

    with pytest.raises(ProviderTransportError, match="invalid JSON response"):
        adapter.generate(make_request(provider="anthropic", model="claude-3-5-haiku-20241022"))

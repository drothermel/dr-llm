from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from dr_llm.errors import ProviderTransportError
from dr_llm.providers.api_provider_config import APIProviderConfig
from dr_llm.providers.google.adapter import GoogleAdapter
from dr_llm.providers.models import Message
from dr_llm.providers.reasoning import GoogleReasoning, ReasoningBudget
from tests.conftest import make_request
from tests.providers.conftest import make_http_client

_GOOGLE_CONFIG = APIProviderConfig(
    name="google",
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key_env="GOOGLE_API_KEY",
    api_key="x",
)

_MOCK_RESPONSE = {
    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
}


def test_payload_serializes_messages() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = GoogleAdapter(config=_GOOGLE_CONFIG, client=client)

    request = make_request(
        provider="google",
        model="gemini-test",
        messages=[
            Message(role="system", content="Be concise."),
            Message(role="user", content="find item"),
            Message(role="assistant", content="previous answer"),
        ],
    )
    response = adapter.generate(request)

    assert response.text == "done"
    payload = cast(dict[str, Any], captured["payload"])
    headers = cast(dict[str, str], captured["headers"])
    assert headers["x-goog-api-key"] == "x"
    assert "?key=" not in cast(str, captured["url"])
    assert payload["systemInstruction"] == {"parts": [{"text": "Be concise."}]}
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "find item"}]},
        {"role": "model", "parts": [{"text": "previous answer"}]},
    ]


def test_payload_serializes_budget_reasoning_under_thinking_config() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = GoogleAdapter(config=_GOOGLE_CONFIG, client=client)

    request = make_request(
        provider="google",
        model="gemini-2.5-flash",
        reasoning=ReasoningBudget(tokens=512),
    )
    adapter.generate(request)

    payload = cast(dict[str, Any], captured["payload"])
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}


def test_payload_serializes_dynamic_google_reasoning() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = GoogleAdapter(config=_GOOGLE_CONFIG, client=client)

    request = make_request(
        provider="google",
        model="gemini-2.5-flash",
        reasoning=GoogleReasoning(dynamic=True),
    )
    adapter.generate(request)

    payload = cast(dict[str, Any], captured["payload"])
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}


def test_invalid_json_raises_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(status_code=200, text="{")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = GoogleAdapter(config=_GOOGLE_CONFIG, client=client)

    request = make_request(
        provider="google",
        model="gemini-test",
        messages=[Message(role="user", content="hi")],
    )
    with pytest.raises(ProviderTransportError, match="invalid JSON response"):
        adapter.generate(request)

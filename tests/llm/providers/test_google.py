from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import httpx
import pytest

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.llm.providers.api_config import APIProviderConfig
from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.google.provider import GoogleProvider
from dr_llm.llm.providers.google.request import GoogleRequest
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import (
    GoogleReasoning,
    ReasoningBudget,
    ThinkingLevel,
)
from dr_llm.llm.request import ApiLlmRequest, KimiCodeLlmRequest
from tests.conftest import make_request
from tests.llm.providers.conftest import make_http_client

_GOOGLE_CONFIG = APIProviderConfig(
    name="google",
    base_url="https://generativelanguage.googleapis.com/v1beta",
    api_key_env="GOOGLE_API_KEY",
    api_key="x",
)

_MOCK_RESPONSE = {
    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
    "usageMetadata": {
        "promptTokenCount": 1,
        "candidatesTokenCount": 2,
        "totalTokenCount": 3,
    },
}
_THOUGHT_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "private chain of thought", "thought": True},
                    {"text": "OK"},
                ]
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 1,
        "candidatesTokenCount": 2,
        "totalTokenCount": 3,
    },
}


def _make_api_request(overrides: Mapping[str, Any] | None = None) -> ApiLlmRequest:
    return cast(ApiLlmRequest, make_request(**(overrides or {})))


def test_rejects_unsupported_message_role() -> None:
    """model_construct can bypass validation; unsupported roles must not be dropped silently."""
    tool_msg = Message.model_construct(role="tool", content="secret tool output")
    request = _make_api_request(
        {
            "provider": "google",
            "model": "gemini-test",
            "messages": [tool_msg],
        }
    )
    with pytest.raises(ValueError, match=r"Unsupported Message\.role.*'tool'") as exc:
        GoogleRequest.from_llm_request(request, _GOOGLE_CONFIG)
    message = str(exc.value)
    assert "Content length: 18" in message
    assert "secret tool output" not in message


def test_payload_serializes_messages() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = GoogleProvider(config=_GOOGLE_CONFIG, client=client)

    request = _make_api_request(
        {
            "provider": "google",
            "model": "gemini-test",
            "messages": [
                Message(role="system", content="Be concise."),
                Message(role="user", content="find item"),
                Message(role="assistant", content="previous answer"),
            ],
        }
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
    adapter = GoogleProvider(config=_GOOGLE_CONFIG, client=client)

    request = _make_api_request(
        {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "reasoning": ReasoningBudget(tokens=512),
        }
    )
    adapter.generate(request)

    payload = cast(dict[str, Any], captured["payload"])
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}


def test_payload_serializes_google_budget_controls() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = GoogleProvider(config=_GOOGLE_CONFIG, client=client)

    request = _make_api_request(
        {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "reasoning": GoogleReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
        }
    )
    adapter.generate(request)

    payload = cast(dict[str, Any], captured["payload"])
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}

    request = _make_api_request(
        {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "reasoning": GoogleReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=1024,
                include_thoughts=True,
            ),
        }
    )
    adapter.generate(request)

    payload = cast(dict[str, Any], captured["payload"])
    assert payload["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 1024,
        "includeThoughts": True,
    }


def test_invalid_json_raises_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(status_code=200, text="{")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = GoogleProvider(config=_GOOGLE_CONFIG, client=client)

    request = _make_api_request(
        {
            "provider": "google",
            "model": "gemini-test",
            "messages": [Message(role="user", content="hi")],
        }
    )
    with pytest.raises(ProviderTransportError, match="invalid JSON response"):
        adapter.generate(request)


def test_response_filters_thought_parts_out_of_visible_text() -> None:
    _captured, client = make_http_client(_THOUGHT_RESPONSE)
    adapter = GoogleProvider(config=_GOOGLE_CONFIG, client=client)

    request = make_request(
        provider="google",
        model="gemma-4-31b-it",
        reasoning=GoogleReasoning(thinking_level=ThinkingLevel.HIGH),
    )
    response = adapter.generate(request)

    assert response.text == "OK"
    assert response.reasoning == "private chain of thought"
    assert response.reasoning_details == [
        {"text": "private chain of thought", "thought": True}
    ]


def test_rejects_kimi_code_request_shape() -> None:
    request = KimiCodeLlmRequest(
        provider="kimi-code",
        model="kimi-for-coding",
        messages=[Message(role="user", content="hi")],
        max_tokens=256,
        effort=EffortSpec.HIGH,
    )

    with pytest.raises(
        ProviderSemanticError, match="sampling-capable API request shape"
    ):
        GoogleRequest.from_llm_request(request, _GOOGLE_CONFIG)

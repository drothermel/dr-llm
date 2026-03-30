from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from dr_llm.errors import ProviderTransportError
from dr_llm.providers.models import Message
from dr_llm.providers.openai_compat.adapter import OpenAICompatAdapter
from dr_llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.providers.openai_compat.request import OpenAICompatRequest
from dr_llm.providers.openai_compat.response import OpenAICompatResponse
from dr_llm.providers.reasoning import ReasoningEffort
from tests.conftest import make_request
from tests.providers.conftest import make_http_client

_CONFIG = OpenAICompatConfig(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key="x",
)

_REASONING_RESPONSE_JSON: dict[str, Any] = {
    "choices": [
        {
            "finish_reason": "stop",
            "message": {
                "content": "final answer",
                "reasoning": "internal trace",
                "reasoning_details": [{"type": "reasoning.text", "text": "step 1"}],
            },
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "completion_tokens_details": {"reasoning_tokens": 22},
        "cost": 0.003,
        "prompt_cost": 0.001,
        "completion_cost": 0.002,
        "currency": "USD",
    },
}


# ---------------------------------------------------------------------------
# Adapter-level tests
# ---------------------------------------------------------------------------


def test_forwards_reasoning_and_parses_cost() -> None:
    captured, client = make_http_client(_REASONING_RESPONSE_JSON)

    with OpenAICompatAdapter(config=_CONFIG, client=client) as adapter:
        request = make_request(
            provider="openrouter",
            model="openai/gpt-5",
            reasoning=ReasoningEffort(level="high"),
        )
        result = adapter.generate(request)

    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert result.reasoning == "internal trace"
    assert result.usage.reasoning_tokens == 22
    assert result.cost is not None
    assert result.cost.total_cost_usd == 0.003


def test_invalid_json_raises_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, text="{")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(config=_CONFIG, client=client)

    with pytest.raises(ProviderTransportError, match="invalid JSON response"):
        adapter.generate(make_request(provider="openrouter", model="some-model"))


# ---------------------------------------------------------------------------
# Client lifecycle tests
# ---------------------------------------------------------------------------


def test_close_does_not_close_injected_client() -> None:
    _, client = make_http_client({})
    adapter = OpenAICompatAdapter(config=_CONFIG, client=client)
    adapter.close()
    assert not client.is_closed


def test_close_closes_adapter_owned_client() -> None:
    adapter = OpenAICompatAdapter(config=_CONFIG)
    owned_client = cast(httpx.Client, adapter._client)
    adapter.close()
    assert owned_client.is_closed


# ---------------------------------------------------------------------------
# Request unit tests
# ---------------------------------------------------------------------------


def test_request_builds_endpoint_and_headers() -> None:
    request = make_request(
        provider="openrouter",
        model="some-model",
        messages=[Message(role="user", content="hi")],
        metadata={"idempotency_key": "fixed-key"},
    )
    provider_request = OpenAICompatRequest.from_llm_request(request, _CONFIG)

    assert provider_request.endpoint() == "https://openrouter.ai/api/v1/chat/completions"
    assert provider_request.headers()["Idempotency-Key"] == "fixed-key"
    assert provider_request.messages == [{"role": "user", "content": "hi"}]


def test_request_generates_idempotency_key_when_missing() -> None:
    request = make_request(provider="openrouter", model="some-model")
    provider_request = OpenAICompatRequest.from_llm_request(request, _CONFIG)
    assert provider_request.idempotency_key


# ---------------------------------------------------------------------------
# Response unit tests
# ---------------------------------------------------------------------------


def test_response_parses_reasoning_and_cost() -> None:
    http_response = httpx.Response(status_code=200, json=_REASONING_RESPONSE_JSON)
    provider_response = OpenAICompatResponse.from_http_response(http_response)

    request = make_request(provider="openrouter", model="some-model")
    result = provider_response.to_llm_response(request, latency_ms=42, warnings=[])

    assert result.text == "final answer"
    assert result.reasoning == "internal trace"
    assert result.reasoning_details == [{"type": "reasoning.text", "text": "step 1"}]
    assert result.usage.reasoning_tokens == 22
    assert result.cost is not None
    assert result.cost.total_cost_usd == 0.003
    assert result.latency_ms == 42

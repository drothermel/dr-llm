from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from dr_llm.errors import ProviderTransportError
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.openai_compat.provider import OpenAICompatProvider
from dr_llm.llm.providers.openai_compat.config import OpenAICompatConfig
from dr_llm.llm.providers.openai_compat.request import OpenAICompatRequest
from dr_llm.llm.providers.openai_compat.response import OpenAICompatResponse
from dr_llm.llm.providers.reasoning import (
    GlmReasoning,
    OpenRouterReasoning,
    ThinkingLevel,
)
from tests.conftest import make_request
from tests.llm.providers.conftest import make_http_client

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

    with OpenAICompatProvider(config=_CONFIG, client=client) as adapter:
        request = make_request(
            provider="openrouter",
            model="openai/gpt-oss-20b",
            reasoning=OpenRouterReasoning(effort="high"),
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
    adapter = OpenAICompatProvider(config=_CONFIG, client=client)

    with pytest.raises(ProviderTransportError, match="invalid JSON response"):
        adapter.generate(
            make_request(provider="openrouter", model="deepseek/deepseek-chat")
        )


# ---------------------------------------------------------------------------
# Client lifecycle tests
# ---------------------------------------------------------------------------


def test_close_does_not_close_injected_client() -> None:
    _, client = make_http_client({})
    adapter = OpenAICompatProvider(config=_CONFIG, client=client)
    adapter.close()
    assert not client.is_closed


def test_close_closes_adapter_owned_client() -> None:
    adapter = OpenAICompatProvider(config=_CONFIG)
    owned_client = cast(httpx.Client, adapter._client)
    adapter.close()
    assert owned_client.is_closed


# ---------------------------------------------------------------------------
# Request unit tests
# ---------------------------------------------------------------------------


def test_request_builds_endpoint_and_headers() -> None:
    request = make_request(
        provider="openrouter",
        model="deepseek/deepseek-chat",
        messages=[Message(role="user", content="hi")],
        metadata={"idempotency_key": "fixed-key"},
    )
    provider_request = OpenAICompatRequest.from_llm_request(request, _CONFIG)

    assert (
        provider_request.endpoint() == "https://openrouter.ai/api/v1/chat/completions"
    )
    assert provider_request.headers()["Idempotency-Key"] == "fixed-key"
    assert provider_request.messages == [{"role": "user", "content": "hi"}]


def test_request_generates_idempotency_key_when_missing() -> None:
    request = make_request(provider="openrouter", model="deepseek/deepseek-chat")
    provider_request = OpenAICompatRequest.from_llm_request(request, _CONFIG)
    assert provider_request.idempotency_key


def test_request_omits_reasoning_when_not_configured() -> None:
    request = make_request(provider="openrouter", model="deepseek/deepseek-chat")
    provider_request = OpenAICompatRequest.from_llm_request(request, _CONFIG)
    assert provider_request.reasoning_effort is None
    assert "reasoning" not in provider_request.json_payload()


def test_glm_request_serializes_native_thinking_payload() -> None:
    glm_config = OpenAICompatConfig(
        name="glm",
        base_url="https://api.z.ai/api/coding/paas/v4",
        api_key="x",
    )
    request = make_request(
        provider="glm",
        model="glm-4.5",
        reasoning=GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    provider_request = OpenAICompatRequest.from_llm_request(request, glm_config)
    assert provider_request.reasoning_effort is None
    assert provider_request.json_payload()["thinking"] == {"type": "enabled"}


def test_request_rejects_extra_body_key_collisions() -> None:
    provider_request = OpenAICompatRequest(
        provider="openrouter",
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"model": "other-model"},
        base_url="https://openrouter.ai/api/v1",
        chat_path="/chat/completions",
        api_key_env="OPENROUTER_API_KEY",
        api_key="x",
        idempotency_key="fixed-key",
    )

    with pytest.raises(ValueError, match="extra_body conflicts with validated payload keys: model"):
        provider_request.json_payload()


# ---------------------------------------------------------------------------
# Response unit tests
# ---------------------------------------------------------------------------


def test_response_parses_reasoning_and_cost() -> None:
    http_response = httpx.Response(status_code=200, json=_REASONING_RESPONSE_JSON)
    provider_response = OpenAICompatResponse.from_http_response(http_response)

    request = make_request(provider="openrouter", model="deepseek/deepseek-chat")
    result = provider_response.to_llm_response(request, latency_ms=42, warnings=[])

    assert result.text == "final answer"
    assert result.reasoning == "internal trace"
    assert result.reasoning_details == [{"type": "reasoning.text", "text": "step 1"}]
    assert result.usage.reasoning_tokens == 22
    assert result.cost is not None
    assert result.cost.total_cost_usd == 0.003
    assert result.latency_ms == 42

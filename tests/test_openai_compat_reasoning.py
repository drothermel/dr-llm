from __future__ import annotations

import json
from typing import Any, cast

import httpx

from dr_llm.providers.openai_compat_request import OpenAICompatRequest
from dr_llm.providers.openai_compat import (
    OpenAICompatAdapter,
    OpenAICompatConfig,
)
from dr_llm.generation.models import LlmRequest, Message, ReasoningConfig


def test_openai_compat_forwards_reasoning_and_parses_reasoning_cost() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "final answer",
                            "reasoning": "internal trace",
                            "reasoning_details": [
                                {"type": "reasoning.text", "text": "step 1"}
                            ],
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
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    request = LlmRequest(
        provider="openrouter",
        model="openai/o3-mini",
        messages=[Message(role="user", content="hello")],
        reasoning=ReasoningConfig(effort="high", exclude=False),
    )

    with OpenAICompatAdapter(
        config=OpenAICompatConfig(
            name="openrouter", base_url="https://openrouter.ai/api/v1", api_key="x"
        ),
        client=client,
    ) as adapter:
        response = adapter.generate(request)

    payload = cast(dict[str, Any], captured["payload"])
    headers = cast(dict[str, str], captured["headers"])
    assert payload["reasoning"] == {"effort": "high", "exclude": False}
    assert headers["idempotency-key"]

    assert response.reasoning == "internal trace"
    assert response.reasoning_details == [{"type": "reasoning.text", "text": "step 1"}]
    assert response.usage.reasoning_tokens == 22
    assert response.cost is not None
    assert response.cost.total_cost_usd == 0.003
    assert response.cost.prompt_cost_usd == 0.001
    assert response.cost.completion_cost_usd == 0.002


def test_openai_compat_close_does_not_close_injected_client() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    adapter = OpenAICompatAdapter(
        config=OpenAICompatConfig(
            name="openrouter", base_url="https://openrouter.ai/api/v1", api_key="x"
        ),
        client=client,
    )
    assert not client.is_closed
    adapter.close()
    assert not client.is_closed
    client.close()


def test_openai_compat_close_closes_adapter_owned_client() -> None:
    adapter = OpenAICompatAdapter(
        config=OpenAICompatConfig(
            name="openrouter", base_url="https://openrouter.ai/api/v1", api_key="x"
        ),
    )
    # Intentional private access: there is no public accessor for the internally
    # created client, and this test must verify adapter-owned client shutdown.
    client = adapter._client
    assert client is not None
    assert not client.is_closed
    adapter.close()
    assert client.is_closed


def test_openai_compat_set_client_does_not_close_injected_clients() -> None:
    first = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    second = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    adapter = OpenAICompatAdapter(
        config=OpenAICompatConfig(
            name="openrouter", base_url="https://openrouter.ai/api/v1", api_key="x"
        ),
        client=first,
    )
    adapter.set_client(second)
    assert not first.is_closed
    assert not second.is_closed
    adapter.close()
    assert not second.is_closed
    first.close()
    second.close()


def test_openai_compat_request_builds_endpoint_headers_and_payload() -> None:
    provider_request = OpenAICompatRequest.from_llm_request(
        LlmRequest(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            messages=[Message(role="user", content="hi")],
            metadata={"idempotency_key": "fixed-key"},
        ),
        OpenAICompatConfig(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1/",
            api_key="x",
        ),
    )

    assert provider_request.endpoint() == "https://openrouter.ai/api/v1/chat/completions"
    assert provider_request.headers() == {
        "Authorization": "Bearer x",
        "Content-Type": "application/json",
        "Idempotency-Key": "fixed-key",
    }
    assert provider_request.json_payload() == {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_openai_compat_request_generates_idempotency_key_when_missing() -> None:
    provider_request = OpenAICompatRequest.from_llm_request(
        LlmRequest(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            messages=[Message(role="user", content="hi")],
        ),
        OpenAICompatConfig(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="x",
        ),
    )

    assert provider_request.idempotency_key

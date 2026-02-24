from __future__ import annotations

import json
from typing import Any, cast

import httpx

from llm_pool.providers.openai_compat import (
    OpenAICompatAdapter,
    OpenAICompatConfig,
    _OpenAICompatRequestPayload,
)
from llm_pool.types import (
    LlmRequest,
    Message,
    ProviderToolSpec,
    ReasoningConfig,
    ToolFunctionSpec,
)


def test_openai_compat_forwards_reasoning_and_parses_reasoning_cost() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
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
        name="openrouter",
        config=OpenAICompatConfig(base_url="https://openrouter.ai/api/v1", api_key="x"),
        client=client,
    ) as adapter:
        response = adapter.generate(request)

    payload = cast(dict[str, Any], captured["payload"])
    assert payload["reasoning"] == {"effort": "high", "exclude": False}

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
        name="openrouter",
        config=OpenAICompatConfig(base_url="https://openrouter.ai/api/v1", api_key="x"),
        client=client,
    )
    assert not client.is_closed
    adapter.close()
    assert not client.is_closed
    client.close()


def test_openai_compat_close_closes_adapter_owned_client() -> None:
    adapter = OpenAICompatAdapter(
        name="openrouter",
        config=OpenAICompatConfig(base_url="https://openrouter.ai/api/v1", api_key="x"),
    )
    client = adapter._client
    assert client is not None
    assert not client.is_closed
    adapter.close()
    assert client.is_closed


def test_openai_compat_set_client_closes_previous_client() -> None:
    first = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    second = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    adapter = OpenAICompatAdapter(
        name="openrouter",
        config=OpenAICompatConfig(base_url="https://openrouter.ai/api/v1", api_key="x"),
        client=first,
    )
    adapter.set_client(second)
    assert not first.is_closed
    assert not second.is_closed
    adapter.close()
    assert not second.is_closed
    first.close()
    second.close()


def test_openai_compat_request_payload_serializes_tools() -> None:
    tool = ProviderToolSpec(
        function=ToolFunctionSpec(
            name="lookup",
            description="Lookup a value",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
    )
    payload = _OpenAICompatRequestPayload(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tools=[tool],
    ).model_dump(mode="json", exclude_none=True)

    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup a value",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }
    ]

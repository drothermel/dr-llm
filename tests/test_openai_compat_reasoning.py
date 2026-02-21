from __future__ import annotations

import json

import httpx

from llm_pool.providers.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from llm_pool.types import LlmRequest, Message, ReasoningConfig


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
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(
        name="openrouter",
        config=OpenAICompatConfig(base_url="https://openrouter.ai/api/v1", api_key="x"),
        client=client,
    )

    request = LlmRequest(
        provider="openrouter",
        model="openai/o3-mini",
        messages=[Message(role="user", content="hello")],
        reasoning=ReasoningConfig(effort="high", exclude=False),
    )

    response = adapter.generate(request)

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["reasoning"] == {"effort": "high", "exclude": False}

    assert response.reasoning == "internal trace"
    assert response.reasoning_details == [{"type": "reasoning.text", "text": "step 1"}]
    assert response.usage.reasoning_tokens == 22
    assert response.cost is not None
    assert response.cost.total_cost_usd == 0.003
    assert response.cost.prompt_cost_usd == 0.001
    assert response.cost.completion_cost_usd == 0.002

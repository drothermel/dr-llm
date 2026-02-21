from __future__ import annotations

import json
from typing import Any, cast

import httpx

from llm_pool.providers.anthropic import AnthropicAdapter, AnthropicConfig
from llm_pool.providers.google import GoogleAdapter, GoogleConfig
from llm_pool.types import LlmRequest, Message, ModelToolCall


def test_anthropic_payload_preserves_tool_context() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": "end_turn",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = AnthropicAdapter(
        config=AnthropicConfig(
            api_key="x", base_url="https://api.anthropic.com/v1/messages"
        ),
        client=client,
    )

    request = LlmRequest(
        provider="anthropic",
        model="claude-test",
        messages=[
            Message(role="user", content="find item"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ModelToolCall(
                        tool_call_id="tc_1", name="lookup", arguments={"q": "abc"}
                    )
                ],
            ),
            Message(
                role="tool",
                name="lookup",
                tool_call_id="tc_1",
                content='{"result": 123}',
            ),
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
    )

    response = adapter.generate(request)
    assert response.text == "done"

    payload = cast(dict[str, Any], captured["payload"])
    messages = cast(list[dict[str, Any]], payload["messages"])
    assert any(
        message["role"] == "assistant"
        and any(
            block.get("type") == "tool_use" and block.get("id") == "tc_1"
            for block in message["content"]
        )
        for message in messages
    )
    assert any(
        message["role"] == "user"
        and any(
            block.get("type") == "tool_result" and block.get("tool_use_id") == "tc_1"
            for block in message["content"]
        )
        for message in messages
    )
    tools = cast(list[dict[str, Any]], payload["tools"])
    assert tools[0]["name"] == "lookup"


def test_google_payload_preserves_tool_context() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "done"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 3,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = GoogleAdapter(
        config=GoogleConfig(
            api_key="x", base_url="https://generativelanguage.googleapis.com/v1beta"
        ),
        client=client,
    )

    request = LlmRequest(
        provider="google",
        model="gemini-test",
        messages=[
            Message(role="user", content="find item"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ModelToolCall(
                        tool_call_id="tc_1", name="lookup", arguments={"q": "abc"}
                    )
                ],
            ),
            Message(
                role="tool",
                name="lookup",
                tool_call_id="tc_1",
                content='{"result": 123}',
            ),
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
    )

    response = adapter.generate(request)
    assert response.text == "done"

    payload = cast(dict[str, Any], captured["payload"])
    contents = cast(list[dict[str, Any]], payload["contents"])
    assert any(
        content["role"] == "model"
        and any(
            part.get("functionCall", {}).get("name") == "lookup"
            for part in content["parts"]
        )
        for content in contents
    )
    assert any(
        content["role"] == "user"
        and any(
            part.get("functionResponse", {}).get("name") == "lookup"
            for part in content["parts"]
        )
        for content in contents
    )
    tools = cast(list[dict[str, Any]], payload["tools"])
    declarations = cast(list[dict[str, Any]], tools[0]["functionDeclarations"])
    assert declarations[0]["name"] == "lookup"

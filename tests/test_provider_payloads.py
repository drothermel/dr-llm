from __future__ import annotations

import json
import subprocess
from typing import Any, cast

import httpx

from llm_pool.providers.anthropic import AnthropicAdapter, AnthropicConfig
from llm_pool.providers.google import GoogleAdapter, GoogleConfig
from llm_pool.providers.headless import CodexHeadlessAdapter
from llm_pool.types import (
    LlmRequest,
    Message,
    ModelToolCall,
    ProviderToolSpec,
    ToolFunctionSpec,
)


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
            ProviderToolSpec(
                function=ToolFunctionSpec(
                    name="lookup",
                    description="Lookup a value",
                    parameters={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                )
            )
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
            ProviderToolSpec(
                function=ToolFunctionSpec(
                    name="lookup",
                    description="Lookup a value",
                    parameters={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                )
            )
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


def test_google_tool_call_ids_are_sequential_for_valid_function_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "thinking"},
                                {"functionCall": {"name": "", "args": {}}},
                                {
                                    "functionCall": {
                                        "name": "lookup",
                                        "args": {"q": "x"},
                                    }
                                },
                                {"text": "more text"},
                                {
                                    "functionCall": {
                                        "name": "search",
                                        "args": {"q": "y"},
                                    }
                                },
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
            },
        )

    adapter = GoogleAdapter(
        config=GoogleConfig(
            api_key="x", base_url="https://generativelanguage.googleapis.com/v1beta"
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = adapter.generate(
        LlmRequest(
            provider="google",
            model="gemini-test",
            messages=[Message(role="user", content="go")],
        )
    )

    assert [call.tool_call_id for call in response.tool_calls] == [
        "google_call_1",
        "google_call_2",
    ]
    assert [call.name for call in response.tool_calls] == ["lookup", "search"]


def test_headless_tool_call_ids_are_sequential_for_valid_items(monkeypatch) -> None:  # noqa: ANN001
    body = {
        "text": "ok",
        "tool_calls": [
            "not-a-dict",
            {"arguments": {"q": "ignored"}},
            {"name": "lookup", "arguments": {"q": "x"}},
            {"name": "search", "arguments": {"q": "y"}},
        ],
    }

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return subprocess.CompletedProcess(
            args=kwargs.get("args") or [],
            returncode=0,
            stdout=json.dumps(body, ensure_ascii=True),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = CodexHeadlessAdapter(command=["codex", "--headless", "--json"])
    response = adapter.generate(
        LlmRequest(
            provider="codex",
            model="codex-test",
            messages=[Message(role="user", content="go")],
        )
    )

    assert [call.tool_call_id for call in response.tool_calls] == ["call_1", "call_2"]
    assert [call.name for call in response.tool_calls] == ["lookup", "search"]

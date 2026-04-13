from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.llm.providers.anthropic.provider import AnthropicProvider
from dr_llm.llm.providers.anthropic.config import AnthropicConfig
from dr_llm.llm.providers.anthropic.request import AnthropicRequest
from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import AnthropicReasoning, ThinkingLevel
from dr_llm.llm.request import ApiLlmRequest, KimiCodeLlmRequest
from tests.conftest import make_request
from tests.llm.providers.conftest import make_http_client

_MOCK_RESPONSE: dict[str, Any] = {
    "content": [{"type": "text", "text": "done"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "stop_reason": "end_turn",
}


def _make_config() -> AnthropicConfig:
    return AnthropicConfig(
        api_key="x", base_url="https://api.anthropic.com/v1/messages"
    )


def _make_api_request(**overrides: Any) -> ApiLlmRequest | KimiCodeLlmRequest:
    return cast(ApiLlmRequest | KimiCodeLlmRequest, make_request(**overrides))


def test_payload_serializes_messages() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = AnthropicProvider(config=_make_config(), client=client)

    request = _make_api_request(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        max_tokens=256,
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


def test_missing_max_tokens_error_uses_actual_provider_name() -> None:
    request = _make_api_request(
        provider="kimi-code",
        model="kimi-k2-instruct",
        max_tokens=256,
    )
    request = request.model_copy(update={"max_tokens": None})

    with pytest.raises(
        ProviderSemanticError, match="kimi-code requests require max_tokens"
    ):
        AnthropicRequest.from_llm_request(request, _make_config())


def test_payload_serializes_effort_output_config() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = AnthropicProvider(config=_make_config(), client=client)

    request = _make_api_request(
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_tokens=256,
        effort=EffortSpec.MEDIUM,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
    )
    adapter.generate(request)

    payload = captured["payload"]
    assert payload["output_config"] == {"effort": "medium"}


def test_payload_serializes_manual_thinking() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = AnthropicProvider(config=_make_config(), client=client)

    request = _make_api_request(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        reasoning=AnthropicReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=2048,
            display="omitted",
        ),
    )
    adapter.generate(request)

    payload = captured["payload"]
    assert payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
        "display": "omitted",
    }


def test_payload_omits_thinking_for_off() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = AnthropicProvider(config=_make_config(), client=client)

    request = make_request(
        provider="anthropic",
        model="claude-sonnet-4-6",
        max_tokens=256,
        effort=EffortSpec.MEDIUM,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
    )
    adapter.generate(request)

    payload = captured["payload"]
    assert "thinking" not in payload


def test_invalid_json_raises_transport_error() -> None:
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code=200, text="{")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = AnthropicProvider(config=_make_config(), client=client)

    with pytest.raises(ProviderTransportError, match="invalid JSON response"):
        adapter.generate(
            _make_api_request(
                provider="anthropic",
                model="claude-3-5-haiku-20241022",
                max_tokens=256,
            )
        )
    assert call_count == 1


def test_transport_failure_retries_raw_http_send_only_once_before_success() -> None:
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(status_code=200, json=_MOCK_RESPONSE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = AnthropicProvider(config=_make_config(), client=client)

    result = adapter.generate(
        _make_api_request(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
        )
    )

    assert result.text == "done"
    assert call_count == 2

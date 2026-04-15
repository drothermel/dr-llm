from __future__ import annotations

from typing import Any

from dr_llm.llm.providers.anthropic.config import AnthropicConfig
from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.minimax import (
    MINIMAX_BASE_URL,
    MINIMAX_PROVIDER_NAME,
    MiniMaxProvider,
)
from dr_llm.llm.providers.reasoning import AnthropicReasoning, ThinkingLevel
from tests.conftest import make_request
from tests.llm.providers.conftest import make_http_client

_MOCK_RESPONSE: dict[str, Any] = {
    "content": [
        {"type": "thinking", "thinking": "hidden"},
        {"type": "text", "text": "done"},
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "stop_reason": "end_turn",
}


def _minimax_test_config() -> AnthropicConfig:
    return AnthropicConfig(
        name=MINIMAX_PROVIDER_NAME,
        base_url=MINIMAX_BASE_URL,
        api_key="test-key",
    )


def test_minimax_serializes_effort_without_thinking_payload() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = MiniMaxProvider(config=_minimax_test_config(), client=client)

    request = make_request(
        provider=MINIMAX_PROVIDER_NAME,
        model="MiniMax-M2.7",
        effort=EffortSpec.HIGH,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.NA),
    )
    result = adapter.generate(request)

    assert result.text == "done"
    assert captured["url"] == MINIMAX_BASE_URL
    assert captured["payload"]["output_config"] == {"effort": "high"}
    assert "thinking" not in captured["payload"]
    assert "max_tokens" not in captured["payload"]


def test_minimax_serializes_optional_max_tokens_when_present() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = MiniMaxProvider(config=_minimax_test_config(), client=client)

    request = make_request(
        provider=MINIMAX_PROVIDER_NAME,
        model="MiniMax-M2.5",
        max_tokens=2048,
        effort=EffortSpec.MAX,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.NA),
    )
    adapter.generate(request)

    assert captured["payload"]["max_tokens"] == 2048
    assert captured["payload"]["output_config"] == {"effort": "max"}

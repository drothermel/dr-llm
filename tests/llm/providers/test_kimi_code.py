from __future__ import annotations

from typing import Any

from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.kimi_code import (
    KIMI_CODE_BASE_URL,
    KIMI_CODE_PROVIDER_NAME,
    KimiCodeProvider,
)
from dr_llm.llm.providers.reasoning import AnthropicReasoning, ThinkingLevel
from tests.conftest import make_request
from tests.llm.providers.conftest import make_http_client

_MOCK_RESPONSE: dict[str, Any] = {
    "content": [{"type": "text", "text": "done"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "stop_reason": "end_turn",
}


def test_kimi_code_serializes_effort_and_adaptive_thinking() -> None:
    captured, client = make_http_client(_MOCK_RESPONSE)
    adapter = KimiCodeProvider(client=client)

    request = make_request(
        provider=KIMI_CODE_PROVIDER_NAME,
        model="kimi-for-coding",
        max_tokens=256,
        effort=EffortSpec.HIGH,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE),
    )
    result = adapter.generate(request)

    assert result.text == "done"
    assert captured["url"] == KIMI_CODE_BASE_URL
    assert captured["payload"]["output_config"] == {"effort": "high"}
    assert captured["payload"]["thinking"] == {"type": "adaptive"}


def test_kimi_code_serializes_budget_and_disabled_thinking() -> None:
    budget_captured, budget_client = make_http_client(_MOCK_RESPONSE)
    budget_adapter = KimiCodeProvider(client=budget_client)
    budget_request = make_request(
        provider=KIMI_CODE_PROVIDER_NAME,
        model="kimi-for-coding",
        max_tokens=2048,
        effort=EffortSpec.MAX,
        reasoning=AnthropicReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=1024,
        ),
    )
    budget_adapter.generate(budget_request)
    assert budget_captured["payload"]["output_config"] == {"effort": "max"}
    assert budget_captured["payload"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }

    off_captured, off_client = make_http_client(_MOCK_RESPONSE)
    off_adapter = KimiCodeProvider(client=off_client)
    off_request = make_request(
        provider=KIMI_CODE_PROVIDER_NAME,
        model="kimi-for-coding",
        max_tokens=256,
        effort=EffortSpec.LOW,
        reasoning=AnthropicReasoning(thinking_level=ThinkingLevel.OFF),
    )
    off_adapter.generate(off_request)
    assert off_captured["payload"]["output_config"] == {"effort": "low"}
    assert off_captured["payload"]["thinking"] == {"type": "disabled"}

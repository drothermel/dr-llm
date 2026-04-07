from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dr_llm.pool.pending.models import PendingSample
from dr_llm.pool.pending.workers import make_llm_process_fn
from dr_llm.providers.llm_config import LlmConfig
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.models import CallMode, Message
from dr_llm.providers.usage import TokenUsage


def _make_sample(payload: dict[str, Any]) -> PendingSample:
    return PendingSample(
        key_values={"llm_config": "cfg1", "prompt": "p1"},
        payload=payload,
    )


def _make_response() -> LlmResponse:
    return LlmResponse(
        text="hello world",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        provider="openai",
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )


def _make_registry(response: LlmResponse | None = None) -> MagicMock:
    resp = response or _make_response()
    adapter = MagicMock()
    adapter.generate.return_value = resp
    registry = MagicMock()
    registry.get.return_value = adapter
    return registry


def _sample_payload() -> dict[str, Any]:
    config = LlmConfig(provider="openai", model="gpt-4.1-mini", temperature=0.5)
    messages = [Message(role="user", content="Say hello")]
    return {
        "llm_config": config.model_dump(),
        "prompt": [m.model_dump() for m in messages],
    }


def test_dispatches_via_registry() -> None:
    registry = _make_registry()
    process_fn = make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())

    result = process_fn(sample)

    registry.get.assert_called_once_with("openai")
    adapter = registry.get.return_value
    call_args = adapter.generate.call_args[0][0]
    assert call_args.provider == "openai"
    assert call_args.model == "gpt-4.1-mini"
    assert call_args.temperature == 0.5
    assert len(call_args.messages) == 1
    assert call_args.messages[0].content == "Say hello"
    assert result["text"] == "hello world"
    assert result["provider"] == "openai"


def test_returns_full_response_dump() -> None:
    response = _make_response()
    registry = _make_registry(response)
    process_fn = make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())

    result = process_fn(sample)

    expected = response.model_dump()
    assert result == expected


def test_records_call_when_recorder_provided() -> None:
    registry = _make_registry()
    recorder = MagicMock()
    recorder.record_call.return_value = "call-123"
    process_fn = make_llm_process_fn(
        registry, recorder=recorder, run_id="run-abc"
    )
    sample = _make_sample(_sample_payload())

    result = process_fn(sample)

    recorder.record_call.assert_called_once()
    call_kwargs = recorder.record_call.call_args[1]
    assert call_kwargs["run_id"] == "run-abc"
    assert call_kwargs["request"].provider == "openai"
    assert call_kwargs["response"].text == "hello world"
    assert result["call_id"] == "call-123"


def test_no_call_id_without_recorder() -> None:
    registry = _make_registry()
    process_fn = make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())

    result = process_fn(sample)

    assert "call_id" not in result


def test_error_propagates() -> None:
    adapter = MagicMock()
    adapter.generate.side_effect = RuntimeError("API down")
    registry = MagicMock()
    registry.get.return_value = adapter
    process_fn = make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())

    with pytest.raises(RuntimeError, match="API down"):
        process_fn(sample)


def test_missing_llm_config_key_raises() -> None:
    registry = _make_registry()
    process_fn = make_llm_process_fn(registry)
    sample = _make_sample({"prompt": [{"role": "user", "content": "hi"}]})

    with pytest.raises(KeyError, match="llm_config"):
        process_fn(sample)


def test_missing_prompt_key_raises() -> None:
    registry = _make_registry()
    process_fn = make_llm_process_fn(registry)
    config = LlmConfig(provider="openai", model="gpt-4.1-mini")
    sample = _make_sample({"llm_config": config.model_dump()})

    with pytest.raises(KeyError, match="prompt"):
        process_fn(sample)


def test_custom_key_names() -> None:
    registry = _make_registry()
    config = LlmConfig(provider="openai", model="gpt-4.1-mini")
    messages = [Message(role="user", content="custom")]
    process_fn = make_llm_process_fn(
        registry, llm_config_key="model_cfg", prompt_key="msgs"
    )
    sample = _make_sample({
        "model_cfg": config.model_dump(),
        "msgs": [m.model_dump() for m in messages],
    })

    result = process_fn(sample)

    assert result["text"] == "hello world"

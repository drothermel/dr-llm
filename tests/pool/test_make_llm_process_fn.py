from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dr_llm.logging.events import generation_log_context
from dr_llm.llm.config import LlmConfig
from dr_llm.llm.messages import CallMode, Message
from dr_llm.llm.providers.usage import TokenUsage
from dr_llm.llm.response import LlmResponse
from dr_llm.pool import llm_pool_adapter
from dr_llm.pool.pending.pending_sample import PendingSample


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
    adapter.mode = resp.mode
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
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
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
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())

    result = process_fn(sample)

    expected = response.model_dump()
    assert result == expected


def test_process_result_has_no_call_id() -> None:
    registry = _make_registry()
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())

    result = process_fn(sample)

    assert "call_id" not in result


def test_emits_worker_logging_events(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _make_registry()
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())
    events: list[dict[str, Any]] = []

    monkeypatch.setattr(
        llm_pool_adapter,
        "emit_generation_event",
        lambda *, event_type, stage, payload: events.append(
            {"event_type": event_type, "stage": stage, "payload": payload}
        ),
    )

    with generation_log_context({"pool_name": "demo", "worker_id": "worker-1"}):
        result = process_fn(sample)

    assert result["text"] == "hello world"
    assert [event["event_type"] for event in events] == [
        "llm_call.started",
        "llm_call.succeeded",
    ]
    for event in events:
        assert event["payload"]["pool_name"] == "demo"
        assert event["payload"]["worker_id"] == "worker-1"
        assert event["payload"]["pending_id"] == sample.pending_id
        assert event["payload"]["sample_idx"] == sample.sample_idx
        assert event["payload"]["key_values"] == sample.key_values


def test_failed_worker_call_emits_failure_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MagicMock()
    adapter.generate.side_effect = RuntimeError("API down")
    adapter.mode = CallMode.api
    registry = MagicMock()
    registry.get.return_value = adapter
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())
    events: list[dict[str, Any]] = []

    monkeypatch.setattr(
        llm_pool_adapter,
        "emit_generation_event",
        lambda *, event_type, stage, payload: events.append(
            {"event_type": event_type, "stage": stage, "payload": payload}
        ),
    )

    with (
        generation_log_context({"pool_name": "demo", "worker_id": "worker-1"}),
        pytest.raises(RuntimeError, match="API down"),
    ):
        process_fn(sample)

    assert [event["event_type"] for event in events] == [
        "llm_call.started",
        "llm_call.failed",
    ]
    assert events[-1]["payload"]["message"] == "API down"


def test_error_propagates() -> None:
    adapter = MagicMock()
    adapter.generate.side_effect = RuntimeError("API down")
    adapter.mode = CallMode.api
    registry = MagicMock()
    registry.get.return_value = adapter
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    sample = _make_sample(_sample_payload())

    with pytest.raises(RuntimeError, match="API down"):
        process_fn(sample)


def test_missing_llm_config_key_raises() -> None:
    registry = _make_registry()
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    sample = _make_sample({"prompt": [{"role": "user", "content": "hi"}]})

    with pytest.raises(KeyError, match="llm_config"):
        process_fn(sample)


def test_missing_prompt_key_raises() -> None:
    registry = _make_registry()
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    config = LlmConfig(provider="openai", model="gpt-4.1-mini")
    sample = _make_sample({"llm_config": config.model_dump()})

    with pytest.raises(KeyError, match="prompt"):
        process_fn(sample)


def test_custom_key_names() -> None:
    registry = _make_registry()
    config = LlmConfig(provider="openai", model="gpt-4.1-mini")
    messages = [Message(role="user", content="custom")]
    process_fn = llm_pool_adapter.make_llm_process_fn(
        registry, llm_config_key="model_cfg", prompt_key="msgs"
    )
    sample = _make_sample(
        {
            "model_cfg": config.model_dump(),
            "msgs": [m.model_dump() for m in messages],
        }
    )

    result = process_fn(sample)

    assert result["text"] == "hello world"

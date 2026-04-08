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
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.models import InsertResult
from dr_llm.pool.pending.grid import Axis, AxisMember, GridCell
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


def test_explicit_none_payload_field_raises_value_error() -> None:
    registry = _make_registry()
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    sample = _make_sample({"llm_config": None, "prompt": [{"role": "user", "content": "hi"}]})

    with pytest.raises(ValueError, match=r"PendingSample\.payload\['llm_config'\]"):
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


def _make_seed_store(schema: PoolSchema) -> tuple[MagicMock, list[PendingSample]]:
    """Stub PoolStore that captures the rows seed_llm_grid would insert."""
    captured: list[PendingSample] = []

    def _insert_many(
        samples: list[PendingSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        captured.extend(samples)
        return InsertResult(inserted=len(samples), skipped=0)

    store = MagicMock()
    store.schema = schema
    store.pending.insert_many.side_effect = _insert_many
    store.metadata.upsert.side_effect = lambda key, value: None
    return store, captured


def test_seed_llm_grid_round_trips_with_make_llm_process_fn() -> None:
    """Rows produced by seed_llm_grid must be consumable by make_llm_process_fn."""
    schema = PoolSchema.from_axis_names("rt", ["llm_config", "prompt"])
    store, captured = _make_seed_store(schema)

    cfg = LlmConfig(provider="openai", model="gpt-4.1-mini", temperature=0.5)
    msgs = [Message(role="user", content="round trip")]

    def _build_request(cell: GridCell) -> tuple[list[Message], LlmConfig]:
        assert cell.values["llm_config"] == cfg
        assert cell.values["prompt"] == msgs
        return msgs, cfg

    result = llm_pool_adapter.seed_llm_grid(
        store,
        axes=[
            Axis(
                name="llm_config",
                members=[AxisMember[LlmConfig](id="cfg1", value=cfg)],
            ),
            Axis(
                name="prompt",
                members=[AxisMember[list[Message]](id="p1", value=msgs)],
            ),
        ],
        build_request=_build_request,
        n=1,
    )

    assert result.inserted == 1
    assert len(captured) == 1
    seeded = captured[0]
    assert seeded.key_values == {"llm_config": "cfg1", "prompt": "p1"}
    assert "llm_config" in seeded.payload
    assert "prompt" in seeded.payload

    # And now: feed the seeded payload through make_llm_process_fn end-to-end.
    registry = _make_registry()
    process_fn = llm_pool_adapter.make_llm_process_fn(registry)
    response = process_fn(seeded)

    assert response["text"] == "hello world"
    call_args = registry.get.return_value.generate.call_args[0][0]
    assert call_args.provider == "openai"
    assert call_args.model == "gpt-4.1-mini"
    assert call_args.temperature == 0.5
    assert call_args.messages[0].content == "round trip"


def test_seed_llm_grid_honors_custom_payload_keys() -> None:
    schema = PoolSchema.from_axis_names("rt2", ["llm_config", "prompt"])
    store, captured = _make_seed_store(schema)

    cfg = LlmConfig(provider="openai", model="gpt-4.1-mini")
    msgs = [Message(role="user", content="hi")]

    llm_pool_adapter.seed_llm_grid(
        store,
        axes=[
            Axis(
                name="llm_config",
                members=[AxisMember[LlmConfig](id="cfg1", value=cfg)],
            ),
            Axis(
                name="prompt",
                members=[AxisMember[list[Message]](id="p1", value=msgs)],
            ),
        ],
        build_request=lambda _cell: (msgs, cfg),
        llm_config_key="model_cfg",
        prompt_key="msgs",
    )

    seeded = captured[0]
    assert "model_cfg" in seeded.payload
    assert "msgs" in seeded.payload
    assert "llm_config" not in seeded.payload
    assert "prompt" not in seeded.payload

    # The custom-keyed payload must round-trip through a matching process_fn.
    registry = _make_registry()
    process_fn = llm_pool_adapter.make_llm_process_fn(
        registry, llm_config_key="model_cfg", prompt_key="msgs"
    )
    response = process_fn(seeded)
    assert response["text"] == "hello world"

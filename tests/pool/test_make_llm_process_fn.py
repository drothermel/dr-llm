from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from dr_llm.logging.events import generation_log_context
from dr_llm.llm import (
    CallMode,
    CodexReasoning,
    LlmConfig,
    LlmRequest,
    LlmResponse,
    Message,
    MessageRole,
    ProviderName,
    SamplingControls,
    ThinkingLevel,
    TokenUsage,
    parse_llm_request,
)
from dr_llm.pool import backend as pool_backend
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.seed_grid import Axis, AxisMember, GridCell, seed_llm_grid
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.insert_result import InsertResult


def _make_sample(request: dict[str, Any]) -> PoolSample:
    return PoolSample(
        sample_id="sample-1",
        key_values={"llm_config": "cfg1", "prompt": "p1"},
        request=request,
    )


def _make_response() -> LlmResponse:
    return LlmResponse(
        text="hello world",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )


def _make_headless_response() -> LlmResponse:
    return LlmResponse(
        text="hello from codex",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        provider=ProviderName.CODEX,
        model="gpt-5.4-mini",
        mode=CallMode.headless,
    )


def _make_registry(response: LlmResponse | None = None) -> MagicMock:
    resp = response or _make_response()
    orchestrator = MagicMock()
    orchestrator.name = resp.provider
    orchestrator.build_request_from_config.side_effect = (
        _build_request_from_config_for_response(resp)
    )
    orchestrator.generate.return_value = resp
    orchestrator.mode = resp.mode
    registry = MagicMock()
    registry.get.return_value = orchestrator
    return registry


def _build_request_from_config_for_response(
    resp: LlmResponse,
) -> Callable[..., LlmRequest]:
    def _build_request_from_config(
        *,
        config: LlmConfig,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> LlmRequest:
        payload: dict[str, Any] = {
            "provider": resp.provider,
            "model": config.model,
            "mode": resp.mode,
            "messages": messages,
            "effort": config.effort,
            "reasoning": config.reasoning,
            "sampling": config.sampling,
            "metadata": metadata or {},
        }
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        return parse_llm_request(payload)

    return _build_request_from_config


def _sample_request() -> dict[str, Any]:
    config = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        sampling=SamplingControls(temperature=0.5),
    )
    messages = [Message(role=MessageRole.USER, content="Say hello")]
    return {
        "llm_config": config.model_dump(),
        "prompt": [m.model_dump() for m in messages],
    }


def test_dispatches_via_registry() -> None:
    registry = _make_registry()
    process_fn = pool_backend.make_llm_process_fn(registry)
    sample = _make_sample(_sample_request())

    result = process_fn(sample)

    registry.get.assert_called_once_with(ProviderName.OPENAI)
    orchestrator = registry.get.return_value
    call_args = orchestrator.generate.call_args[0][0]
    assert call_args.provider == ProviderName.OPENAI
    assert call_args.model == "gpt-4.1-mini"
    assert call_args.sampling_temperature == 0.5
    assert len(call_args.messages) == 1
    assert call_args.messages[0].content == "Say hello"
    assert result.text == "hello world"
    assert result.provider == ProviderName.OPENAI


def test_returns_llm_response_object() -> None:
    response = _make_response()
    registry = _make_registry(response)
    process_fn = pool_backend.make_llm_process_fn(registry)
    sample = _make_sample(_sample_request())

    result = process_fn(sample)

    assert result == response


def test_emits_worker_logging_events(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _make_registry()
    process_fn = pool_backend.make_llm_process_fn(registry)
    sample = _make_sample(_sample_request())
    events: list[dict[str, Any]] = []

    monkeypatch.setattr(
        pool_backend,
        "emit_generation_event",
        lambda *, event_type, stage, payload: events.append(
            {"event_type": event_type, "stage": stage, "payload": payload}
        ),
    )

    with generation_log_context(
        {"pool_name": "demo", "worker_id": "worker-1"}
    ):
        result = process_fn(sample)

    assert result.text == "hello world"
    assert [event["event_type"] for event in events] == [
        "llm_call.started",
        "llm_call.succeeded",
    ]
    for event in events:
        assert event["payload"]["pool_name"] == "demo"
        assert event["payload"]["worker_id"] == "worker-1"
        assert event["payload"]["sample_id"] == sample.sample_id
        assert event["payload"]["sample_idx"] == sample.sample_idx
        assert event["payload"]["key_values"] == sample.key_values


def test_failed_worker_call_emits_failure_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = MagicMock()
    orchestrator.name = ProviderName.OPENAI
    orchestrator.build_request_from_config.side_effect = (
        _build_request_from_config_for_response(_make_response())
    )
    orchestrator.generate.side_effect = RuntimeError("API down")
    orchestrator.mode = CallMode.api
    registry = MagicMock()
    registry.get.return_value = orchestrator
    process_fn = pool_backend.make_llm_process_fn(registry)
    sample = _make_sample(_sample_request())
    events: list[dict[str, Any]] = []

    monkeypatch.setattr(
        pool_backend,
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
    orchestrator = MagicMock()
    orchestrator.name = ProviderName.OPENAI
    orchestrator.build_request_from_config.side_effect = (
        _build_request_from_config_for_response(_make_response())
    )
    orchestrator.generate.side_effect = RuntimeError("API down")
    orchestrator.mode = CallMode.api
    registry = MagicMock()
    registry.get.return_value = orchestrator
    process_fn = pool_backend.make_llm_process_fn(registry)
    sample = _make_sample(_sample_request())

    with pytest.raises(RuntimeError, match="API down"):
        process_fn(sample)


def test_missing_llm_config_key_raises() -> None:
    registry = _make_registry()
    process_fn = pool_backend.make_llm_process_fn(registry)
    sample = _make_sample({"prompt": [{"role": "user", "content": "hi"}]})

    with pytest.raises(KeyError, match="llm_config"):
        process_fn(sample)


def test_missing_prompt_key_raises() -> None:
    registry = _make_registry()
    process_fn = pool_backend.make_llm_process_fn(registry)
    config = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )
    sample = _make_sample({"llm_config": config.model_dump()})

    with pytest.raises(KeyError, match="prompt"):
        process_fn(sample)


def test_explicit_none_request_field_raises_value_error() -> None:
    registry = _make_registry()
    process_fn = pool_backend.make_llm_process_fn(registry)
    sample = _make_sample(
        {"llm_config": None, "prompt": [{"role": "user", "content": "hi"}]}
    )

    with pytest.raises(
        ValueError, match=r"PoolSample\.request\['llm_config'\]"
    ):
        process_fn(sample)


def test_custom_key_names() -> None:
    registry = _make_registry()
    config = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )
    messages = [Message(role=MessageRole.USER, content="custom")]
    process_fn = pool_backend.make_llm_process_fn(
        registry, llm_config_key="model_cfg", prompt_key="msgs"
    )
    sample = _make_sample(
        {
            "model_cfg": config.model_dump(),
            "msgs": [m.model_dump() for m in messages],
        }
    )

    result = process_fn(sample)

    assert result.text == "hello world"


def _make_seed_store(schema: PoolSchema) -> tuple[MagicMock, list[PoolSample]]:
    """Stub PoolStore that captures the rows seed_llm_grid would insert."""
    captured: list[PoolSample] = []

    def _insert_samples(
        samples: list[PoolSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        assert ignore_conflicts is True
        captured.extend(samples)
        return InsertResult(inserted=len(samples), skipped=0)

    store = MagicMock()
    store.schema = schema
    store.insert_samples.side_effect = _insert_samples
    return store, captured


def test_seed_llm_grid_round_trips_with_make_llm_process_fn() -> None:
    """Rows produced by seed_llm_grid must be consumable by make_llm_process_fn."""
    schema = PoolSchema.from_axis_names("rt", ["llm_config", "prompt"])
    store, captured = _make_seed_store(schema)

    cfg = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
        sampling=SamplingControls(temperature=0.5),
    )
    msgs = [Message(role=MessageRole.USER, content="round trip")]

    def _build_request(cell: GridCell) -> tuple[list[Message], LlmConfig]:
        assert cell.values["llm_config"] == cfg
        assert cell.values["prompt"] == msgs
        return msgs, cfg

    result = seed_llm_grid(
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
    assert "llm_config" in seeded.request
    assert "prompt" in seeded.request

    registry = _make_registry()
    process_fn = pool_backend.make_llm_process_fn(registry)
    response = process_fn(seeded)

    assert response.text == "hello world"
    call_args = registry.get.return_value.generate.call_args[0][0]
    assert call_args.provider == ProviderName.OPENAI
    assert call_args.model == "gpt-4.1-mini"
    assert call_args.sampling_temperature == 0.5
    assert call_args.messages[0].content == "round trip"


def test_seed_llm_grid_round_trips_headless_config() -> None:
    """Headless LLM configs should seed and deserialize into headless requests."""
    schema = PoolSchema.from_axis_names(
        "rt_headless", ["llm_config", "prompt"]
    )
    store, captured = _make_seed_store(schema)

    cfg = LlmConfig(
        provider=ProviderName.CODEX,
        model="gpt-5.4-mini",
        mode=CallMode.headless,
        reasoning=CodexReasoning(thinking_level=ThinkingLevel.XHIGH),
    )
    msgs = [Message(role=MessageRole.USER, content="headless round trip")]

    def _build_request(
        cell: GridCell,
    ) -> tuple[list[Message], LlmConfig]:
        assert cell.values["llm_config"] == cfg
        assert cell.values["prompt"] == msgs
        return msgs, cfg

    result = seed_llm_grid(
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
    seeded = captured[0]

    registry = _make_registry(_make_headless_response())
    process_fn = pool_backend.make_llm_process_fn(registry)
    response = process_fn(seeded)

    assert response.text == "hello from codex"
    call_args = registry.get.return_value.generate.call_args[0][0]
    assert call_args.provider == ProviderName.CODEX
    assert call_args.model == "gpt-5.4-mini"
    assert call_args.messages[0].content == "headless round trip"
    assert call_args.reasoning == CodexReasoning(
        thinking_level=ThinkingLevel.XHIGH
    )
    assert call_args.sampling is None


def test_seed_llm_grid_honors_custom_request_keys() -> None:
    schema = PoolSchema.from_axis_names("rt2", ["llm_config", "prompt"])
    store, captured = _make_seed_store(schema)

    cfg = LlmConfig(
        provider=ProviderName.OPENAI,
        model="gpt-4.1-mini",
        mode=CallMode.api,
    )
    msgs = [Message(role=MessageRole.USER, content="hi")]

    seed_llm_grid(
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
    assert "model_cfg" in seeded.request
    assert "msgs" in seeded.request
    assert "llm_config" not in seeded.request
    assert "prompt" not in seeded.request

    registry = _make_registry()
    process_fn = pool_backend.make_llm_process_fn(
        registry, llm_config_key="model_cfg", prompt_key="msgs"
    )
    response = process_fn(seeded)
    assert response.text == "hello world"

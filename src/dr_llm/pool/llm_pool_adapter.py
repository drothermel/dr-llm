"""LLM-backed process function adapter for pool workers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from dr_llm.llm.config import LlmConfig, parse_llm_config
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.registry import ProviderRegistry
from dr_llm.logging.events import generation_log_context, get_generation_log_context
from dr_llm.logging.sinks import emit_generation_event
from dr_llm.pool.seed_grid import Axis, GridCell, seed_grid
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.pool.results import InsertResult

ProcessFn = Callable[[PoolSample], dict[str, Any]]


def _require_request_field(sample: PoolSample, key: str) -> Any:
    if key not in sample.request:
        raise KeyError(
            f"PoolSample.request is missing key {key!r}; "
            "was the pool seeded with a rich grid for this column?"
        )
    value = sample.request[key]
    if value is None:
        raise ValueError(
            f"PoolSample.request[{key!r}] is present but has explicit None value"
        )
    return value


def make_llm_process_fn(
    registry: ProviderRegistry,
    *,
    llm_config_key: str = "llm_config",
    prompt_key: str = "prompt",
) -> ProcessFn:
    """Build a ``ProcessFn`` that dispatches LLM calls via the provider registry.

    Expects pool samples whose ``request`` contains serialized
    :class:`LlmConfig` (under *llm_config_key*) and ``list[Message]``
    (under *prompt_key*).
    """

    def _process(sample: PoolSample) -> dict[str, Any]:
        raw_config = _require_request_field(sample, llm_config_key)
        raw_messages = _require_request_field(sample, prompt_key)

        config = parse_llm_config(raw_config)
        messages = [Message(**message) for message in raw_messages]
        request = config.to_request(messages)

        context = get_generation_log_context()
        model_provider = registry.get(request.provider)
        call_id = uuid4().hex
        worker_payload = {
            "pool_name": context.get("pool_name"),
            "sample_id": sample.sample_id,
            "sample_idx": sample.sample_idx,
            "worker_id": context.get("worker_id"),
            "key_values": sample.key_values,
        }

        with generation_log_context(
            {
                "call_id": call_id,
                "provider": request.provider,
                "model": request.model,
                "mode": model_provider.mode,
            }
        ):
            emit_generation_event(
                event_type="llm_call.started",
                stage="pool_worker.before_provider",
                payload={
                    **worker_payload,
                    "request": request.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    ),
                },
            )
            try:
                response = model_provider.generate(request)
            except Exception as exc:  # noqa: BLE001
                emit_generation_event(
                    event_type="llm_call.failed",
                    stage="pool_worker.provider_exception",
                    payload={
                        **worker_payload,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise

            emit_generation_event(
                event_type="llm_call.succeeded",
                stage="pool_worker.after_provider",
                payload={
                    **worker_payload,
                    "response": response.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    ),
                },
            )
            return response.model_dump()

    return _process


def seed_llm_grid(
    store: PoolStore,
    *,
    axes: list[Axis[Any]],
    build_request: Callable[[GridCell], tuple[list[Message], LlmConfig]],
    n: int = 1,
    build_metadata: Callable[[GridCell], dict[str, Any]] | None = None,
    chunk_size: int = 500,
    llm_config_key: str = "llm_config",
    prompt_key: str = "prompt",
) -> InsertResult:
    """Seed a pool from a cross-product of axes for LLM workers.

    Wraps :func:`seed_grid` with a request shape that
    :func:`make_llm_process_fn` can consume directly: each row's request
    is ``{llm_config_key: <serialized LlmConfig>, prompt_key: <list of
    serialized Messages>}``.

    Args:
        store: Target pool store. Its schema's key columns must match
            the axis names in order.
        axes: Ordered list of variant axes.
        build_request: Per-cell callback returning ``(messages, llm_config)``
            for that cell.
        n: Number of samples per cell (each gets a distinct ``sample_idx``).
        build_metadata: Optional per-cell row-metadata builder.
        chunk_size: Maximum rows per ``insert_samples`` round-trip.
        llm_config_key: Request key for the serialized LlmConfig. Must
            match the value passed to :func:`make_llm_process_fn`.
        prompt_key: Request key for the serialized message list. Must
            match the value passed to :func:`make_llm_process_fn`.

    Returns:
        Cumulative :class:`InsertResult` summed across all chunks.
    """

    def _build_payload(cell: GridCell) -> dict[str, Any]:
        messages, llm_config = build_request(cell)
        return {
            llm_config_key: llm_config.model_dump(mode="json"),
            prompt_key: [message.model_dump(mode="json") for message in messages],
        }

    return seed_grid(
        store,
        axes=axes,
        build_payload=_build_payload,
        n=n,
        build_metadata=build_metadata,
        chunk_size=chunk_size,
    )

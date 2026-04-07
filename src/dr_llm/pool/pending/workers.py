"""Helpers for seeding and draining pending pool work."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Any
from uuid import uuid4

from dr_llm.logging import emit_generation_event, generation_log_context
from dr_llm.logging.events import get_generation_log_context
from dr_llm.pool.errors import PoolSchemaError
from dr_llm.pool.pending.models import (
    PendingSample,
)
from dr_llm.pool.pending.pool_worker_controller import PoolWorkerController
from dr_llm.pool.pending.threadsafe_worker_stats import (
    ThreadsafeWorkerStats,
    WorkerSnapshot,
)
from dr_llm.pool.pending.utils import serialize_payload_value
from dr_llm.pool.results import InsertResult
from dr_llm.pool.sample_store import PoolStore
from dr_llm.llm.config import LlmConfig
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

ProcessFn = Callable[[PendingSample], dict[str, Any]]


def _parse_key_grid(
    column_names: list[str],
    key_grid: Mapping[str, Iterable[Any] | dict[str, Any]],
) -> tuple[list[list[Any]], dict[str, dict[str, Any]]] | None:
    """Parse key_grid into (grid_keys, rich_columns) or None if any column is empty."""
    grid_keys: list[list[Any]] = []
    rich_columns: dict[str, dict[str, Any]] = {}

    for name in column_names:
        raw = key_grid[name]
        if isinstance(raw, dict):
            keys = list(raw.keys())
            if not keys:
                return None
            grid_keys.append(keys)
            rich_columns[name] = dict(raw)
        else:
            if isinstance(raw, (str, bytes)):
                raise TypeError(f"key_grid[{name!r}] must be an iterable of values")
            values = list(raw)
            if not values:
                return None
            grid_keys.append(values)

    return grid_keys, rich_columns


def seed_pending(
    store: PoolStore,
    *,
    key_grid: Mapping[str, Iterable[Any] | dict[str, Any]],
    n: int,
    priority: int = 0,
) -> InsertResult:
    """Seed the pending queue from a key-dimension cartesian product.

    Grid values may be plain iterables (column stores value directly) or
    ``dict[str, Any]`` mappings (dict keys become column values, dict values
    are serialized into the pending sample's ``payload`` under the column
    name).
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    store.init_schema()
    expected = set(store.schema.key_column_names)
    provided = set(key_grid.keys())
    missing = expected - provided
    if missing:
        raise PoolSchemaError(f"Missing key columns: {missing}. Expected: {expected}")
    extra = provided - expected
    if extra:
        raise PoolSchemaError(f"Unexpected key columns: {extra}. Expected: {expected}")
    if n == 0:
        return InsertResult()

    column_names = store.schema.key_column_names
    parsed = _parse_key_grid(column_names, key_grid)
    if parsed is None:
        return InsertResult()
    grid_keys, rich_columns = parsed

    inserted = 0
    skipped = 0
    failed = 0
    for combination in product(*grid_keys):
        key_values = dict(zip(column_names, combination, strict=True))
        payload: dict[str, Any] = {}
        for name, col_value in key_values.items():
            if name in rich_columns:
                payload[name] = serialize_payload_value(rich_columns[name][col_value])
        for sample_idx in range(n):
            did_insert = store.pending.insert_pending(
                PendingSample(
                    key_values=key_values,
                    sample_idx=sample_idx,
                    payload=payload,
                    priority=priority,
                ),
                ignore_conflicts=True,
            )
            if did_insert:
                inserted += 1
            else:
                skipped += 1

    return InsertResult(inserted=inserted, skipped=skipped, failed=failed)


def start_workers(
    store: PoolStore,
    *,
    process_fn: ProcessFn,
    num_workers: int,
    lease_seconds: int = 300,
    min_poll_interval_s: float = 0.5,
    max_poll_interval_s: float = 5.0,
    backoff_factor: float = 2.0,
    max_retries: int = 0,
    key_filter: dict[str, Any] | None = None,
) -> PoolWorkerController:
    """Start long-lived queue-draining workers and return a controller."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if min_poll_interval_s <= 0 or max_poll_interval_s <= 0:
        raise ValueError("poll intervals must be positive")
    if max_poll_interval_s < min_poll_interval_s:
        raise ValueError("max_poll_interval_s must be >= min_poll_interval_s")
    if backoff_factor < 1.0:
        raise ValueError("backoff_factor must be >= 1.0")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    store.init_schema()

    stop_event = threading.Event()
    stats = ThreadsafeWorkerStats()
    executor = ThreadPoolExecutor(
        max_workers=num_workers, thread_name_prefix="pool-fill"
    )
    futures = [
        executor.submit(
            _worker_loop,
            store=store,
            process_fn=process_fn,
            worker_id=f"worker-{idx}-{uuid4().hex[:8]}",
            stop_event=stop_event,
            stats=stats,
            lease_seconds=lease_seconds,
            min_poll_interval_s=min_poll_interval_s,
            max_poll_interval_s=max_poll_interval_s,
            backoff_factor=backoff_factor,
            max_retries=max_retries,
            key_filter=key_filter,
        )
        for idx in range(num_workers)
    ]
    return PoolWorkerController(
        store=store,
        executor=executor,
        futures=futures,
        stop_event=stop_event,
        stats=stats,
        worker_count=num_workers,
        key_filter=key_filter,
    )


def run_workers(store: PoolStore, **kwargs: Any) -> WorkerSnapshot:
    """Start workers and block until interrupted, then stop them cleanly."""
    controller = start_workers(store, **kwargs)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Stopping workers on keyboard interrupt")
    finally:
        controller.stop()
    return controller.join()


def _worker_loop(
    *,
    store: PoolStore,
    process_fn: ProcessFn,
    worker_id: str,
    stop_event: threading.Event,
    stats: ThreadsafeWorkerStats,
    lease_seconds: int,
    min_poll_interval_s: float,
    max_poll_interval_s: float,
    backoff_factor: float,
    max_retries: int,
    key_filter: dict[str, Any] | None,
) -> None:
    poll_interval_s = min_poll_interval_s
    while not stop_event.is_set():
        claimed = store.pending.claim_pending(
            worker_id=worker_id,
            limit=1,
            lease_seconds=lease_seconds,
            key_filter=key_filter,
        )
        if not claimed:
            stats.incr("idle_polls")
            if stop_event.wait(poll_interval_s):
                break
            poll_interval_s = min(max_poll_interval_s, poll_interval_s * backoff_factor)
            continue

        poll_interval_s = min_poll_interval_s
        pending_sample = claimed[0]
        stats.incr("claimed")

        try:
            with generation_log_context(
                {"pool_name": store.schema.name, "worker_id": worker_id}
            ):
                payload = process_fn(pending_sample)
            if not isinstance(payload, dict):
                raise TypeError("process_fn must return a payload dict")
            promoted = store.pending.promote_pending(
                pending_id=pending_sample.pending_id,
                payload=payload,
            )
            if promoted is None:
                raise RuntimeError(
                    f"Failed to promote leased pending sample {pending_sample.pending_id}"
                )
            stats.incr("promoted")
        except Exception as exc:
            logger.exception(
                "Worker %s failed while processing pending sample %s",
                worker_id,
                pending_sample.pending_id,
            )
            stats.incr("process_errors")
            _handle_process_error(
                store=store,
                pending_sample=pending_sample,
                worker_id=worker_id,
                exc=exc,
                max_retries=max_retries,
                stats=stats,
            )


def _handle_process_error(
    *,
    store: PoolStore,
    pending_sample: PendingSample,
    worker_id: str,
    exc: Exception,
    max_retries: int,
    stats: ThreadsafeWorkerStats,
) -> None:
    if pending_sample.attempt_count <= max_retries:
        store.pending.release_pending_lease(
            pending_id=pending_sample.pending_id,
            worker_id=worker_id,
        )
        stats.incr("retried")
        return

    store.pending.fail_pending(
        pending_id=pending_sample.pending_id,
        worker_id=worker_id,
        reason=_format_error_reason(exc),
    )
    stats.incr("failed")


def _format_error_reason(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def make_llm_process_fn(
    registry: ProviderRegistry,
    *,
    llm_config_key: str = "llm_config",
    prompt_key: str = "prompt",
) -> ProcessFn:
    """Build a ``ProcessFn`` that dispatches LLM calls via the provider registry.

    Expects pending samples whose ``payload`` contains serialized
    :class:`LlmConfig` (under *llm_config_key*) and ``list[Message]``
    (under *prompt_key*), as produced by :func:`seed_pending` with rich
    grid values.
    """

    def _process(sample: PendingSample) -> dict[str, Any]:
        raw_config = sample.payload.get(llm_config_key)
        if raw_config is None:
            raise KeyError(
                f"Pending sample payload missing {llm_config_key!r}; "
                "was the pool seeded with a rich grid for this column?"
            )
        raw_messages = sample.payload.get(prompt_key)
        if raw_messages is None:
            raise KeyError(
                f"Pending sample payload missing {prompt_key!r}; "
                "was the pool seeded with a rich grid for this column?"
            )

        config = LlmConfig(**raw_config)
        messages = [Message(**m) for m in raw_messages]
        request = config.to_request(messages)

        context = get_generation_log_context()
        model_provider = registry.get(request.provider)
        call_id = uuid4().hex
        worker_payload = {
            "pool_name": context.get("pool_name"),
            "pending_id": sample.pending_id,
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

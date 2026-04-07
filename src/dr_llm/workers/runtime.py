from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from uuid import uuid4

from dr_llm.workers.backend import (
    ErrorAction,
    ProcessContextFactory,
    ProcessFn,
    WorkerBackend,
)
from dr_llm.workers.models import WorkerConfig, WorkerSnapshot
from dr_llm.workers.threadsafe_worker_stats import ThreadsafeWorkerStats
from dr_llm.workers.worker_controller import WorkerController

logger = logging.getLogger(__name__)


def start_workers[TWorkItem, TResult](
    backend: WorkerBackend[TWorkItem, TResult],
    *,
    process_fn: ProcessFn[TWorkItem, TResult],
    config: WorkerConfig,
    process_context_factory: ProcessContextFactory[TWorkItem] | None = None,
) -> WorkerController:
    """Start long-lived queue-draining workers and return a controller."""
    stop_event = threading.Event()
    stats = ThreadsafeWorkerStats()
    executor = ThreadPoolExecutor(
        max_workers=config.num_workers,
        thread_name_prefix=config.thread_name_prefix,
    )
    futures = [
        executor.submit(
            _worker_loop,
            backend=backend,
            process_fn=process_fn,
            worker_id=f"{config.thread_name_prefix}-{idx}-{uuid4().hex[:8]}",
            stop_event=stop_event,
            stats=stats,
            config=config,
            process_context_factory=process_context_factory,
        )
        for idx in range(config.num_workers)
    ]
    return WorkerController(
        backend=backend,
        executor=executor,
        futures=futures,
        stop_event=stop_event,
        stats=stats,
        worker_count=config.num_workers,
    )


def run_workers[TWorkItem, TResult](
    backend: WorkerBackend[TWorkItem, TResult],
    *,
    process_fn: ProcessFn[TWorkItem, TResult],
    config: WorkerConfig,
    process_context_factory: ProcessContextFactory[TWorkItem] | None = None,
) -> WorkerSnapshot:
    """Start workers and block until interrupted, then stop them cleanly."""
    controller = start_workers(
        backend,
        process_fn=process_fn,
        config=config,
        process_context_factory=process_context_factory,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Stopping workers on keyboard interrupt")
    finally:
        controller.stop()
    return controller.join()


def _worker_loop[TWorkItem, TResult](
    *,
    backend: WorkerBackend[TWorkItem, TResult],
    process_fn: ProcessFn[TWorkItem, TResult],
    worker_id: str,
    stop_event: threading.Event,
    stats: ThreadsafeWorkerStats,
    config: WorkerConfig,
    process_context_factory: ProcessContextFactory[TWorkItem] | None,
) -> None:
    poll_interval_s = config.min_poll_interval_s
    context_factory = process_context_factory or _null_context_factory

    while not stop_event.is_set():
        claimed = backend.claim(
            worker_id=worker_id,
            limit=1,
            lease_seconds=config.lease_seconds,
        )
        if not claimed:
            stats.incr("idle_polls")
            if stop_event.wait(poll_interval_s):
                break
            poll_interval_s = min(
                config.max_poll_interval_s,
                poll_interval_s * config.backoff_factor,
            )
            continue

        poll_interval_s = config.min_poll_interval_s
        item = claimed[0]
        stats.incr("claimed")

        try:
            with context_factory(item, worker_id):
                result = process_fn(item)
            backend.complete(item=item, result=result, worker_id=worker_id)
            stats.incr("completed")
        except Exception as exc:
            logger.exception("Worker %s failed while processing work item", worker_id)
            stats.incr("process_errors")
            action = backend.handle_process_error(
                item=item,
                worker_id=worker_id,
                exc=exc,
                max_retries=config.max_retries,
            )
            if action is ErrorAction.retry:
                stats.incr("retried")
                continue
            if action is ErrorAction.fail:
                stats.incr("failed")
                continue
            raise ValueError(f"Unsupported error action: {action!r}")


def _null_context_factory(_item: object, _worker_id: str):
    return nullcontext()

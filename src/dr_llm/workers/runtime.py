from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from pydantic import BaseModel

from dr_llm.workers.backend import (
    ErrorDecision,
    ProcessFn,
    WorkerBackend,
)
from dr_llm.workers.models import WorkerConfig, WorkerSnapshot, WorkerStatKey
from dr_llm.workers.threadsafe_worker_stats import ThreadsafeWorkerStats
from dr_llm.workers.worker_controller import WorkerController

logger = logging.getLogger(__name__)

_ERROR_DECISION_STAT: dict[ErrorDecision, WorkerStatKey] = {
    ErrorDecision.retry: "retried",
    ErrorDecision.fail: "failed",
}


def _make_worker_id(*, prefix: str, idx: int) -> str:
    return f"{prefix}-{idx}-{uuid4().hex[:8]}"


def start_workers[TWorkItem, TResult, TBackendState: BaseModel](
    backend: WorkerBackend[TWorkItem, TResult, TBackendState],
    *,
    process_fn: ProcessFn[TWorkItem, TResult],
    config: WorkerConfig,
) -> WorkerController[TBackendState]:
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
            worker_id=_make_worker_id(prefix=config.thread_name_prefix, idx=idx),
            stop_event=stop_event,
            stats=stats,
            config=config,
        )
        for idx in range(config.num_workers)
    ]
    return WorkerController(
        get_backend_state=backend.snapshot,
        executor=executor,
        futures=futures,
        stop_event=stop_event,
        stats=stats,
    )


def run_workers_forever[TWorkItem, TResult, TBackendState: BaseModel](
    backend: WorkerBackend[TWorkItem, TResult, TBackendState],
    *,
    process_fn: ProcessFn[TWorkItem, TResult],
    config: WorkerConfig,
) -> WorkerSnapshot[TBackendState]:
    """Start workers and block until interrupted, then stop them cleanly."""
    controller = start_workers(
        backend,
        process_fn=process_fn,
        config=config,
    )
    try:
        controller.wait_for_stop()
    except KeyboardInterrupt:
        logger.info("Stopping workers on keyboard interrupt")
    finally:
        controller.stop()
    return controller.join()


def _worker_loop[TWorkItem, TResult, TBackendState: BaseModel](
    *,
    backend: WorkerBackend[TWorkItem, TResult, TBackendState],
    process_fn: ProcessFn[TWorkItem, TResult],
    worker_id: str,
    stop_event: threading.Event,
    stats: ThreadsafeWorkerStats,
    config: WorkerConfig,
) -> None:
    poll_interval_s = config.min_poll_interval_s
    while not stop_event.is_set():
        item = backend.claim(worker_id=worker_id, lease_seconds=config.lease_seconds)
        if item is None:
            next_interval = _handle_idle_poll(
                stop_event=stop_event,
                stats=stats,
                poll_interval_s=poll_interval_s,
                config=config,
            )
            if next_interval is None:
                break
            poll_interval_s = next_interval
            continue
        poll_interval_s = config.min_poll_interval_s
        _process_one_item(
            backend=backend,
            process_fn=process_fn,
            item=item,
            worker_id=worker_id,
            stats=stats,
        )


def _handle_idle_poll(
    *,
    stop_event: threading.Event,
    stats: ThreadsafeWorkerStats,
    poll_interval_s: float,
    config: WorkerConfig,
) -> float | None:
    """Sleep through an idle poll; return next interval, or None if stopping."""
    stats.incr("idle_polls")
    if stop_event.wait(poll_interval_s):
        return None
    return min(config.max_poll_interval_s, poll_interval_s * config.backoff_factor)


def _process_one_item[TWorkItem, TResult, TBackendState: BaseModel](
    *,
    backend: WorkerBackend[TWorkItem, TResult, TBackendState],
    process_fn: ProcessFn[TWorkItem, TResult],
    item: TWorkItem,
    worker_id: str,
    stats: ThreadsafeWorkerStats,
) -> None:
    """Run a claimed item through process_fn and record success/error stats."""
    stats.incr("claimed")
    try:
        with backend.process_context(item=item, worker_id=worker_id):
            result = process_fn(item)
        backend.complete(item=item, result=result, worker_id=worker_id)
        stats.incr("completed")
    except Exception as exc:
        logger.exception("Worker %s failed while processing work item", worker_id)
        stats.incr("process_errors")
        action = backend.handle_process_error(item=item, worker_id=worker_id, exc=exc)
        stats.incr(_ERROR_DECISION_STAT[action])

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from dr_llm.pool.pending.threadsafe_worker_stats import (
    ThreadsafeWorkerStats,
    WorkerSnapshot,
)
from dr_llm.pool.sample_store import PoolStore


class PoolWorkerController:
    """Manage background workers draining a pool's pending queue."""

    def __init__(
        self,
        *,
        store: PoolStore,
        executor: ThreadPoolExecutor,
        futures: list[Future[None]],
        stop_event: threading.Event,
        stats: ThreadsafeWorkerStats,
        worker_count: int,
        key_filter: dict[str, Any] | None,
    ) -> None:
        self._store = store
        self._executor = executor
        self._futures = futures
        self._stop_event = stop_event
        self._stats = stats
        self._worker_count = worker_count
        self._key_filter = key_filter
        self._joined = False

    def stop(self) -> None:
        """Request graceful worker shutdown."""
        self._stop_event.set()

    def snapshot(self) -> WorkerSnapshot:
        """Return cumulative worker stats plus current queue status counts."""
        return self._stats.snapshot(
            worker_count=self._worker_count,
            stop_requested=self._stop_event.is_set(),
            status_counts=self._store.pending.status_counts(
                key_filter=self._key_filter
            ),
        )

    def join(self, timeout: float | None = None) -> WorkerSnapshot:
        """Wait for workers to exit, returning a final snapshot."""
        deadline = None if timeout is None else time.monotonic() + timeout
        timed_out = False
        try:
            for future in self._futures:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        raise TimeoutError("Timed out waiting for workers to stop")
                future.result(timeout=remaining)
        except TimeoutError:
            if not self._joined:
                self._executor.shutdown(wait=False, cancel_futures=False)
                self._joined = True
            raise
        if not self._joined:
            self._executor.shutdown(wait=not timed_out, cancel_futures=False)
            self._joined = True
        return self.snapshot()

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from dr_llm.workers.backend import WorkerBackend
from dr_llm.workers.threadsafe_worker_stats import ThreadsafeWorkerStats
from dr_llm.workers.models import WorkerSnapshot

TBackendState = TypeVar("TBackendState", bound=BaseModel)


class WorkerController(Generic[TBackendState]):
    """Manage background workers draining a generic lease-based backend."""

    def __init__(
        self,
        *,
        backend: WorkerBackend[Any, Any, TBackendState],
        executor: ThreadPoolExecutor,
        futures: list[Future[None]],
        stop_event: threading.Event,
        stats: ThreadsafeWorkerStats[TBackendState],
        worker_count: int,
    ) -> None:
        self._backend = backend
        self._executor = executor
        self._futures = futures
        self._stop_event = stop_event
        self._stats = stats
        self._worker_count = worker_count
        self._joined = False

    def stop(self) -> None:
        """Request graceful worker shutdown."""
        self._stop_event.set()

    def snapshot(self) -> WorkerSnapshot[TBackendState]:
        """Return cumulative worker stats plus current backend state."""
        return self._stats.snapshot(
            worker_count=self._worker_count,
            stop_requested=self._stop_event.is_set(),
            backend_state=self._backend.snapshot(),
        )

    def join(self, timeout: float | None = None) -> WorkerSnapshot[TBackendState]:
        """Wait for workers to exit, returning a final snapshot."""
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            for future in self._futures:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for workers to stop")
                future.result(timeout=remaining)
        except TimeoutError:
            if not self._joined:
                self._executor.shutdown(wait=False, cancel_futures=False)
                self._joined = True
            raise
        if not self._joined:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._joined = True
        return self.snapshot()

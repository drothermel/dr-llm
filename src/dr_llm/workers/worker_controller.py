from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait

from pydantic import BaseModel

from dr_llm.workers.models import WorkerSnapshot
from dr_llm.workers.threadsafe_worker_stats import ThreadsafeWorkerStats


class WorkerController[TBackendState: BaseModel]:
    """Manage background workers draining a generic lease-based backend."""

    def __init__(
        self,
        *,
        get_backend_state: Callable[[], TBackendState | None],
        executor: ThreadPoolExecutor,
        futures: list[Future[None]],
        stop_event: threading.Event,
        stats: ThreadsafeWorkerStats,
    ) -> None:
        self._get_backend_state = get_backend_state
        self._executor = executor
        self._futures = futures
        self._stop_event = stop_event
        self._stats = stats
        self._final_snapshot: WorkerSnapshot[TBackendState] | None = None
        self._join_lock = threading.Lock()

    @property
    def final_snapshot(self) -> WorkerSnapshot[TBackendState] | None:
        """Snapshot captured when :meth:`join` runs its cleanup, if any."""
        return self._final_snapshot

    def stop(self) -> None:
        """Request graceful worker shutdown."""
        self._stop_event.set()

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        """Block until stop has been requested. Returns True if stop fired."""
        return self._stop_event.wait(timeout)

    def snapshot(self) -> WorkerSnapshot[TBackendState]:
        """Return cumulative worker stats plus current backend state."""
        return WorkerSnapshot(
            worker_count=len(self._futures),
            stop_requested=self._stop_event.is_set(),
            counts=self._stats.snapshot(),
            backend_state=self._get_backend_state(),
        )

    def join(self, timeout: float | None = None) -> WorkerSnapshot[TBackendState]:
        """Wait for workers to exit, returning a final snapshot.

        Safe to call multiple times: subsequent calls return the snapshot
        cached on the first call without re-waiting or re-raising.
        """
        if self._final_snapshot is not None:
            return self._final_snapshot

        with self._join_lock:
            if self._final_snapshot is not None:
                return self._final_snapshot

            _, not_done = wait(self._futures, timeout=timeout)
            all_done = not not_done
            try:
                if not_done:
                    raise TimeoutError("Timed out waiting for workers to stop")
                for future in self._futures:
                    future.result()
            finally:
                # Skip waiting on shutdown after a timeout to avoid blocking
                # indefinitely on workers that failed to stop.
                self._executor.shutdown(wait=all_done, cancel_futures=False)
                self._final_snapshot = self.snapshot()
            return self._final_snapshot

from __future__ import annotations

import threading

from pydantic import BaseModel

from dr_llm.workers.models import WorkerSnapshot, WorkerStatCounts, WorkerStatKey


class ThreadsafeWorkerStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = WorkerStatCounts()

    def incr(self, key: WorkerStatKey, amount: int = 1) -> None:
        with self._lock:
            self._counts.increment(key, amount)

    def snapshot(
        self,
        *,
        worker_count: int,
        stop_requested: bool,
        backend_state: BaseModel | None,
    ) -> WorkerSnapshot:
        with self._lock:
            counts = self._counts.model_copy()
        return WorkerSnapshot(
            worker_count=worker_count,
            stop_requested=stop_requested,
            counts=counts,
            backend_state=backend_state,
        )

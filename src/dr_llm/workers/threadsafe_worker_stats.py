from __future__ import annotations

import threading

from dr_llm.workers.models import WorkerStatCounts, WorkerStatKey


class ThreadsafeWorkerStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = WorkerStatCounts()

    def incr(self, key: WorkerStatKey, amount: int = 1) -> None:
        with self._lock:
            current: int = getattr(self._counts, key)
            self._counts = self._counts.model_copy(update={key: current + amount})

    def snapshot(self) -> WorkerStatCounts:
        with self._lock:
            return self._counts

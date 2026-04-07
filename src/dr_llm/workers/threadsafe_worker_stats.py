from __future__ import annotations

import threading
from typing import Generic, TypeVar

from pydantic import BaseModel

from dr_llm.workers.models import (
    WORKER_STAT_KEYS,
    WorkerSnapshot,
    WorkerStatCounts,
    WorkerStatKey,
)

TBackendState = TypeVar("TBackendState", bound=BaseModel)


class ThreadsafeWorkerStats(Generic[TBackendState]):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[WorkerStatKey, int] = {key: 0 for key in WORKER_STAT_KEYS}

    def incr(self, key: WorkerStatKey, amount: int = 1) -> None:
        with self._lock:
            self._counts[key] += amount

    def snapshot(
        self,
        *,
        worker_count: int,
        stop_requested: bool,
        backend_state: TBackendState | None,
    ) -> WorkerSnapshot[TBackendState]:
        with self._lock:
            counts = WorkerStatCounts(**self._counts)
        return WorkerSnapshot(
            worker_count=worker_count,
            stop_requested=stop_requested,
            counts=counts,
            backend_state=backend_state,
        )

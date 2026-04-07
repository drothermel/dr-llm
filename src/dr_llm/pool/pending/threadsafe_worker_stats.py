from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.pending.models import PendingStatusCounts


class WorkerStatCounts(BaseModel):
    claimed: int = 0
    promoted: int = 0
    failed: int = 0
    retried: int = 0
    process_errors: int = 0
    idle_polls: int = 0

    def increment(self, key: str, amount: int = 1) -> None:
        setattr(self, key, getattr(self, key) + amount)


class WorkerSnapshot(BaseModel):
    """Observable state for a running pool worker controller."""

    model_config = ConfigDict(frozen=True)

    worker_count: int
    stop_requested: bool = False
    counts: WorkerStatCounts = Field(default_factory=WorkerStatCounts)
    status_counts: PendingStatusCounts = Field(default_factory=PendingStatusCounts)


class ThreadsafeWorkerStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = WorkerStatCounts()

    def incr(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counts.increment(key, amount)

    def snapshot(
        self,
        *,
        worker_count: int,
        stop_requested: bool,
        status_counts: PendingStatusCounts,
    ) -> WorkerSnapshot:
        with self._lock:
            counts = self._counts.model_copy()
        return WorkerSnapshot(
            worker_count=worker_count,
            stop_requested=stop_requested,
            counts=counts,
            status_counts=status_counts,
        )

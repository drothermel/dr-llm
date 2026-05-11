from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any

from pydantic import BaseModel, ConfigDict

from dr_llm.pool import (
    LlmPoolBackendState,
    drain_pool,
    format_worker_progress_line,
    pool_is_idle,
)
from dr_llm.workers import (
    ErrorDecision,
    WorkerConfig,
    WorkerSnapshot,
    WorkerStatCounts,
    start_workers,
)
from dr_llm.workers.backend import WorkerBackend


class StateWithoutIncomplete(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    complete: int = 0


class StateWithLease(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incomplete: int = 0
    complete: int = 0
    leased: int = 0


class MiniPoolBackend(WorkerBackend[str, str, LlmPoolBackendState]):
    def __init__(self, items: list[str]) -> None:
        self._lock = threading.Lock()
        self._queued = list(items)
        self._incomplete = len(items)
        self._complete = 0

    def claim(self, *, worker_id: str, lease_seconds: int) -> str | None:
        del worker_id, lease_seconds
        with self._lock:
            if not self._queued:
                return None
            return self._queued.pop(0)

    def complete(self, *, item: str, result: str, worker_id: str) -> None:
        del item, result, worker_id
        with self._lock:
            self._complete += 1
            self._incomplete -= 1

    def handle_process_error(
        self,
        *,
        item: str,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision:
        del item, worker_id, exc
        return ErrorDecision.fail

    def snapshot(self) -> LlmPoolBackendState:
        with self._lock:
            return LlmPoolBackendState(
                incomplete=self._incomplete,
                complete=self._complete,
            )

    def process_context(self, *, item: str, worker_id: str) -> Any:
        del item, worker_id
        return nullcontext()


def test_pool_is_idle_requires_backend_state() -> None:
    snapshot = WorkerSnapshot[LlmPoolBackendState](worker_count=1)

    assert not pool_is_idle(snapshot)


def test_pool_is_idle_requires_incomplete_field() -> None:
    snapshot = WorkerSnapshot[StateWithoutIncomplete](
        worker_count=1,
        backend_state=StateWithoutIncomplete(complete=3),
    )

    assert not pool_is_idle(snapshot)


def test_pool_is_idle_uses_incomplete_count() -> None:
    busy = WorkerSnapshot[LlmPoolBackendState](
        worker_count=1,
        backend_state=LlmPoolBackendState(incomplete=1, complete=2),
    )
    idle = WorkerSnapshot[LlmPoolBackendState](
        worker_count=1,
        backend_state=LlmPoolBackendState(incomplete=0, complete=3),
    )

    assert not pool_is_idle(busy)
    assert pool_is_idle(idle)


def test_format_worker_progress_line_includes_pool_fields() -> None:
    snapshot = WorkerSnapshot[StateWithLease](
        worker_count=2,
        counts=WorkerStatCounts(
            claimed=3,
            completed=2,
            failed=1,
            retried=4,
            process_errors=5,
        ),
        backend_state=StateWithLease(incomplete=6, complete=7, leased=8),
    )

    assert format_worker_progress_line(snapshot) == (
        "claimed=3 | completed=2 | failed=1 | retried=4 | errors=5 | "
        "incomplete=6 | complete=7 | leased=8"
    )


def test_drain_pool_drains_and_joins_workers() -> None:
    backend = MiniPoolBackend(["a", "b"])
    controller = start_workers(
        backend,
        process_fn=lambda item: item,
        config=WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.02,
        ),
    )

    snapshot = drain_pool(controller, poll_interval_s=0.01)

    assert snapshot.stop_requested is True
    assert snapshot.counts.completed == 2
    assert snapshot.backend_state is not None
    assert snapshot.backend_state.incomplete == 0
    assert snapshot.backend_state.complete == 2

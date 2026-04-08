from __future__ import annotations

import threading
import time
from contextlib import contextmanager, nullcontext
from collections.abc import Callable
from typing import Any, NoReturn

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from dr_llm.workers import (
    ErrorDecision,
    WorkerConfig,
    WorkerController,
    WorkerSnapshot,
    start_workers,
)
from dr_llm.workers.backend import WorkerBackend


class FakeBackendState(BaseModel):
    model_config = ConfigDict(frozen=True)

    queued: int = 0
    completed: int = 0
    failed: int = 0
    retries: int = 0
    claims: int = 0


class FakeWorkerBackend(WorkerBackend[str, dict[str, Any], FakeBackendState]):
    def __init__(
        self,
        items: list[str],
        *,
        max_retries: int = 0,
        process_context_enabled: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._queued = list(items)
        self._completed: list[str] = []
        self._failed: list[str] = []
        self._attempts: dict[str, int] = {}
        self._claims = 0
        self._retries = 0
        self._max_retries = max_retries
        self._process_context_enabled = process_context_enabled
        self._events = events

    def claim(self, *, worker_id: str, lease_seconds: int) -> str | None:
        del worker_id, lease_seconds
        with self._lock:
            if not self._queued:
                return None
            self._claims += 1
            return self._queued.pop(0)

    def complete(self, *, item: str, result: dict[str, Any], worker_id: str) -> None:
        del worker_id
        assert result["item"] == item
        with self._lock:
            self._completed.append(item)

    def handle_process_error(
        self,
        *,
        item: str,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision:
        del worker_id, exc
        with self._lock:
            attempts = self._attempts.get(item, 0) + 1
            self._attempts[item] = attempts
            if attempts <= self._max_retries:
                self._retries += 1
                self._queued.append(item)
                return ErrorDecision.retry
            self._failed.append(item)
            return ErrorDecision.fail

    def snapshot(self) -> FakeBackendState:
        with self._lock:
            return FakeBackendState(
                queued=len(self._queued),
                completed=len(self._completed),
                failed=len(self._failed),
                retries=self._retries,
                claims=self._claims,
            )

    @contextmanager
    def process_context(self, *, item: str, worker_id: str):
        if not self._process_context_enabled:
            with nullcontext():
                yield
            return

        assert self._events is not None
        self._events.append(f"enter:{item}:{worker_id}")
        try:
            yield
        finally:
            self._events.append(f"exit:{item}:{worker_id}")


class ExplodingClaimBackend(WorkerBackend[str, dict[str, Any], FakeBackendState]):
    def claim(self, *, worker_id: str, lease_seconds: int) -> str | None:
        del worker_id, lease_seconds
        raise RuntimeError("claim exploded")

    def complete(
        self,
        *,
        item: str,
        result: dict[str, Any],
        worker_id: str,
    ) -> None:
        raise AssertionError("complete should never be called")

    def handle_process_error(
        self,
        *,
        item: str,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision:
        raise AssertionError("handle_process_error should never be called")

    def snapshot(self) -> FakeBackendState:
        return FakeBackendState()


def _wait_for(
    controller: WorkerController[FakeBackendState],
    predicate: Callable[[WorkerSnapshot[FakeBackendState]], bool],
    *,
    timeout_s: float = 2.0,
) -> WorkerSnapshot[FakeBackendState]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = controller.snapshot()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for worker state")


def _stop_controller(
    controller: WorkerController[FakeBackendState],
) -> WorkerSnapshot[FakeBackendState]:
    controller.stop()
    return controller.join(timeout=5.0)


def test_worker_config_validation() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkerConfig(num_workers=0)
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkerConfig(num_workers=1, lease_seconds=0)
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkerConfig(num_workers=1, min_poll_interval_s=0.0)
    with pytest.raises(ValidationError, match="max_poll_interval_s must be >="):
        WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.2,
            max_poll_interval_s=0.1,
        )
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        WorkerConfig(num_workers=1, backoff_factor=0.5)
    with pytest.raises(ValidationError, match="thread_name_prefix must be non-empty"):
        WorkerConfig(num_workers=1, thread_name_prefix="  ")


def test_start_workers_completes_claimed_items() -> None:
    backend = FakeWorkerBackend(["a", "b", "c"])
    controller = start_workers(
        backend,
        process_fn=lambda item: {"item": item},
        config=WorkerConfig(
            num_workers=2,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.05,
        ),
    )
    try:
        snapshot = _wait_for(
            controller,
            lambda snap: (
                snap.backend_state is not None and snap.backend_state.completed == 3
            ),
        )
    finally:
        snapshot = _stop_controller(controller)

    assert snapshot.counts.claimed == 3
    assert snapshot.counts.completed == 3
    assert snapshot.counts.failed == 0
    assert snapshot.counts.retried == 0
    assert snapshot.backend_state is not None
    assert snapshot.backend_state.completed == 3
    assert snapshot.backend_state.queued == 0


def test_idle_polling_increments_and_stops_cleanly() -> None:
    backend = FakeWorkerBackend([])
    controller = start_workers(
        backend,
        process_fn=lambda item: {"item": item},
        config=WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.02,
        ),
    )
    try:
        snapshot = _wait_for(controller, lambda snap: snap.counts.idle_polls >= 2)
    finally:
        snapshot = _stop_controller(controller)

    assert snapshot.counts.claimed == 0
    assert snapshot.counts.idle_polls >= 2
    assert snapshot.stop_requested is True


def test_retry_then_success() -> None:
    backend = FakeWorkerBackend(["retry-once"], max_retries=1)
    attempts = {"count": 0}

    def flaky_process(item: str) -> dict[str, str]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        return {"item": item}

    controller = start_workers(
        backend,
        process_fn=flaky_process,
        config=WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.05,
        ),
    )
    try:
        snapshot = _wait_for(
            controller,
            lambda snap: (
                snap.backend_state is not None and snap.backend_state.completed == 1
            ),
        )
    finally:
        snapshot = _stop_controller(controller)

    assert snapshot.counts.claimed == 2
    assert snapshot.counts.completed == 1
    assert snapshot.counts.retried == 1
    assert snapshot.counts.failed == 0
    assert snapshot.counts.process_errors == 1
    assert snapshot.backend_state is not None
    assert snapshot.backend_state.retries == 1
    assert snapshot.backend_state.failed == 0


def _raise_runtime_boom(_item: str) -> NoReturn:
    raise RuntimeError("boom")


def test_retry_then_fail() -> None:
    backend = FakeWorkerBackend(["always-fail"], max_retries=1)

    controller = start_workers(
        backend,
        process_fn=_raise_runtime_boom,
        config=WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.05,
        ),
    )
    try:
        snapshot = _wait_for(
            controller,
            lambda snap: (
                snap.backend_state is not None and snap.backend_state.failed == 1
            ),
        )
    finally:
        snapshot = _stop_controller(controller)

    assert snapshot.counts.claimed == 2
    assert snapshot.counts.completed == 0
    assert snapshot.counts.retried == 1
    assert snapshot.counts.failed == 1
    assert snapshot.counts.process_errors == 2


def test_process_context_hook_runs_around_work() -> None:
    events: list[str] = []
    backend = FakeWorkerBackend(
        ["ctx"],
        process_context_enabled=True,
        events=events,
    )

    controller = start_workers(
        backend,
        process_fn=lambda item: {"item": item},
        config=WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.05,
        ),
    )
    try:
        snapshot = _wait_for(
            controller,
            lambda snap: (
                snap.backend_state is not None and snap.backend_state.completed == 1
            ),
        )
    finally:
        _stop_controller(controller)

    assert snapshot.counts.completed == 1
    assert len(events) == 2
    assert events[0].startswith("enter:ctx:")
    assert events[1].startswith("exit:ctx:")


def test_snapshot_includes_backend_state() -> None:
    backend = FakeWorkerBackend(["only-item"])
    controller = start_workers(
        backend,
        process_fn=lambda item: {"item": item},
        config=WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.05,
        ),
    )
    try:
        snapshot = controller.snapshot()
    finally:
        _stop_controller(controller)

    assert snapshot.backend_state is not None
    assert isinstance(snapshot.backend_state, FakeBackendState)


def test_snapshot_counts_are_frozen() -> None:
    backend = FakeWorkerBackend(["only-item"])
    controller = start_workers(
        backend,
        process_fn=lambda item: {"item": item},
        config=WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.05,
        ),
    )
    try:
        snapshot = controller.snapshot()
    finally:
        _stop_controller(controller)

    with pytest.raises(ValidationError, match="Instance is frozen"):
        snapshot.counts.claimed = 1


def test_join_shuts_down_executor_on_non_timeout_exception() -> None:
    controller = start_workers(
        ExplodingClaimBackend(),
        process_fn=lambda item: {"item": item},
        config=WorkerConfig(
            num_workers=1,
            min_poll_interval_s=0.01,
            max_poll_interval_s=0.05,
        ),
    )

    with pytest.raises(RuntimeError, match="claim exploded"):
        controller.join(timeout=5.0)

    assert controller.final_snapshot is not None

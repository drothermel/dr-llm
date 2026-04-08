"""Unit tests for pool progress helpers."""

from __future__ import annotations

from dr_llm.pool.pending.backend import PoolPendingBackendState
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.pending.progress import (
    format_pool_progress_line,
    pool_is_idle,
    pool_progress_key,
)
from dr_llm.workers.models import WorkerSnapshot, WorkerStatCounts


def _snapshot(
    *,
    counts: WorkerStatCounts | None = None,
    backend_state: PoolPendingBackendState | None = None,
) -> WorkerSnapshot[PoolPendingBackendState]:
    return WorkerSnapshot[PoolPendingBackendState](
        worker_count=2,
        counts=counts or WorkerStatCounts(),
        backend_state=backend_state,
    )


def _backend_state(
    *,
    pending: int = 0,
    leased: int = 0,
    promoted: int = 0,
    failed: int = 0,
) -> PoolPendingBackendState:
    return PoolPendingBackendState(
        status_counts=PendingStatusCounts(
            pending=pending, leased=leased, promoted=promoted, failed=failed
        ),
    )


def test_format_pool_progress_line_with_backend_state() -> None:
    snap = _snapshot(
        counts=WorkerStatCounts(claimed=6, completed=4, failed=1),
        backend_state=_backend_state(pending=2, leased=1, promoted=4, failed=1),
    )
    assert (
        format_pool_progress_line(snap)
        == "claimed=6 completed=4 failed=1 pending=2 leased=1"
    )


def test_format_pool_progress_line_backend_state_none() -> None:
    snap = _snapshot(counts=WorkerStatCounts(claimed=1, completed=0, failed=0))
    assert (
        format_pool_progress_line(snap)
        == "claimed=1 completed=0 failed=0 pending=? leased=?"
    )


def test_pool_progress_key_with_backend_state() -> None:
    snap = _snapshot(
        counts=WorkerStatCounts(claimed=6, completed=4, failed=1),
        backend_state=_backend_state(pending=2, leased=1, promoted=4, failed=1),
    )
    assert pool_progress_key(snap) == (6, 4, 1, 2, 1)


def test_pool_progress_key_backend_state_none() -> None:
    snap = _snapshot(counts=WorkerStatCounts(claimed=2, completed=1, failed=0))
    assert pool_progress_key(snap) == (2, 1, 0, -1, -1)


def test_pool_progress_key_ignores_unrelated_fields() -> None:
    """Mutating only `retried`, `process_errors`, or `idle_polls` does not
    change the key. This is the whole point of the helper — guards against
    polling loops being fooled by `idle_polls` ticking up on every empty
    poll.
    """
    base = _snapshot(
        counts=WorkerStatCounts(claimed=3, completed=2, failed=0),
        backend_state=_backend_state(pending=1, leased=0),
    )
    noisy = _snapshot(
        counts=WorkerStatCounts(
            claimed=3,
            completed=2,
            failed=0,
            retried=99,
            process_errors=7,
            idle_polls=12345,
        ),
        backend_state=_backend_state(pending=1, leased=0),
    )
    assert pool_progress_key(base) == pool_progress_key(noisy)


def test_pool_is_idle_true_when_in_flight_zero() -> None:
    snap = _snapshot(
        counts=WorkerStatCounts(claimed=6, completed=5, failed=1),
        backend_state=_backend_state(pending=0, leased=0, promoted=5, failed=1),
    )
    assert pool_is_idle(snap) is True


def test_pool_is_idle_false_when_backend_state_none() -> None:
    snap = _snapshot(counts=WorkerStatCounts(claimed=0, completed=0, failed=0))
    assert pool_is_idle(snap) is False


def test_pool_is_idle_false_when_pending_nonzero() -> None:
    snap = _snapshot(backend_state=_backend_state(pending=3, leased=0))
    assert pool_is_idle(snap) is False


def test_pool_is_idle_false_when_leased_nonzero() -> None:
    snap = _snapshot(backend_state=_backend_state(pending=0, leased=2))
    assert pool_is_idle(snap) is False

"""Unit tests for pool progress helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from dr_llm.pool.pending import progress as progress_module
from dr_llm.pool.pending.backend import PoolPendingBackendState
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.pending.progress import (
    drain,
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
    failed: int = 0,
) -> PoolPendingBackendState:
    return PoolPendingBackendState(
        status_counts=PendingStatusCounts(
            pending=pending, leased=leased, failed=failed
        ),
    )


def test_format_pool_progress_line_with_backend_state() -> None:
    snap = _snapshot(
        counts=WorkerStatCounts(claimed=6, completed=4, failed=1),
        backend_state=_backend_state(pending=2, leased=1, failed=1),
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
        backend_state=_backend_state(pending=2, leased=1, failed=1),
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
        backend_state=_backend_state(pending=0, leased=0, failed=1),
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


def _fake_controller_yielding(
    snapshots: list[WorkerSnapshot[PoolPendingBackendState]],
) -> MagicMock:
    """Build a MagicMock WorkerController whose .snapshot() returns the given
    sequence on successive calls (last value sticks)."""
    controller = MagicMock()
    iterator = iter(snapshots)
    last: list[WorkerSnapshot[PoolPendingBackendState]] = [snapshots[-1]]

    def _next() -> WorkerSnapshot[PoolPendingBackendState]:
        try:
            value = next(iterator)
            last[0] = value
            return value
        except StopIteration:
            return last[0]

    controller.snapshot.side_effect = _next
    return controller


def test_drain_returns_idle_snapshot_and_skips_sleep_at_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(progress_module.time, "sleep", sleep_calls.append)

    snapshots = [
        _snapshot(
            counts=WorkerStatCounts(claimed=0, completed=0, failed=0),
            backend_state=_backend_state(pending=2, leased=0),
        ),
        _snapshot(
            counts=WorkerStatCounts(claimed=1, completed=0, failed=0),
            backend_state=_backend_state(pending=1, leased=1),
        ),
        _snapshot(
            counts=WorkerStatCounts(claimed=2, completed=2, failed=0),
            backend_state=_backend_state(pending=0, leased=0),
        ),
    ]
    controller = _fake_controller_yielding(snapshots)

    final = drain(controller, poll_interval_s=0.01)

    # Drained to the idle snapshot, with one sleep per non-idle poll.
    assert pool_is_idle(final) is True
    assert final is snapshots[-1]
    assert len(sleep_calls) == 2


def test_drain_calls_on_change_only_on_visible_state_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_module.time, "sleep", lambda _s: None)

    in_flight = _snapshot(
        counts=WorkerStatCounts(claimed=1, completed=0, failed=0),
        backend_state=_backend_state(pending=1, leased=0),
    )
    in_flight_noisy = _snapshot(
        counts=WorkerStatCounts(claimed=1, completed=0, failed=0, idle_polls=99),
        backend_state=_backend_state(pending=1, leased=0),
    )
    progressed = _snapshot(
        counts=WorkerStatCounts(claimed=2, completed=1, failed=0),
        backend_state=_backend_state(pending=0, leased=1),
    )
    idle = _snapshot(
        counts=WorkerStatCounts(claimed=2, completed=2, failed=0),
        backend_state=_backend_state(pending=0, leased=0),
    )
    controller = _fake_controller_yielding(
        [in_flight, in_flight_noisy, progressed, idle]
    )

    seen: list[Any] = []
    drain(
        controller,
        on_change=lambda snap: seen.append(pool_progress_key(snap)),
        poll_interval_s=0.01,
    )

    # 4 polls total but only 3 visible state changes (the noisy duplicate is
    # collapsed into the previous in-flight key).
    assert len(seen) == 3
    assert seen[0] == pool_progress_key(in_flight)
    assert seen[1] == pool_progress_key(progressed)
    assert seen[2] == pool_progress_key(idle)


def test_drain_rejects_nonpositive_poll_interval() -> None:
    controller = MagicMock()
    with pytest.raises(ValueError, match="poll_interval_s"):
        drain(controller, poll_interval_s=0)

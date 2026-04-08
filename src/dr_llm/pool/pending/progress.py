"""Progress helpers for polling a pool-backed WorkerController.

These helpers are the single source of truth for formatting per-poll
progress output and detecting visible-state changes when draining a
``PoolPendingBackend``. They exist so demo and CLI scripts do not have
to hand-roll the same tuple construction and string formatting at every
call site.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from dr_llm.pool.pending.backend import PoolPendingBackendState
from dr_llm.workers.models import WorkerSnapshot
from dr_llm.workers.worker_controller import WorkerController

ProgressKey = tuple[int, int, int, int, int]

_UNKNOWN = -1


def format_pool_progress_line(
    snapshot: WorkerSnapshot[PoolPendingBackendState],
) -> str:
    """Render a single-line progress summary for a pool worker poll.

    Format: ``"claimed=X completed=Y failed=Z pending=P leased=L"``.

    When ``snapshot.backend_state`` is ``None`` (the first poll can
    arrive before the backend has reported), the queue fields are
    rendered as ``"?"`` rather than raising.
    """
    worker_counts = snapshot.counts
    backend_state = snapshot.backend_state
    if backend_state is None:
        pending: int | str = "?"
        leased: int | str = "?"
    else:
        pending = backend_state.status_counts.pending
        leased = backend_state.status_counts.leased
    return (
        f"claimed={worker_counts.claimed} "
        f"completed={worker_counts.completed} "
        f"failed={worker_counts.failed} "
        f"pending={pending} "
        f"leased={leased}"
    )


def pool_progress_key(
    snapshot: WorkerSnapshot[PoolPendingBackendState],
) -> ProgressKey:
    """Return a hashable key over the visible progress fields.

    Use this in polling loops to decide whether anything user-visible
    has changed since the previous poll. Only the five fields rendered
    by :func:`format_pool_progress_line` are included, so noisy fields
    like ``WorkerStatCounts.idle_polls`` (which ticks up on every empty
    poll) do not trigger spurious change events.

    When ``snapshot.backend_state`` is ``None``, the queue slots are
    returned as ``-1`` sentinels so the first populated snapshot will
    register as a change.
    """
    worker_counts = snapshot.counts
    backend_state = snapshot.backend_state
    if backend_state is None:
        return (
            worker_counts.claimed,
            worker_counts.completed,
            worker_counts.failed,
            _UNKNOWN,
            _UNKNOWN,
        )
    status_counts = backend_state.status_counts
    return (
        worker_counts.claimed,
        worker_counts.completed,
        worker_counts.failed,
        status_counts.pending,
        status_counts.leased,
    )


def pool_is_idle(
    snapshot: WorkerSnapshot[PoolPendingBackendState],
) -> bool:
    """Return ``True`` iff the pool has no in-flight work.

    Returns ``False`` when ``snapshot.backend_state`` is ``None`` so
    callers continue polling until the backend has actually reported
    its first state.
    """
    backend_state = snapshot.backend_state
    if backend_state is None:
        return False
    return backend_state.status_counts.in_flight == 0


def drain(
    controller: WorkerController[PoolPendingBackendState],
    *,
    on_change: Callable[[WorkerSnapshot[PoolPendingBackendState]], None] | None = None,
    poll_interval_s: float = 0.5,
) -> WorkerSnapshot[PoolPendingBackendState]:
    """Block until the controller's backend reports no in-flight work.

    Polls the controller every ``poll_interval_s`` seconds and returns
    the first snapshot for which :func:`pool_is_idle` is true. When
    ``on_change`` is provided, it is called with the latest snapshot
    only when :func:`pool_progress_key` differs from the previous poll
    — so callers get one notification per visible-state change rather
    than one per poll. Note: this does *not* call ``controller.stop()``
    or ``controller.join()``; the caller still owns shutdown.
    """
    if poll_interval_s <= 0:
        raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")
    last_key: ProgressKey | None = None
    while True:
        snapshot = controller.snapshot()
        key = pool_progress_key(snapshot)
        if on_change is not None and key != last_key:
            on_change(snapshot)
            last_key = key
        if pool_is_idle(snapshot):
            return snapshot
        time.sleep(poll_interval_s)

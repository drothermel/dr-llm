"""Progress helpers for polling a pool-backed WorkerController.

These helpers are the single source of truth for formatting per-poll
progress output and detecting visible-state changes when draining a
``PoolPendingBackend``. They exist so demo and CLI scripts do not have
to hand-roll the same tuple construction and string formatting at every
call site.
"""

from __future__ import annotations

from dr_llm.pool.pending.backend import PoolPendingBackendState
from dr_llm.workers.models import WorkerSnapshot

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

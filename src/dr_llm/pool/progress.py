from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from dr_llm.workers import WorkerController, WorkerSnapshot, drain_until


def format_worker_progress_line[TBackendState: BaseModel](
    snapshot: WorkerSnapshot[TBackendState],
) -> str:
    """Format common worker and pool progress fields as a compact line."""
    counts = snapshot.counts
    parts = [
        f"claimed={counts.claimed}",
        f"completed={counts.completed}",
        f"failed={counts.failed}",
        f"retried={counts.retried}",
        f"errors={counts.process_errors}",
    ]
    backend_state = snapshot.backend_state
    if backend_state is not None:
        state = backend_state.model_dump()
        if "incomplete" in state:
            parts.append(f"incomplete={state['incomplete']}")
        if "complete" in state:
            parts.append(f"complete={state['complete']}")
        if "leased" in state:
            parts.append(f"leased={state['leased']}")
    return " | ".join(parts)


def pool_is_idle[TBackendState: BaseModel](
    snapshot: WorkerSnapshot[TBackendState],
) -> bool:
    """Return whether a pool-style backend snapshot has no incomplete work."""
    backend_state = snapshot.backend_state
    if backend_state is None:
        return False
    state = backend_state.model_dump()
    if "incomplete" not in state:
        return False
    return int(state["incomplete"] or 0) == 0


def drain_pool[TBackendState: BaseModel](
    controller: WorkerController[TBackendState],
    *,
    on_change: Callable[[WorkerSnapshot[TBackendState]], None] | None = None,
    poll_interval_s: float = 1.0,
) -> WorkerSnapshot[TBackendState]:
    """Drain a pool-backed worker controller until no incomplete work remains."""
    return drain_until(
        controller,
        is_done=pool_is_idle,
        on_change=on_change,
        snapshot_key=format_worker_progress_line,
        poll_interval_s=poll_interval_s,
    )

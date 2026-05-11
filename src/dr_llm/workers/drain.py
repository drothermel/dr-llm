from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel

from dr_llm.workers.models import WorkerSnapshot
from dr_llm.workers.worker_controller import WorkerController


def drain_until[TBackendState: BaseModel](
    controller: WorkerController[TBackendState],
    *,
    is_done: Callable[[WorkerSnapshot[TBackendState]], bool],
    on_change: Callable[[WorkerSnapshot[TBackendState]], None] | None = None,
    snapshot_key: Callable[[WorkerSnapshot[TBackendState]], object]
    | None = None,
    poll_interval_s: float = 1.0,
) -> WorkerSnapshot[TBackendState]:
    """Poll a worker controller until ``is_done`` returns true, then stop it."""
    if poll_interval_s <= 0:
        raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")

    missing_key = object()
    last_key: object = missing_key
    cleanup_needed = True
    try:
        while True:
            snapshot = controller.snapshot()
            current_key = (
                snapshot if snapshot_key is None else snapshot_key(snapshot)
            )
            if on_change is not None and current_key != last_key:
                on_change(snapshot)
                last_key = current_key
            if is_done(snapshot):
                controller.stop()
                cleanup_needed = False
                return controller.join()
            time.sleep(poll_interval_s)
    finally:
        if cleanup_needed:
            controller.stop()
            controller.join()

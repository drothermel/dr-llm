#!/usr/bin/env python3
"""Demo: seed a pool's pending queue and fill it with generic workers.

Usage:
  uv run python scripts/demo-pool-fill.py

The demo:
  - Creates a simple pool keyed by (model, prompt)
  - Seeds the pending queue with the full cartesian product times N samples
  - Starts background workers using the generic pool-fill API
  - Prints progress until the queued work is complete
"""

from __future__ import annotations

import time
from typing import Annotated

import typer

from dr_llm.pool import (
    DbConfig,
    DbRuntime,
    KeyColumn,
    PoolSchema,
    PoolStore,
    seed_pending,
    start_workers,
)
from dr_llm.pool.sample_models import PendingSample, WorkerSnapshot

app = typer.Typer()


def _demo_process_fn(sample: PendingSample) -> dict[str, str | int]:
    return {
        "completion": (
            f"completion::{sample.key_values['model']}::{sample.key_values['prompt']}::"
            f"{sample.sample_idx}"
        ),
        "sample_idx": sample.sample_idx,
    }


def _print_progress(snapshot: WorkerSnapshot) -> None:
    counts = snapshot.status_counts
    print(
        "Progress: "
        f"claimed={snapshot.claimed} "
        f"promoted={snapshot.promoted} "
        f"failed={snapshot.failed} "
        f"pending={counts.pending} "
        f"leased={counts.leased}"
    )


@app.command()
def main(
    dsn: Annotated[
        str | None,
        typer.Option(help="Optional PostgreSQL DSN. Defaults to DR_LLM_DATABASE_URL."),
    ] = None,
    pool_name: Annotated[
        str,
        typer.Option(help="Pool name to create for the demo."),
    ] = "demo_pool_fill",
    num_workers: Annotated[
        int,
        typer.Option(help="Number of concurrent workers to run."),
    ] = 4,
    samples_per_cell: Annotated[
        int,
        typer.Option(help="Number of samples to queue for each (model, prompt) cell."),
    ] = 3,
    models: Annotated[
        list[str],
        typer.Option(help="Model values for the cartesian key grid."),
    ] = ["gpt-4.1-mini", "gpt-5-mini"],
    prompts: Annotated[
        list[str],
        typer.Option(help="Prompt values for the cartesian key grid."),
    ] = ["math", "history"],
) -> None:
    schema = PoolSchema(
        name=pool_name,
        key_columns=[KeyColumn(name="model"), KeyColumn(name="prompt")],
    )
    runtime = DbRuntime(DbConfig() if dsn is None else DbConfig(dsn=dsn))
    store = PoolStore(schema, runtime)
    controller = None

    try:
        store.init_schema()
        seed_result = seed_pending(
            store,
            key_grid={"model": models, "prompt": prompts},
            n=samples_per_cell,
        )
        print(
            f"Seeded {seed_result.inserted} pending rows"
            f" (skipped {seed_result.skipped} existing rows)"
        )

        controller = start_workers(
            store,
            process_fn=_demo_process_fn,
            num_workers=num_workers,
            min_poll_interval_s=0.05,
            max_poll_interval_s=0.25,
        )
        try:
            last_progress: tuple[int, int, int, int, int] | None = None
            while True:
                snapshot = controller.snapshot()
                current_progress = (
                    snapshot.claimed,
                    snapshot.promoted,
                    snapshot.failed,
                    snapshot.status_counts.pending,
                    snapshot.status_counts.leased,
                )
                if current_progress != last_progress:
                    _print_progress(snapshot)
                    last_progress = current_progress
                if (
                    snapshot.status_counts.pending == 0
                    and snapshot.status_counts.leased == 0
                ):
                    break
                time.sleep(0.05)
        finally:
            controller.stop()
            final_snapshot = controller.join()

        final_counts = final_snapshot.status_counts
        print(
            "Final queue counts: "
            f"pending={final_counts.pending} "
            f"leased={final_counts.leased} "
            f"promoted={final_counts.promoted} "
            f"failed={final_counts.failed}"
        )

        coverage = store.coverage()
        total_samples = sum(row.count for row in coverage)
        print(f"Stored {total_samples} samples across {len(coverage)} cells")
    finally:
        if controller is not None:
            controller.stop()
            controller.join()
        runtime.close()


if __name__ == "__main__":
    app()

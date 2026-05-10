#!/usr/bin/env python3
"""Demo: seed a pool with LLM configs and prompts, fill it with real provider calls.

Usage:
  uv run python scripts/demo-pool-fill.py
  uv run python scripts/demo-pool-fill.py --dsn postgresql://postgres:postgres@localhost:5433/dr_llm_test

Prerequisites:
  1. OpenAI and Google API keys for the default LLM configs
  2. Docker running, unless --dsn points at an existing Postgres database

When no --dsn is provided, the script auto-creates a temporary Docker-managed
Postgres project, runs the demo, and destroys it on exit.

The demo:
  - Uses shared reasoning-valid LlmConfig instances for OpenAI and Google
  - Defines short prompts as Message lists
  - Seeds the (llm_config x prompt) cross product into the samples table
    using ``seed_llm_grid``
  - Starts background workers using ``make_llm_process_fn`` (real LLM calls)
  - Drains incomplete samples, printing progress on visible state changes
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated

import typer

from dr_llm.demo import (
    create_demo_project,
    DemoCounts,
    DemoPrompts,
    POOL_PROGRESS_FIELDS,
    demo_pool_fill_llm_configs,
    ensure_docker_available,
    require_demo_project_dsn,
    temporary_demo_project_name,
)
from dr_llm.llm import (
    LlmConfig,
    Message,
    build_default_registry,
)
from dr_llm.project import destroy_project
from dr_llm.pool import (
    Axis,
    AxisMember,
    DbConfig,
    DbRuntime,
    GridCell,
    LlmPoolBackend,
    LlmPoolBackendConfig,
    LlmPoolBackendState,
    PoolSchema,
    PoolStore,
    make_llm_process_fn,
    seed_llm_grid,
)
from dr_llm.workers import (
    WorkerConfig,
    WorkerController,
    WorkerSnapshot,
    start_workers,
)

app = typer.Typer()

PROMPTS: dict[str, list[Message]] = {
    "haiku": [Message(role="user", content=DemoPrompts.PROGRAMMING_HAIKU)],
    "math": [Message(role="user", content=DemoPrompts.TWO_PLUS_TWO)],
}


def _pool_is_idle(snapshot: WorkerSnapshot[LlmPoolBackendState]) -> bool:
    backend_state = snapshot.backend_state
    return backend_state is not None and backend_state.incomplete == 0


def _drain(
    controller: WorkerController[LlmPoolBackendState],
    *,
    on_change: Callable[[DemoCounts], None] | None = None,
    poll_interval_s: float = 0.5,
) -> WorkerSnapshot[LlmPoolBackendState]:
    if poll_interval_s <= 0:
        raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")
    last_counts: DemoCounts | None = None
    while True:
        snapshot = controller.snapshot()
        counts = DemoCounts.from_pool_snapshot(snapshot)
        if on_change is not None and counts.changed_from(
            last_counts, POOL_PROGRESS_FIELDS
        ):
            on_change(counts)
            last_counts = counts
        if _pool_is_idle(snapshot):
            return snapshot
        time.sleep(poll_interval_s)


def _llm_config_axis() -> Axis[LlmConfig]:
    llm_configs = demo_pool_fill_llm_configs()
    return Axis(
        name="llm_config",
        members=[
            AxisMember[LlmConfig](
                id=cfg_id,
                value=cfg,
                metadata={"provider": cfg.provider, "model": cfg.model},
            )
            for cfg_id, cfg in llm_configs.items()
        ],
    )


def _prompt_axis() -> Axis[list[Message]]:
    return Axis(
        name="prompt",
        members=[
            AxisMember[list[Message]](
                id=prompt_id,
                value=messages,
                metadata={"first_user_text": messages[0].content},
            )
            for prompt_id, messages in PROMPTS.items()
        ],
    )


def _build_request(cell: GridCell) -> tuple[list[Message], LlmConfig]:
    return cell.values["prompt"], cell.values["llm_config"]


def _run_demo(
    dsn: str,
    pool_name: str,
    num_workers: int,
    samples_per_cell: int,
) -> None:
    schema = PoolSchema.from_axis_names(pool_name, ["llm_config", "prompt"])
    runtime = DbRuntime(DbConfig(dsn=dsn))
    registry = build_default_registry()
    store = PoolStore(schema, runtime)
    store.ensure_schema()
    controller = None

    try:
        seed_result = seed_llm_grid(
            store,
            axes=[_llm_config_axis(), _prompt_axis()],
            build_request=_build_request,
            n=samples_per_cell,
        )
        print(
            f"Seeded {seed_result.inserted} sample rows"
            f" (skipped {seed_result.skipped} existing rows)"
        )

        controller = start_workers(
            LlmPoolBackend(
                store,
                config=LlmPoolBackendConfig(max_retries=1),
            ),
            process_fn=make_llm_process_fn(registry),
            config=WorkerConfig(
                num_workers=num_workers,
                min_poll_interval_s=0.5,
                max_poll_interval_s=3.0,
                thread_name_prefix="pool-fill",
            ),
        )
        try:
            _drain(
                controller,
                on_change=lambda counts: print(
                    f"Progress: {counts.format_line(POOL_PROGRESS_FIELDS)}"
                ),
            )
        finally:
            controller.stop()
            final_snapshot = controller.join()

        assert final_snapshot.backend_state is not None
        print(
            "Final sample counts: "
            f"incomplete={final_snapshot.backend_state.incomplete} "
            f"complete={final_snapshot.backend_state.complete}"
        )

        print(f"Stored {store.sample_count()} samples")

        samples = store.bulk_load()
        for sample in samples[:4]:
            response = sample.response or {}
            text = str(response.get("text", ""))[:80]
            print(
                f"  [{sample.key_values['llm_config']}] "
                f"[{sample.key_values['prompt']}] "
                f"-> {text!r}"
            )
    finally:
        registry.close()
        runtime.close()


@app.command()
def main(
    dsn: Annotated[
        str | None,
        typer.Option(
            help="PostgreSQL DSN. If omitted, a temporary Docker project is created."
        ),
    ] = None,
    pool_name: Annotated[
        str,
        typer.Option(help="Pool name to create for the demo."),
    ] = "demo_pool_fill",
    num_workers: Annotated[
        int,
        typer.Option(help="Number of concurrent workers to run."),
    ] = 2,
    samples_per_cell: Annotated[
        int,
        typer.Option(
            help="Number of samples to queue for each (llm_config, prompt) cell."
        ),
    ] = 1,
) -> None:
    """Seed an LLM config x prompt pool and fill it with worker calls."""
    temporary_project_name: str | None = None
    try:
        if dsn is None:
            ensure_docker_available(
                reason=(
                    "This demo creates a temporary Postgres project when no "
                    "--dsn is provided."
                ),
                recovery_hint=(
                    "Install Docker, start the daemon, or pass --dsn to use "
                    "an existing database."
                ),
            )

            project_name = temporary_demo_project_name("demo_pool_fill")
            print(f"Creating temporary project '{project_name}'...")
            project = create_demo_project(project_name)
            temporary_project_name = project_name
            dsn = require_demo_project_dsn(project)
            print(f"Postgres ready at {dsn}")

        _run_demo(
            dsn,
            pool_name,
            num_workers,
            samples_per_cell,
        )
    finally:
        if temporary_project_name is not None:
            print(
                f"Destroying temporary project '{temporary_project_name}'..."
            )
            destroy_project(temporary_project_name)


if __name__ == "__main__":
    app()

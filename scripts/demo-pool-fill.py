#!/usr/bin/env python3
"""Demo: seed a pool with LLM configs and prompts, fill it with real provider calls.

Usage:
  uv run python scripts/demo-pool-fill.py
  uv run python scripts/demo-pool-fill.py --keep-project
  uv run python scripts/demo-pool-fill.py --dsn postgresql://postgres:postgres@localhost:5433/dr_llm_test

Prerequisites:
  1. OpenAI and Google API keys for the default LLM configs
  2. Docker running, unless --dsn points at an existing Postgres database

When no --dsn is provided, the script auto-creates a temporary Docker-managed
Postgres project. By default the project is destroyed on exit; pass
--keep-project to inspect the pool afterward.

The demo:
  - Uses shared reasoning-valid LlmConfig instances for OpenAI and Google
  - Defines short prompts as Message lists
  - Seeds the (llm_config x prompt) cross product into the samples table
    using ``seed_llm_grid``
  - Starts background workers using ``make_llm_process_fn`` (real LLM calls)
  - Drains incomplete samples, printing progress on visible state changes
"""

from __future__ import annotations

from typing import Annotated

import typer

from dr_llm.demo import (
    DemoCounts,
    DemoPrompts,
    POOL_PROGRESS_FIELDS,
    cleanup_demo_dsn,
    command_hint,
    demo_pool_fill_llm_configs,
    prepare_demo_dsn,
)
from dr_llm.llm import (
    LlmConfig,
    Message,
    build_default_registry,
)
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
    drain_pool,
    make_llm_process_fn,
    seed_llm_grid,
)
from dr_llm.workers import (
    WorkerSnapshot,
    WorkerConfig,
    start_workers,
)

app = typer.Typer()

PROMPTS: dict[str, list[Message]] = {
    "haiku": [Message(role="user", content=DemoPrompts.PROGRAMMING_HAIKU)],
    "math": [Message(role="user", content=DemoPrompts.TWO_PLUS_TWO)],
}

LLM_CONFIG_AXIS: Axis[LlmConfig] = Axis(
    name="llm_config",
    members=[
        AxisMember[LlmConfig](
            id=cfg_id,
            value=cfg,
            metadata={"provider": cfg.provider, "model": cfg.model},
        )
        for cfg_id, cfg in demo_pool_fill_llm_configs().items()
    ],
)

PROMPT_AXIS: Axis[list[Message]] = Axis(
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


def _seed_fill_pool(
    dsn: str,
    pool_name: str,
    num_workers: int,
    samples_per_cell: int,
) -> None:
    def _build_request(cell: GridCell) -> tuple[list[Message], LlmConfig]:
        return cell.values["prompt"], cell.values["llm_config"]

    def _snapshot_progress(
        snapshot: WorkerSnapshot[LlmPoolBackendState],
    ) -> None:
        counts = DemoCounts.from_pool_snapshot(snapshot)
        print(f"Progress: {counts.format_line(POOL_PROGRESS_FIELDS)}")

    schema = PoolSchema.from_axis_names(pool_name, ["llm_config", "prompt"])
    runtime = DbRuntime(DbConfig(dsn=dsn))
    registry = build_default_registry()
    store = PoolStore(schema, runtime)
    store.ensure_schema()

    try:
        seed_result = seed_llm_grid(
            store,
            axes=[LLM_CONFIG_AXIS, PROMPT_AXIS],
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
        final_snapshot = drain_pool(
            controller,
            on_change=_snapshot_progress,
            poll_interval_s=0.5,
        )

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


def _ensure_dsn_and_seed_fill_pool(
    *,
    dsn: str | None,
    project_name: str | None,
    keep_project: bool,
    pool_name: str,
    num_workers: int,
    samples_per_cell: int,
) -> None:
    lease = prepare_demo_dsn(
        dsn=dsn,
        project_prefix="demo_pool_fill",
        project_name=project_name,
        keep_project=keep_project,
    )
    if lease.project_name is not None:
        print(f"Postgres ready at {lease.dsn}")

    try:
        _seed_fill_pool(
            lease.dsn,
            pool_name,
            num_workers,
            samples_per_cell,
        )
    finally:
        if lease.should_destroy_project and lease.project_name is not None:
            print(f"Destroying temporary project '{lease.project_name}'...")
            cleanup_demo_dsn(lease)

    if lease.project_name is not None and not lease.should_destroy_project:
        print(
            f"Project '{lease.project_name}' is still running with your data."
        )
        command_hint(
            "Destroy permanently",
            "uv run dr-llm project destroy "
            f"{lease.project_name} --yes-really-delete-everything",
        )


@app.command()
def main(
    dsn: Annotated[
        str | None,
        typer.Option(
            help=(
                "PostgreSQL DSN. If omitted, a Docker demo project is created."
            )
        ),
    ] = None,
    project_name: Annotated[
        str | None,
        typer.Option(
            help=(
                "Name for the auto-created Docker project. Defaults to a "
                "unique temporary name."
            )
        ),
    ] = None,
    keep_project: Annotated[
        bool,
        typer.Option(
            "--keep-project",
            help="Keep the auto-created Docker project for inspection.",
        ),
    ] = False,
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
    _ensure_dsn_and_seed_fill_pool(
        dsn=dsn,
        project_name=project_name,
        keep_project=keep_project,
        pool_name=pool_name,
        num_workers=num_workers,
        samples_per_cell=samples_per_cell,
    )


if __name__ == "__main__":
    app()

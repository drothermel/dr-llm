#!/usr/bin/env python3
"""Demo: seed a pool with LLM configs and prompts, fill it with real provider calls.

Usage:
  uv run python scripts/demo-pool-fill.py
  uv run python scripts/demo-pool-fill.py --dsn postgresql://postgres:postgres@localhost:5433/dr_llm_test

When no --dsn is provided, the script auto-creates a Docker-managed Postgres
project via project_service, runs the demo, and destroys it on exit.

The demo:
  - Defines reasoning-valid LlmConfig instances for OpenAI and Google
  - Defines short prompts as Message lists
  - Seeds the (llm_config x prompt) cross product into the pending queue
    using ``seed_llm_grid``
  - Shuffles the pending priorities so workers interleave across providers
    instead of draining the queue in cross-product order
  - Starts background workers using ``make_llm_process_fn`` (real LLM calls)
  - Drains the queue with ``drain``, printing progress on visible state changes
"""

from __future__ import annotations

import shutil
from typing import Annotated
from uuid import uuid4

import typer

from dr_llm.llm.config import LlmConfig
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import (
    GoogleReasoning,
    OpenAIReasoning,
    ThinkingLevel,
)
from dr_llm.llm.providers.registry import build_default_registry
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.llm_pool_adapter import make_llm_process_fn, seed_llm_grid
from dr_llm.pool.pending.backend import (
    PoolPendingBackend,
    PoolPendingBackendConfig,
)
from dr_llm.pool.pending.grid import Axis, AxisMember, GridCell
from dr_llm.pool.pending.progress import drain, format_pool_progress_line
from dr_llm.pool.pool_store import PoolStore
from dr_llm.project.project_info import ProjectInfo
from dr_llm.project.project_service import create_project, destroy_project
from dr_llm.workers import WorkerConfig, start_workers

app = typer.Typer()

LLM_CONFIGS: dict[str, LlmConfig] = {
    "gpt-5-mini-low": LlmConfig(
        provider="openai",
        model="gpt-5-mini",
        max_tokens=64,
        reasoning=OpenAIReasoning(thinking_level=ThinkingLevel.LOW),
    ),
    "gemini-flash-budget": LlmConfig(
        provider="google",
        model="gemini-2.5-flash",
        max_tokens=64,
        reasoning=GoogleReasoning(
            thinking_level=ThinkingLevel.BUDGET,
            budget_tokens=512,
        ),
    ),
}

PROMPTS: dict[str, list[Message]] = {
    "haiku": [Message(role="user", content="Write a haiku about programming.")],
    "math": [
        Message(role="user", content="What is 17 * 23? Reply with just the number.")
    ],
}


def _llm_config_axis() -> Axis[LlmConfig]:
    return Axis(
        name="llm_config",
        members=[
            AxisMember[LlmConfig](
                id=cfg_id,
                value=cfg,
                metadata={"provider": cfg.provider, "model": cfg.model},
            )
            for cfg_id, cfg in LLM_CONFIGS.items()
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
    shuffle: bool,
    shuffle_seed: int | None,
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
            f"Seeded {seed_result.inserted} pending rows"
            f" (skipped {seed_result.skipped} existing rows)"
        )

        if shuffle:
            shuffled = store.pending.shuffle_priorities(seed=shuffle_seed)
            seed_note = f" (seed={shuffle_seed})" if shuffle_seed is not None else ""
            print(f"Shuffled priorities for {shuffled} pending rows{seed_note}")

        controller = start_workers(
            PoolPendingBackend(
                store,
                config=PoolPendingBackendConfig(max_retries=1),
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
            drain(
                controller,
                on_change=lambda snap: print(
                    f"Progress: {format_pool_progress_line(snap)}"
                ),
            )
        finally:
            controller.stop()
            final_snapshot = controller.join()

        assert final_snapshot.backend_state is not None
        final_counts = final_snapshot.backend_state.status_counts
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

        samples = store.bulk_load()
        for sample in samples[:4]:
            text = sample.payload.get("text", "")[:80]
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
    shuffle: Annotated[
        bool,
        typer.Option(
            "--shuffle/--no-shuffle",
            help=(
                "Shuffle pending priorities after seeding so workers "
                "interleave across providers instead of draining the queue "
                "in cross-product order."
            ),
        ),
    ] = True,
    shuffle_seed: Annotated[
        int | None,
        typer.Option(
            help="Optional seed for reproducible shuffles. Ignored when --no-shuffle."
        ),
    ] = None,
) -> None:
    if dsn is not None:
        _run_demo(
            dsn,
            pool_name,
            num_workers,
            samples_per_cell,
            shuffle,
            shuffle_seed,
        )
        return

    # Auto-manage a Docker Postgres project
    if not shutil.which("docker"):
        print("Error: Docker is required when no --dsn is provided.")
        print("Either install Docker or pass --dsn to use an existing database.")
        raise typer.Exit(1)

    project_name = f"demo-pool-fill-{uuid4().hex[:8]}"
    project: ProjectInfo | None = None
    try:
        print(f"Creating temporary project '{project_name}'...")
        project = create_project(project_name)
        assert project.dsn is not None
        print(f"Postgres ready at {project.dsn}")
        _run_demo(
            project.dsn,
            pool_name,
            num_workers,
            samples_per_cell,
            shuffle,
            shuffle_seed,
        )
    finally:
        if project is not None:
            print(f"Destroying temporary project '{project_name}'...")
            destroy_project(project_name)


if __name__ == "__main__":
    app()

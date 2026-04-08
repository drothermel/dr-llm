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
  - Manually constructs PendingSample rows for the (llm_config x prompt) cross
    product, embedding serialized configs/messages in each payload
  - Inserts them via store.pending.insert_many
  - Starts background workers using make_llm_process_fn (real LLM calls)
  - Prints progress until the queued work is complete
"""

from __future__ import annotations

import shutil
import time
from itertools import product
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
from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.llm_pool_adapter import make_llm_process_fn
from dr_llm.pool.pending.backend import (
    PoolPendingBackend,
    PoolPendingBackendConfig,
)
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.progress import (
    ProgressKey,
    format_pool_progress_line,
    pool_is_idle,
    pool_progress_key,
)
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


def _build_pending_samples(
    *,
    llm_configs: dict[str, LlmConfig],
    prompts: dict[str, list[Message]],
    samples_per_cell: int,
) -> list[PendingSample]:
    """Build PendingSample rows for the (llm_config x prompt) cross product.

    Each pending row stores its key IDs in ``key_values`` and the serialized
    LlmConfig + messages in ``payload``, which is the shape that
    ``make_llm_process_fn`` expects.
    """
    samples: list[PendingSample] = []
    for cfg_id, prompt_id in product(llm_configs, prompts):
        payload = {
            "llm_config": llm_configs[cfg_id].model_dump(mode="json"),
            "prompt": [m.model_dump(mode="json") for m in prompts[prompt_id]],
        }
        for sample_idx in range(samples_per_cell):
            samples.append(
                PendingSample(
                    key_values={"llm_config": cfg_id, "prompt": prompt_id},
                    sample_idx=sample_idx,
                    payload=payload,
                )
            )
    return samples


def _run_demo(
    dsn: str, pool_name: str, num_workers: int, samples_per_cell: int
) -> None:
    schema = PoolSchema(
        name=pool_name,
        key_columns=[KeyColumn(name="llm_config"), KeyColumn(name="prompt")],
    )
    runtime = DbRuntime(DbConfig(dsn=dsn))
    registry = build_default_registry()
    store = PoolStore(schema, runtime)
    store.ensure_schema()
    controller = None

    try:
        pending_samples = _build_pending_samples(
            llm_configs=LLM_CONFIGS,
            prompts=PROMPTS,
            samples_per_cell=samples_per_cell,
        )
        seed_result = store.pending.insert_many(
            pending_samples, ignore_conflicts=True
        )
        print(
            f"Seeded {seed_result.inserted} pending rows"
            f" (skipped {seed_result.skipped} existing rows)"
        )

        process_fn = make_llm_process_fn(registry)
        controller = start_workers(
            PoolPendingBackend(
                store,
                config=PoolPendingBackendConfig(max_retries=1),
            ),
            process_fn=process_fn,
            config=WorkerConfig(
                num_workers=num_workers,
                min_poll_interval_s=0.5,
                max_poll_interval_s=3.0,
                thread_name_prefix="pool-fill",
            ),
        )
        try:
            last_key: ProgressKey | None = None
            while True:
                snapshot = controller.snapshot()
                key = pool_progress_key(snapshot)
                if key != last_key:
                    print(f"Progress: {format_pool_progress_line(snapshot)}")
                    last_key = key
                if pool_is_idle(snapshot):
                    break
                time.sleep(0.5)
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
) -> None:
    if dsn is not None:
        _run_demo(dsn, pool_name, num_workers, samples_per_cell)
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
        _run_demo(project.dsn, pool_name, num_workers, samples_per_cell)
    finally:
        if project is not None:
            print(f"Destroying temporary project '{project_name}'...")
            destroy_project(project_name)


if __name__ == "__main__":
    app()

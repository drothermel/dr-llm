#!/usr/bin/env python3
"""Demo: sync a local project into a Postgres-compatible target database.

Demonstrates the project sync primitive without requiring Neon credentials:
- Create one Docker project as the source and admin server.
- Seed a small typed pool in the source project.
- Run `dr-llm project sync-postgres` into a separate database.
- Read the synced pool from the target database.

Prerequisites:
  1. Docker running.
  2. `psql` available on PATH for the restore step.

Usage:
  uv run python scripts/demo-project-sync-postgres.py
  uv run python scripts/demo-project-sync-postgres.py --keep-projects
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import typer

from dr_llm.demo import (
    cleanup_demo_dsn,
    command_hint,
    header,
    ok,
    prepare_demo_dsn,
    run_dr_llm_streaming,
    step,
)
from dr_llm.pool import (
    DbConfig,
    DbRuntime,
    KeyColumn,
    PoolReader,
    PoolSample,
    PoolSchema,
    PoolStore,
)

app = typer.Typer()

PROJECT_PREFIX = "demo_sync"
POOL_SCHEMA = PoolSchema(
    name="sync_demo_pool",
    key_columns=[KeyColumn(name="case_id")],
)


def _database_url(base_url: str, database_name: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            f"/{database_name}",
            parts.query,
            parts.fragment,
        )
    )


def _seed_source_pool(dsn: str) -> str:
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=4))
    try:
        store = PoolStore(POOL_SCHEMA, runtime)
        store.ensure_schema()
        sample = PoolSample(
            key_values={"case_id": "alpha"},
            request={"prompt": "sync demo prompt"},
            response={"text": "sync demo response"},
            finish_reason="stop",
        )
        store.insert_sample(sample)
        return sample.sample_id
    finally:
        runtime.close()


def _verify_target_pool(dsn: str, expected_sample_id: str) -> None:
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=4))
    try:
        reader = PoolReader.open(POOL_SCHEMA.name, runtime=runtime)
        samples = reader.samples_list()
    finally:
        runtime.close()

    if len(samples) != 1:
        raise RuntimeError(f"Expected 1 synced sample, found {len(samples)}.")
    [sample] = samples
    if sample.sample_id != expected_sample_id:
        raise RuntimeError(
            "Synced sample ID mismatch: "
            f"expected {expected_sample_id}, found {sample.sample_id}."
        )
    if sample.response != {"text": "sync demo response"}:
        raise RuntimeError(f"Unexpected synced response: {sample.response!r}.")


@app.command()
def main(
    keep_projects: bool = typer.Option(
        False,
        "--keep-projects",
        help="Keep the temporary source project for inspection.",
    ),
) -> None:
    header("Project Postgres Sync Demo")
    suffix = uuid4().hex[:8]
    source_project = f"{PROJECT_PREFIX}_{suffix}"
    target_database = f"{PROJECT_PREFIX}_target_{suffix}"

    source_lease = prepare_demo_dsn(
        dsn=None,
        project_prefix=PROJECT_PREFIX,
        project_name=source_project,
        keep_project=keep_projects,
        docker_reason=(
            "This demo creates a source Postgres project for sync input."
        ),
    )

    try:
        step("Seed source project")
        sample_id = _seed_source_pool(source_lease.dsn)
        ok(f"created source sample {sample_id}")

        step("Sync source project to target Postgres database")
        run_dr_llm_streaming(
            "project",
            "sync-postgres",
            source_project,
            "--admin-url",
            source_lease.dsn,
            "--target-database",
            target_database,
            "--drop-previous",
        )

        step("Verify synced target database")
        target_dsn = _database_url(source_lease.dsn, target_database)
        _verify_target_pool(target_dsn, sample_id)
        ok("target pool contains the synced sample")

        if keep_projects:
            command_hint(
                "inspect source",
                f"uv run dr-llm project use {source_project}",
            )
            command_hint(
                "inspect target",
                f"DR_LLM_DATABASE_URL='{target_dsn}' uv run dr-llm pool inspect-dsn "
                f"{POOL_SCHEMA.name}",
            )
    finally:
        if keep_projects:
            ok("kept demo projects for inspection")
        else:
            cleanup_demo_dsn(source_lease)
            ok("removed demo project")


if __name__ == "__main__":
    app()

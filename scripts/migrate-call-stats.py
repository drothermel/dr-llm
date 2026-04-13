#!/usr/bin/env python3
"""Create call_stats tables for existing pools and optionally backfill from payload_json.

Usage:
  uv run python scripts/migrate-call-stats.py                    # all pools, create only
  uv run python scripts/migrate-call-stats.py --backfill         # all pools, create + backfill
  uv run python scripts/migrate-call-stats.py --pool mypool      # specific pool only
  uv run python scripts/migrate-call-stats.py --dry-run          # show what would be done

Intended as a one-off migration tool. Safe to re-run: tables use
CREATE IF NOT EXISTS, and backfill uses ON CONFLICT DO NOTHING.
"""

from __future__ import annotations

import re
from os import getenv
from typing import Annotated

import typer
from sqlalchemy import text as sa_text

from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import KeyColumn, PoolSchema
from dr_llm.pool.db.tables import PoolTables

app = typer.Typer(add_completion=False)

_FIXED_SAMPLE_COLUMNS = frozenset(
    {
        "sample_id",
        "sample_idx",
        "payload_json",
        "source_run_id",
        "metadata_json",
        "status",
        "created_at",
    }
)

_POOL_TABLE_RE = re.compile(r"^pool_(.+)_samples$")


def _discover_pools(runtime: DbRuntime) -> list[str]:
    """Find pool names by matching table names against pool_{name}_samples."""
    with runtime.connect() as conn:
        rows = conn.execute(
            sa_text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "AND table_name LIKE 'pool\\_%\\_samples' "
                "ORDER BY table_name"
            )
        ).fetchall()
    names: list[str] = []
    for (table_name,) in rows:
        m = _POOL_TABLE_RE.match(table_name)
        if m:
            names.append(m.group(1))
    return names


def _discover_key_columns(runtime: DbRuntime, pool_name: str) -> list[str]:
    """Infer key column names from the samples table structure."""
    samples_table = f"pool_{pool_name}_samples"
    with runtime.connect() as conn:
        rows = conn.execute(
            sa_text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "AND table_name = :table_name "
                "ORDER BY ordinal_position"
            ),
            {"table_name": samples_table},
        ).fetchall()
    all_columns = [row[0] for row in rows]
    return [c for c in all_columns if c not in _FIXED_SAMPLE_COLUMNS]


def _backfill_pool(
    runtime: DbRuntime, pool_name: str, key_columns: list[str]
) -> int:
    """Backfill call_stats from existing sample payloads + pending attempt_count."""
    samples_table = f"pool_{pool_name}_samples"
    pending_table = f"pool_{pool_name}_pending"
    call_stats_table = f"pool_{pool_name}_call_stats"

    key_join_clauses = " AND ".join(
        f"p.{col} = s.{col}" for col in key_columns
    )
    pending_join = (
        f"LEFT JOIN {pending_table} p "
        f"ON p.status = 'promoted' "
        f"AND p.sample_idx = s.sample_idx "
        f"AND {key_join_clauses}"
    )

    sql = f"""
        INSERT INTO {call_stats_table}
            (sample_id, latency_ms, total_cost_usd, prompt_tokens,
             completion_tokens, reasoning_tokens, total_tokens,
             attempt_count, finish_reason)
        SELECT
            s.sample_id,
            COALESCE((s.payload_json->>'latency_ms')::integer, 0),
            (s.payload_json->'cost'->>'total_cost_usd')::double precision,
            COALESCE((s.payload_json->'usage'->>'prompt_tokens')::integer, 0),
            COALESCE((s.payload_json->'usage'->>'completion_tokens')::integer, 0),
            NULLIF((s.payload_json->'usage'->>'reasoning_tokens')::integer, 0),
            COALESCE((s.payload_json->'usage'->>'total_tokens')::integer, 0),
            COALESCE(p.attempt_count, 1),
            s.payload_json->>'finish_reason'
        FROM {samples_table} s
        {pending_join}
        ON CONFLICT (sample_id) DO NOTHING
    """  # noqa: S608

    with runtime.begin() as conn:
        result = conn.execute(sa_text(sql))
        return result.rowcount


def _process_pool(
    runtime: DbRuntime,
    pool_name: str,
    *,
    backfill: bool,
    dry_run: bool,
) -> None:
    key_columns = _discover_key_columns(runtime, pool_name)
    if not key_columns:
        typer.echo(f"  [skip] {pool_name}: no key columns found, skipping")
        return

    typer.echo(f"  {pool_name}: key_columns={key_columns}")

    if dry_run:
        typer.echo(f"  [dry-run] would create table pool_{pool_name}_call_stats")
        if backfill:
            typer.echo(f"  [dry-run] would backfill from pool_{pool_name}_samples")
        return

    schema = PoolSchema(
        name=pool_name,
        key_columns=[KeyColumn(name=col) for col in key_columns],
    )
    tables = PoolTables(schema)

    with runtime.begin() as conn:
        tables.sa_metadata.create_all(
            bind=conn,
            tables=[tables.call_stats],
            checkfirst=True,
        )
    typer.echo(f"  [ok] table pool_{pool_name}_call_stats ensured")

    if backfill:
        count = _backfill_pool(runtime, pool_name, key_columns)
        typer.echo(f"  [ok] backfilled {count} rows")


@app.command()
def main(
    dsn: Annotated[
        str | None,
        typer.Option(help="Database DSN (default: DR_LLM_DATABASE_URL env var)"),
    ] = None,
    pool: Annotated[
        str | None,
        typer.Option(help="Process only this pool (default: all discovered pools)"),
    ] = None,
    backfill: Annotated[
        bool,
        typer.Option(help="Backfill call_stats from existing sample payloads"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be done without making changes"),
    ] = False,
) -> None:
    """Create call_stats tables for existing pools."""
    resolved_dsn = dsn or getenv(
        "DR_LLM_DATABASE_URL", "postgresql://localhost/dr_llm"
    )
    runtime = DbRuntime(DbConfig(dsn=resolved_dsn))

    if pool:
        pool_names = [pool]
        typer.echo(f"Processing pool: {pool}")
    else:
        pool_names = _discover_pools(runtime)
        typer.echo(f"Discovered {len(pool_names)} pool(s): {pool_names}")

    if not pool_names:
        typer.echo("No pools found.")
        raise typer.Exit()

    for name in pool_names:
        _process_pool(runtime, name, backfill=backfill, dry_run=dry_run)

    runtime.close()
    typer.echo("Done.")


if __name__ == "__main__":
    app()

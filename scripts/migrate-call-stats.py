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
from sqlalchemy import Column, Double, Integer, MetaData, Table, Text, text as sa_text
from sqlalchemy.dialects.postgresql import TIMESTAMP

from dr_llm.pool.db.runtime import DbConfig, DbRuntime

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
_VALID_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_identifier(kind: str, value: str) -> str:
    if not _VALID_IDENTIFIER_RE.match(value):
        raise ValueError(
            f"{kind} must be lowercase alphanumeric with underscores, "
            f"starting with a letter; got {value!r}"
        )
    return value


def _samples_table_name(pool_name: str) -> str:
    return f"pool_{pool_name}_samples"


def _pending_table_name(pool_name: str) -> str:
    return f"pool_{pool_name}_pending"


def _call_stats_table_name(pool_name: str) -> str:
    return f"pool_{pool_name}_call_stats"


def _build_call_stats_table(pool_name: str) -> Table:
    metadata = MetaData()
    return Table(
        _call_stats_table_name(pool_name),
        metadata,
        Column("sample_id", Text, primary_key=True),
        Column("latency_ms", Integer, nullable=False),
        Column("total_cost_usd", Double),
        Column("prompt_tokens", Integer, nullable=False),
        Column("completion_tokens", Integer, nullable=False),
        Column("reasoning_tokens", Integer),
        Column("total_tokens", Integer, nullable=False),
        Column("attempt_count", Integer, nullable=False, server_default=sa_text("1")),
        Column("finish_reason", Text),
        Column(
            "created_at",
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
    )


def _ensure_call_stats_table(runtime: DbRuntime, pool_name: str) -> None:
    table = _build_call_stats_table(pool_name)
    with runtime.begin() as conn:
        table.metadata.create_all(bind=conn, tables=[table], checkfirst=True)


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
    samples_table = _samples_table_name(pool_name)
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


def _backfill_pool(runtime: DbRuntime, pool_name: str, key_columns: list[str]) -> int:
    """Backfill call_stats from existing sample payloads + pending attempt_count."""
    with runtime.begin() as conn:
        preparer = conn.dialect.identifier_preparer

        samples_table = _samples_table_name(pool_name)
        pending_table = _pending_table_name(pool_name)
        call_stats_table = _call_stats_table_name(pool_name)

        quoted_sample_id = preparer.quote_identifier("sample_id")
        quoted_sample_idx = preparer.quote_identifier("sample_idx")
        quoted_payload_json = preparer.quote_identifier("payload_json")
        quoted_attempt_count = preparer.quote_identifier("attempt_count")
        quoted_samples_table = (
            f"{preparer.quote_identifier('public')}."
            f"{preparer.quote_identifier(samples_table)}"
        )
        quoted_pending_table = (
            f"{preparer.quote_identifier('public')}."
            f"{preparer.quote_identifier(pending_table)}"
        )
        quoted_call_stats_table = (
            f"{preparer.quote_identifier('public')}."
            f"{preparer.quote_identifier(call_stats_table)}"
        )
        key_join_clauses = [
            (
                f"p.{preparer.quote_identifier(column)} = "
                f"s.{preparer.quote_identifier(column)}"
            )
            for column in key_columns
        ]
        pending_join_conditions = [
            "p.status = 'promoted'",
            f"p.{quoted_sample_idx} = s.{quoted_sample_idx}",
            *key_join_clauses,
        ]
        pending_join = (
            f"LEFT JOIN {quoted_pending_table} p "
            f"ON {' AND '.join(pending_join_conditions)}"
        )
        insert_columns = ", ".join(
            preparer.quote_identifier(column)
            for column in [
                "sample_id",
                "latency_ms",
                "total_cost_usd",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "total_tokens",
                "attempt_count",
                "finish_reason",
            ]
        )
        sql = f"""
            INSERT INTO {quoted_call_stats_table} ({insert_columns})
            SELECT
                s.{quoted_sample_id},
                COALESCE((s.{quoted_payload_json}->>'latency_ms')::integer, 0),
                (s.{quoted_payload_json}->'cost'->>'total_cost_usd')::double precision,
                COALESCE((s.{quoted_payload_json}->'usage'->>'prompt_tokens')::integer, 0),
                COALESCE((s.{quoted_payload_json}->'usage'->>'completion_tokens')::integer, 0),
                NULLIF((s.{quoted_payload_json}->'usage'->>'reasoning_tokens')::integer, 0),
                COALESCE((s.{quoted_payload_json}->'usage'->>'total_tokens')::integer, 0),
                COALESCE(p.{quoted_attempt_count}, 1),
                s.{quoted_payload_json}->>'finish_reason'
            FROM {quoted_samples_table} s
            {pending_join}
            ON CONFLICT ({quoted_sample_id}) DO NOTHING
        """  # noqa: S608
        result = conn.execute(sa_text(sql))
        return result.rowcount


def _process_pool(
    runtime: DbRuntime,
    pool_name: str,
    *,
    backfill: bool,
    dry_run: bool,
) -> None:
    _validate_identifier("Pool name", pool_name)
    key_columns = [
        _validate_identifier("Key column", column)
        for column in _discover_key_columns(runtime, pool_name)
    ]
    if not key_columns:
        typer.echo(f"  {pool_name}: no key columns inferred")
    else:
        typer.echo(f"  {pool_name}: key_columns={key_columns}")

    if dry_run:
        typer.echo(
            f"  [dry-run] would create table {_call_stats_table_name(pool_name)}"
        )
        if backfill:
            typer.echo(
                f"  [dry-run] would backfill from {_samples_table_name(pool_name)}"
            )
        return

    _ensure_call_stats_table(runtime, pool_name)
    typer.echo(f"  [ok] table {_call_stats_table_name(pool_name)} ensured")

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
        typer.Option(
            "--dry-run", help="Show what would be done without making changes"
        ),
    ] = False,
) -> None:
    """Create call_stats tables for existing pools."""
    resolved_dsn = dsn or getenv("DR_LLM_DATABASE_URL", "postgresql://localhost/dr_llm")
    runtime = DbRuntime(DbConfig(dsn=resolved_dsn))
    try:
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
    finally:
        runtime.close()
    typer.echo("Done.")


if __name__ == "__main__":
    app()

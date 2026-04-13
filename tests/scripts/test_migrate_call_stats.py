from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy.dialects.postgresql import insert as pg_insert
from typer.testing import CliRunner

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.db.tables import PoolTables
from dr_llm.pool.pending.pending_status import PendingStatus
from dr_llm.pool.pool_sample import PoolSample


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate-call-stats.py"
_RUNNER = CliRunner()


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "migrate_call_stats_script", _SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCRIPT = _load_script_module()
_APP = cast(Any, _SCRIPT).app


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


def _make_schema(prefix: str) -> PoolSchema:
    return PoolSchema(
        name=f"{prefix}_{uuid4().hex[:8]}",
        key_columns=[
            KeyColumn(name="dim_a"),
            KeyColumn(name="dim_b", type=ColumnType.integer),
        ],
    )


def _drop_tables(dsn: str, schema: PoolSchema) -> None:
    table_names = (
        schema.call_stats_table,
        schema.metadata_table,
        schema.claims_table,
        schema.pending_table,
        schema.samples_table,
    )
    with psycopg.connect(dsn) as conn:
        for table_name in table_names:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", table_name)
                )
            )
        conn.commit()


def _drop_tables_if_possible(dsn: str, schema: PoolSchema) -> None:
    try:
        _drop_tables(dsn, schema)
    except psycopg.OperationalError:
        pass


def _table_exists(dsn: str, table_name: str) -> bool:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            [table_name],
        ).fetchone()
    assert row is not None
    return bool(row[0])


def _create_legacy_pool(runtime: DbRuntime, schema: PoolSchema) -> PoolTables:
    tables = PoolTables(schema)
    with runtime.begin() as conn:
        tables.sa_metadata.create_all(
            bind=conn,
            tables=[
                tables.samples,
                tables.claims,
                tables.pending,
                tables.metadata_table,
            ],
            checkfirst=True,
        )
        tables.ensure_indexes(conn)
    return tables


def _make_runtime(dsn: str, schema: PoolSchema) -> DbRuntime:
    return DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name=f"migrate_call_stats_{schema.name}",
        )
    )


@pytest.mark.integration
def test_migrate_call_stats_dry_run_does_not_create_table() -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run migration integration tests")

    schema = _make_schema("mcsd")
    runtime: DbRuntime | None = None
    try:
        _drop_tables(dsn, schema)
        runtime = _make_runtime(dsn, schema)
        _create_legacy_pool(runtime, schema)
        assert _table_exists(dsn, schema.call_stats_table) is False

        result = _RUNNER.invoke(
            _APP,
            ["--dsn", dsn, "--pool", schema.name, "--dry-run"],
        )

        assert result.exit_code == 0
        assert f"Processing pool: {schema.name}" in result.output
        assert f"[dry-run] would create table {schema.call_stats_table}" in result.output
        assert _table_exists(dsn, schema.call_stats_table) is False
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        pytest.skip(f"Postgres unavailable for migration integration tests: {exc}")
    finally:
        _drop_tables_if_possible(dsn, schema)
        if runtime is not None:
            runtime.close()


@pytest.mark.integration
def test_migrate_call_stats_backfills_existing_sample_rows() -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run migration integration tests")

    schema = _make_schema("mcsb")
    runtime: DbRuntime | None = None
    try:
        _drop_tables(dsn, schema)
        runtime = _make_runtime(dsn, schema)
        tables = _create_legacy_pool(runtime, schema)

        sample = PoolSample(
            sample_id=f"sample_{uuid4().hex[:8]}",
            key_values={"dim_a": "alpha", "dim_b": 7},
            sample_idx=0,
            payload={
                "text": "generated text",
                "latency_ms": 800,
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75,
                    "reasoning_tokens": 0,
                },
                "cost": {"total_cost_usd": 0.003},
                "finish_reason": "stop",
            },
        )
        pending_row = {
            "pending_id": f"pending_{uuid4().hex[:8]}",
            "dim_a": "alpha",
            "dim_b": 7,
            "sample_idx": 0,
            "payload_json": {"partial": True},
            "metadata_json": {},
            "priority": 0,
            "status": PendingStatus.promoted.value,
            "attempt_count": 3,
        }

        with runtime.begin() as conn:
            conn.execute(pg_insert(tables.samples).values(**sample.to_db_insert_row()))
            conn.execute(pg_insert(tables.pending).values(**pending_row))

        assert _table_exists(dsn, schema.call_stats_table) is False

        result = _RUNNER.invoke(
            _APP,
            ["--dsn", dsn, "--pool", schema.name, "--backfill"],
        )

        assert result.exit_code == 0
        assert f"[ok] table {schema.call_stats_table} ensured" in result.output
        assert "[ok] backfilled 1 rows" in result.output
        assert _table_exists(dsn, schema.call_stats_table) is True

        with psycopg.connect(dsn) as conn:
            row = conn.execute(
                sql.SQL(
                    """
                    SELECT
                        latency_ms,
                        total_cost_usd,
                        prompt_tokens,
                        completion_tokens,
                        reasoning_tokens,
                        total_tokens,
                        attempt_count,
                        finish_reason
                    FROM {}
                    WHERE sample_id = %s
                    """
                ).format(sql.Identifier("public", schema.call_stats_table)),
                [sample.sample_id],
            ).fetchone()
        assert row is not None
        assert row[0] == 800
        assert row[1] == pytest.approx(0.003)
        assert row[2] == 50
        assert row[3] == 25
        assert row[4] is None
        assert row[5] == 75
        assert row[6] == 3
        assert row[7] == "stop"

        rerun = _RUNNER.invoke(
            _APP,
            ["--dsn", dsn, "--pool", schema.name, "--backfill"],
        )
        assert rerun.exit_code == 0
        assert "[ok] backfilled 0 rows" in rerun.output
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        pytest.skip(f"Postgres unavailable for migration integration tests: {exc}")
    finally:
        _drop_tables_if_possible(dsn, schema)
        if runtime is not None:
            runtime.close()

"""Integration tests for pool store operations (requires PostgreSQL)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Generator
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.db.names import (
    IndexNamePrefix,
    PoolIndexName,
    PoolTableType,
    pool_index_name,
)
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.errors import PoolSchemaError
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore


_TEST_SCHEMA = PoolSchema(
    name="itest",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)

_POOL_TABLES = tuple(reversed(_TEST_SCHEMA.table_names()))


def _eq_filter(**key_values: object) -> PoolKeyFilter:
    return PoolKeyFilter.eq(**key_values)


def _in_filter(**key_values: list[object]) -> PoolKeyFilter:
    return PoolKeyFilter.in_(**key_values)


def _drop_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        for tbl in _POOL_TABLES:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", tbl)
                )
            )
        conn.commit()
        try:
            conn.execute(
                "DELETE FROM pool_catalog WHERE pool_name = %s", [_TEST_SCHEMA.name]
            )
        except psycopg.errors.UndefinedTable:
            conn.rollback()
            return
        conn.commit()


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


def _index_exists(dsn: str, *, index_name: str) -> bool:
    with psycopg.connect(dsn) as conn:
        exists = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE schemaname = 'public' AND indexname = %s
            )
            """,
            [index_name],
        ).fetchone()
    assert exists is not None
    return bool(exists[0])


@pytest.fixture(scope="module")
def pool_store() -> Generator[PoolStore, None, None]:
    """Module-scoped store shared across tests. Tests MUST use unique dim_a
    values to avoid cross-test interference."""
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")
    runtime: DbRuntime | None = None
    try:
        _drop_tables(dsn)
        runtime = DbRuntime(
            DbConfig(
                dsn=dsn,
                min_pool_size=1,
                max_pool_size=4,
                application_name="pool_tests",
            )
        )
        store = PoolStore(_TEST_SCHEMA, runtime)
        store.ensure_schema()
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        if runtime is not None:
            runtime.close()
        pytest.skip(f"Postgres unavailable for pool integration tests: {exc}")
    yield store
    _drop_tables(dsn)
    runtime.close()


def _sample(dim_a: str = "a", dim_b: int = 1, **kwargs: Any) -> PoolSample:
    return PoolSample(
        key_values={"dim_a": dim_a, "dim_b": dim_b},
        request=kwargs.get("request", {"prompt": dim_a}),
        run_id=kwargs.get("run_id"),
        metadata=kwargs.get("metadata", {}),
        sample_idx=kwargs.get("sample_idx"),
    )


# --- Sample CRUD ---


@pytest.mark.integration
def test_insert_and_cell_depth(pool_store: PoolStore) -> None:
    s = _sample(sample_idx=0)
    assert pool_store.insert_sample(s) is True
    assert pool_store.cell_depth(key_values={"dim_a": "a", "dim_b": 1}) == 1


@pytest.mark.integration
def test_insert_auto_idx(pool_store: PoolStore) -> None:
    s1 = _sample(dim_a="auto", dim_b=10)
    s2 = _sample(dim_a="auto", dim_b=10)
    assert pool_store.insert_sample(s1) is True
    assert pool_store.insert_sample(s2) is True
    assert pool_store.cell_depth(key_values={"dim_a": "auto", "dim_b": 10}) == 2


@pytest.mark.integration
def test_insert_auto_idx_concurrent(pool_store: PoolStore) -> None:
    dim_a = f"auto_concurrent_{uuid4().hex[:8]}"

    def insert_one(_: int) -> bool:
        return pool_store.insert_sample(_sample(dim_a=dim_a, dim_b=11))

    with ThreadPoolExecutor(max_workers=8) as executor:
        inserted = list(executor.map(insert_one, range(8)))

    assert all(inserted)
    rows = pool_store.bulk_load(key_filter=_eq_filter(dim_a=dim_a, dim_b=11))
    assert sorted(row.sample_idx for row in rows) == list(range(8))


@pytest.mark.integration
def test_insert_duplicate_ignored(pool_store: PoolStore) -> None:
    s = _sample(dim_a="dup", dim_b=20, sample_idx=0)
    assert pool_store.insert_sample(s) is True
    s2 = _sample(dim_a="dup", dim_b=20, sample_idx=0)
    assert pool_store.insert_sample(s2) is False


@pytest.mark.integration
def test_bulk_insert(pool_store: PoolStore) -> None:
    samples = [_sample(dim_a="bulk", dim_b=i, sample_idx=0) for i in range(5)]
    result = pool_store.insert_samples(samples)
    assert result.inserted == 5
    assert result.skipped == 0


@pytest.mark.integration
def test_bootstrap_backfills_missing_unique_indexes() -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")

    schema = PoolSchema(
        name=f"itest_idx_{uuid4().hex[:8]}",
        key_columns=[
            KeyColumn(name="dim_a"),
            KeyColumn(name="dim_b", type=ColumnType.integer),
        ],
    )
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="pool_tests_index_backfill",
        )
    )
    try:
        store = PoolStore(schema, runtime)
        store.ensure_schema()
        store.insert_sample(
            PoolSample(
                key_values={"dim_a": "alpha", "dim_b": 1},
                sample_idx=0,
                request={"prompt": "hello"},
            )
        )

        index_name = pool_index_name(
            IndexNamePrefix.UNIQUE,
            schema.table_name(PoolTableType.SAMPLES),
            PoolIndexName.CELL,
        )
        with psycopg.connect(dsn) as conn:
            conn.execute(
                sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(index_name))
            )
            conn.commit()
        assert _index_exists(dsn, index_name=index_name) is False

        fresh_store = PoolStore(schema, runtime)
        fresh_store.ensure_schema()

        assert _index_exists(dsn, index_name=index_name) is True
        assert (
            fresh_store.insert_sample(
                PoolSample(
                    key_values={"dim_a": "alpha", "dim_b": 1},
                    sample_idx=0,
                    request={"prompt": "hello"},
                )
            )
            is False
        )
    finally:
        with psycopg.connect(dsn) as conn:
            for table_type in reversed(tuple(PoolTableType)):
                conn.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier("public", schema.table_name(table_type))
                    )
                )
            conn.execute("DELETE FROM pool_catalog WHERE pool_name = %s", [schema.name])
            conn.commit()
        runtime.close()


@pytest.mark.integration
def test_runtime_applies_statement_timeout_setting() -> None:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")

    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=1,
            application_name="pool_tests_timeout",
            statement_timeout_ms=1234,
        )
    )
    try:
        with runtime.connect() as conn:
            timeout_ms = conn.exec_driver_sql(
                "SELECT setting::int FROM pg_settings WHERE name = 'statement_timeout'"
            ).scalar_one()
        assert timeout_ms == 1234
    finally:
        runtime.close()


# --- Bulk Load ---


@pytest.mark.integration
def test_bulk_load(pool_store: PoolStore) -> None:
    samples = pool_store.bulk_load()
    assert isinstance(samples, list)


@pytest.mark.integration
def test_bulk_load_with_filter(pool_store: PoolStore) -> None:
    for i in range(3):
        pool_store.insert_sample(
            _sample(
                dim_a="bload",
                dim_b=99,
                sample_idx=i,
                request={"loaded": i},
                run_id=f"bulk-run-{i}",
                metadata={"batch": i},
            )
        )

    filtered = pool_store.bulk_load(key_filter=_eq_filter(dim_a="bload", dim_b=99))
    assert len(filtered) == 3
    assert all(s.key_values["dim_a"] == "bload" for s in filtered)
    first_batch = next(s for s in filtered if s.run_id == "bulk-run-0")
    assert first_batch.request == {"loaded": 0}
    assert first_batch.metadata == {"batch": 0}


# --- Validation ---


@pytest.mark.integration
def test_missing_key_raises(pool_store: PoolStore) -> None:
    with pytest.raises(PoolSchemaError, match="Missing key columns"):
        pool_store.insert_sample(
            PoolSample(key_values={"dim_a": "x"}, request={"prompt": "test"})
        )

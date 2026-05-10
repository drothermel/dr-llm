"""Integration tests for PoolReader (requires PostgreSQL)."""

from __future__ import annotations

import os
from collections.abc import Generator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import text

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.db.names import PoolTableType
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.errors import PoolNotFoundError
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.pool.reader import PoolReader


_READER_SCHEMA = PoolSchema(
    name=f"itest_reader_{uuid4().hex[:8]}",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)


def _get_dsn() -> str:
    dsn = os.getenv("DR_LLM_TEST_DATABASE_URL")
    if dsn is None:
        raise RuntimeError(
            "Set DR_LLM_TEST_DATABASE_URL to run pool integration tests"
        )
    return dsn


def _drop_pool_tables(dsn: str, schema: PoolSchema) -> None:
    with psycopg.connect(dsn) as conn:
        for table_type in reversed(tuple(PoolTableType)):
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", schema.table_name(table_type))
                )
            )
        conn.execute(
            "DELETE FROM pool_catalog WHERE pool_name = %s", [schema.name]
        )
        conn.commit()


@pytest.fixture(scope="module")
def reader_runtime() -> Generator[DbRuntime, None, None]:
    """Module-scoped DbRuntime for reader integration tests.

    Seeds a pool with 5 samples across two key groups:
      - alpha/1: 3 samples (idx 0-2), sample 0 completed ok, sample 1 completed error
      - beta/2:  2 samples (idx 0-1), both incomplete
    Then claims a lease on one incomplete sample.
    """
    dsn = _get_dsn()

    runtime: DbRuntime | None = None
    try:
        _drop_pool_tables(dsn, _READER_SCHEMA)
        runtime = DbRuntime(
            DbConfig(
                dsn=dsn,
                min_pool_size=1,
                max_pool_size=4,
                application_name="pool_reader_tests",
            )
        )
        store = PoolStore(_READER_SCHEMA, runtime)
        store.ensure_schema()

        for i in range(3):
            store.insert_sample(
                PoolSample(
                    key_values={"dim_a": "alpha", "dim_b": 1},
                    sample_idx=i,
                    request={"prompt": f"alpha-{i}"},
                )
            )
        for i in range(2):
            store.insert_sample(
                PoolSample(
                    key_values={"dim_a": "beta", "dim_b": 2},
                    sample_idx=i,
                    request={"prompt": f"beta-{i}"},
                )
            )

        alpha_samples = store.bulk_load(
            key_filter=PoolKeyFilter.eq(dim_a="alpha", dim_b=1)
        )
        alpha_samples.sort(key=lambda s: s.sample_idx or 0)
        store.complete_sample(
            sample_id=alpha_samples[0].sample_id,
            response={"text": "ok"},
            finish_reason="stop",
            attempt_count=1,
        )
        store.complete_sample(
            sample_id=alpha_samples[1].sample_id,
            response={"error": "boom"},
            finish_reason="error",
            attempt_count=3,
        )

        store.claim_lease(worker_id="w-reader", lease_seconds=600)

    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        if runtime is not None:
            runtime.close()
        pytest.skip(
            f"Postgres unavailable for pool reader integration tests: {exc}"
        )

    yield runtime
    _drop_pool_tables(dsn, _READER_SCHEMA)
    runtime.close()


@pytest.mark.integration
def test_open_from_catalog(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.open(_READER_SCHEMA.name, runtime=reader_runtime)
    assert reader.pool_name == _READER_SCHEMA.name
    assert reader.schema == _READER_SCHEMA


@pytest.mark.integration
def test_open_raises_for_missing_pool(reader_runtime: DbRuntime) -> None:
    with pytest.raises(PoolNotFoundError):
        PoolReader.open(
            f"never_created_{uuid4().hex[:8]}", runtime=reader_runtime
        )


@pytest.mark.integration
def test_from_runtime_happy_path(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    assert reader.pool_name == _READER_SCHEMA.name
    assert reader.schema == _READER_SCHEMA


@pytest.mark.integration
def test_samples_list_returns_all_seeded(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    samples = reader.samples_list()
    assert len(samples) == 5


@pytest.mark.integration
def test_samples_list_with_key_filter(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    alpha = reader.samples_list(key_filter=PoolKeyFilter.eq(dim_a="alpha"))
    assert all(s.key_values["dim_a"] == "alpha" for s in alpha)
    assert len(alpha) == 3


@pytest.mark.integration
def test_samples_streaming_iterator(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    streamed = list(reader.samples(key_filter=PoolKeyFilter.eq(dim_a="beta")))
    assert len(streamed) == 2
    assert all(s.key_values["dim_a"] == "beta" for s in streamed)


@pytest.mark.integration
def test_samples_list_with_completion_filter(
    reader_runtime: DbRuntime,
) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)

    complete = reader.samples_list(completion="complete")
    assert len(complete) == 2

    incomplete = reader.samples_list(completion="incomplete")
    assert len(incomplete) == 3

    errors = reader.samples_list(completion="error")
    assert len(errors) == 1
    assert errors[0].finish_reason == "error"


@pytest.mark.integration
def test_samples_list_supports_in_filter(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    rows = reader.samples_list(
        key_filter=PoolKeyFilter.in_(dim_a=["alpha", "beta"])
    )
    assert len(rows) == 5


@pytest.mark.integration
def test_progress_aggregates_correctly(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    progress = reader.progress()
    assert progress.total == 5
    assert progress.complete == 2
    assert progress.incomplete == 3
    assert progress.error == 1
    assert progress.leased >= 1


@pytest.mark.integration
def test_progress_with_key_filter(reader_runtime: DbRuntime) -> None:
    reader = PoolReader.from_runtime(reader_runtime, schema=_READER_SCHEMA)
    alpha_progress = reader.progress(
        key_filter=PoolKeyFilter.eq(dim_a="alpha")
    )
    assert alpha_progress.total == 3
    assert alpha_progress.complete == 2
    assert alpha_progress.incomplete == 1

    beta_progress = reader.progress(key_filter=PoolKeyFilter.eq(dim_a="beta"))
    assert beta_progress.total == 2
    assert beta_progress.complete == 0
    assert beta_progress.incomplete == 2


# --- Lifecycle ---


def _isolated_pool_schema() -> PoolSchema:
    return PoolSchema(
        name=f"itest_reader_iso_{uuid4().hex[:8]}",
        key_columns=[KeyColumn(name="dim_a")],
    )


@pytest.mark.integration
def test_close_disposes_owned_runtime_only() -> None:
    dsn = _get_dsn()
    schema = _isolated_pool_schema()
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="pool_reader_tests_lifecycle",
        )
    )
    try:
        store = PoolStore(schema, runtime)
        store.ensure_schema()

        reader = PoolReader.from_runtime(runtime, schema=schema)
        reader.close()
        reader.close()
        with runtime.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        _drop_pool_tables(dsn, schema)
        runtime.close()


@pytest.mark.integration
def test_context_manager_calls_close() -> None:
    dsn = _get_dsn()
    schema = _isolated_pool_schema()
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=2,
            application_name="pool_reader_tests_ctx",
        )
    )
    try:
        store = PoolStore(schema, runtime)
        store.ensure_schema()

        with PoolReader.from_runtime(runtime, schema=schema) as reader:
            assert reader.samples_list() == []
        with runtime.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        _drop_pool_tables(dsn, schema)
        runtime.close()

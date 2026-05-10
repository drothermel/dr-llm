"""Integration tests for sampling acquisition (requires PostgreSQL)."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import psycopg
import pytest
from psycopg import sql

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.sampling.acquisition import AcquireQuery
from dr_llm.sampling.db.names import claims_table_name
from dr_llm.sampling.errors import PoolTopupError
from dr_llm.sampling.pool_service import PoolService
from dr_llm.sampling.sampling_store import SamplingStore

_TEST_SCHEMA = PoolSchema(
    name="itest_sampling",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)

_CONSUMER_ID = "test_sweep"

_POOL_TABLES = tuple(reversed(_TEST_SCHEMA.table_names()))


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


def _drop_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        for tbl in _POOL_TABLES:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", tbl)
                )
            )
        claims_tbl = claims_table_name(_TEST_SCHEMA.name, _CONSUMER_ID)
        conn.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier("public", claims_tbl)
            )
        )
        conn.commit()


def _sample(dim_a: str = "a", dim_b: int = 1, **kwargs: Any) -> PoolSample:
    return PoolSample(
        key_values={"dim_a": dim_a, "dim_b": dim_b},
        payload=kwargs.get("payload", {"data": "test"}),
        source_run_id=kwargs.get("source_run_id"),
        metadata=kwargs.get("metadata", {}),
        sample_idx=kwargs.get("sample_idx"),
    )


@pytest.fixture(scope="module")
def pool_store() -> Generator[PoolStore, None, None]:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run sampling integration tests")
    runtime: DbRuntime | None = None
    try:
        _drop_tables(dsn)
        runtime = DbRuntime(
            DbConfig(
                dsn=dsn,
                min_pool_size=1,
                max_pool_size=4,
                application_name="sampling_tests",
            )
        )
        store = PoolStore(_TEST_SCHEMA, runtime)
        store.ensure_schema()
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        if runtime is not None:
            runtime.close()
        pytest.skip(f"Postgres unavailable for sampling integration tests: {exc}")
    yield store
    dsn_val = _get_dsn()
    if dsn_val:
        _drop_tables(dsn_val)
    runtime.close()


@pytest.fixture(scope="module")
def sampling_store(pool_store: PoolStore) -> Generator[SamplingStore, None, None]:
    store = SamplingStore(pool_store.schema, pool_store._runtime, pool_store._tables)
    store.setup_consumer(_CONSUMER_ID)
    yield store
    store.teardown_consumer(_CONSUMER_ID)


@pytest.mark.integration
def test_acquire_basic(pool_store: PoolStore, sampling_store: SamplingStore) -> None:
    pool_store.insert_sample(
        _sample(
            dim_a="acq",
            dim_b=1,
            sample_idx=0,
            payload={"data": "primary"},
            source_run_id="seed-run",
            metadata={"kind": "primary"},
        )
    )
    for i in range(1, 3):
        pool_store.insert_sample(_sample(dim_a="acq", dim_b=1, sample_idx=i))

    q = AcquireQuery(run_id="run1", key_values={"dim_a": "acq", "dim_b": 1}, n=2)
    result = sampling_store.acquire(q, _CONSUMER_ID)
    assert result.claimed == 2
    assert len(result.samples) == 2
    assert result.samples[0].sample_idx == 0
    assert result.samples[1].sample_idx == 1
    assert result.samples[0].payload == {"data": "primary"}
    assert result.samples[0].source_run_id == "seed-run"
    assert result.samples[0].metadata == {"kind": "primary"}


@pytest.mark.integration
def test_acquire_no_replacement(
    pool_store: PoolStore, sampling_store: SamplingStore
) -> None:
    for i in range(3):
        pool_store.insert_sample(_sample(dim_a="norep", dim_b=1, sample_idx=i))

    q1 = AcquireQuery(run_id="run_nr", key_values={"dim_a": "norep", "dim_b": 1}, n=2)
    r1 = sampling_store.acquire(q1, _CONSUMER_ID)
    assert r1.claimed == 2

    q2 = AcquireQuery(run_id="run_nr", key_values={"dim_a": "norep", "dim_b": 1}, n=2)
    r2 = sampling_store.acquire(q2, _CONSUMER_ID)
    assert r2.claimed == 1

    q3 = AcquireQuery(run_id="run_nr", key_values={"dim_a": "norep", "dim_b": 1}, n=2)
    r3 = sampling_store.acquire(q3, _CONSUMER_ID)
    assert r3.claimed == 0


@pytest.mark.integration
def test_remaining(pool_store: PoolStore, sampling_store: SamplingStore) -> None:
    for i in range(3):
        pool_store.insert_sample(_sample(dim_a="rem", dim_b=1, sample_idx=i))

    assert (
        sampling_store.remaining(
            run_id="run_rem",
            key_values={"dim_a": "rem", "dim_b": 1},
            consumer_id=_CONSUMER_ID,
        )
        == 3
    )

    sampling_store.acquire(
        AcquireQuery(run_id="run_rem", key_values={"dim_a": "rem", "dim_b": 1}, n=1),
        _CONSUMER_ID,
    )
    assert (
        sampling_store.remaining(
            run_id="run_rem",
            key_values={"dim_a": "rem", "dim_b": 1},
            consumer_id=_CONSUMER_ID,
        )
        == 2
    )


@pytest.mark.integration
def test_service_acquire_or_generate(pool_store: PoolStore) -> None:
    svc_sampling = SamplingStore(
        pool_store.schema, pool_store._runtime, pool_store._tables
    )
    svc_consumer = "svc_sweep"
    svc_sampling.setup_consumer(svc_consumer)
    try:
        service = PoolService(pool_store)
        service._sampling = svc_sampling

        def generator(key_values: dict[str, Any], deficit: int) -> list[PoolSample]:
            return [
                PoolSample(key_values=key_values, payload={"generated": True})
                for _ in range(deficit)
            ]

        q = AcquireQuery(
            run_id="svc_run",
            key_values={"dim_a": "svc", "dim_b": 42},
            n=3,
        )
        result = service.acquire_or_generate(
            q, consumer_id=svc_consumer, generator_fn=generator
        )
        assert len(result.samples) == 3

        depth = pool_store.cell_depth(key_values={"dim_a": "svc", "dim_b": 42})
        assert depth >= 3
    finally:
        svc_sampling.teardown_consumer(svc_consumer)


@pytest.mark.integration
def test_service_generator_error_wraps_as_topup_error(pool_store: PoolStore) -> None:
    svc_sampling = SamplingStore(
        pool_store.schema, pool_store._runtime, pool_store._tables
    )
    svc_consumer = "svc_err_sweep"
    svc_sampling.setup_consumer(svc_consumer)
    try:
        service = PoolService(pool_store)
        service._sampling = svc_sampling

        def failing_generator(
            key_values: dict[str, Any], deficit: int
        ) -> list[PoolSample]:
            raise RuntimeError("LLM call timed out")

        q = AcquireQuery(
            run_id="svc_fail",
            key_values={"dim_a": "svc_err", "dim_b": 99},
            n=3,
        )
        with pytest.raises(PoolTopupError, match="Top-up generation failed"):
            service.acquire_or_generate(
                q, consumer_id=svc_consumer, generator_fn=failing_generator
            )
    finally:
        svc_sampling.teardown_consumer(svc_consumer)

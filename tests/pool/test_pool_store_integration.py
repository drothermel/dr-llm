"""Integration tests for the unified pool store."""

from __future__ import annotations

import os
from collections.abc import Generator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from dr_llm.errors import TransientPersistenceError
from dr_llm.pool.db import ColumnType, KeyColumn, PoolSchema, PoolTableType
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.key_filter import PoolKeyFilter
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore

pytestmark = pytest.mark.integration


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


def _schema() -> PoolSchema:
    return PoolSchema(
        name=f"store_{uuid4().hex[:8]}",
        key_columns=[
            KeyColumn(name="dim_a"),
            KeyColumn(name="dim_b", type=ColumnType.integer),
        ],
    )


def _sample(
    *,
    dim_a: str = "a",
    dim_b: int = 1,
    sample_id: str | None = None,
    sample_idx: int | None = 0,
    request: dict[str, object] | None = None,
) -> PoolSample:
    return PoolSample(
        sample_id=sample_id or uuid4().hex,
        key_values={"dim_a": dim_a, "dim_b": dim_b},
        sample_idx=sample_idx,
        request=request or {"prompt": dim_a},
    )


def _drop_tables(dsn: str, schema: PoolSchema) -> None:
    with psycopg.connect(dsn) as conn:
        for table_type in reversed(tuple(PoolTableType)):
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier("public", schema.table_name(table_type))
                )
            )
        conn.commit()


def _table_exists(dsn: str, table_name: str) -> bool:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            [table_name],
        ).fetchone()
    assert row is not None
    return bool(row[0])


def _lease_worker(dsn: str, schema: PoolSchema, sample_id: str) -> str | None:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            sql.SQL("SELECT worker_id FROM {} WHERE sample_id = %s").format(
                sql.Identifier("public", schema.table_name(PoolTableType.LEASES))
            ),
            [sample_id],
        ).fetchone()
    return None if row is None else str(row[0])


def _expire_lease(dsn: str, schema: PoolSchema, sample_id: str) -> None:
    with psycopg.connect(dsn) as conn:
        conn.execute(
            sql.SQL(
                """
                UPDATE {}
                SET lease_expires_at = now() - interval '1 second'
                WHERE sample_id = %s
                """
            ).format(sql.Identifier("public", schema.table_name(PoolTableType.LEASES))),
            [sample_id],
        )
        conn.commit()


@pytest.fixture
def pool_store() -> Generator[PoolStore, None, None]:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip("Set DR_LLM_TEST_DATABASE_URL to run pool integration tests")

    schema = _schema()
    runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=4,
            application_name="pool_store_phase2_tests",
        )
    )
    store = PoolStore(schema, runtime)
    try:
        store.ensure_schema()
    except (psycopg.OperationalError, TransientPersistenceError) as exc:
        runtime.close()
        pytest.skip(f"Postgres unavailable for pool integration tests: {exc}")

    try:
        yield store
    finally:
        _drop_tables(dsn, schema)
        runtime.close()


def test_ensure_schema_creates_tables_idempotently(pool_store: PoolStore) -> None:
    dsn = _get_dsn()
    assert dsn is not None

    pool_store.ensure_schema()
    pool_store.ensure_schema()

    assert (
        _table_exists(dsn, pool_store.schema.table_name(PoolTableType.SAMPLES)) is True
    )
    assert (
        _table_exists(dsn, pool_store.schema.table_name(PoolTableType.LEASES)) is True
    )


def test_insert_samples_and_complete_sample(pool_store: PoolStore) -> None:
    sample = _sample(dim_a="complete", dim_b=1, sample_id="complete-1")
    assert pool_store.insert_sample(sample) is True
    [stored] = pool_store.bulk_load(key_filter=PoolKeyFilter.eq(dim_a="complete"))
    assert stored.is_complete is False

    assert (
        pool_store.complete_sample(
            sample_id=sample.sample_id,
            response={"text": "done"},
            finish_reason="stop",
            attempt_count=2,
        )
        is True
    )
    assert (
        pool_store.complete_sample(
            sample_id=sample.sample_id,
            response={"text": "again"},
            finish_reason="stop",
            attempt_count=3,
        )
        is False
    )

    [completed] = pool_store.bulk_load(key_filter=PoolKeyFilter.eq(dim_a="complete"))
    assert completed.is_complete is True
    assert completed.response == {"text": "done"}
    assert completed.finish_reason == "stop"
    assert completed.attempt_count == 2


def test_claim_lease_skips_active_leases(pool_store: PoolStore) -> None:
    samples = [
        _sample(dim_a="claim", dim_b=1, sample_id="claim-1", sample_idx=0),
        _sample(dim_a="claim", dim_b=1, sample_id="claim-2", sample_idx=1),
    ]
    assert pool_store.insert_samples(samples).inserted == 2

    first = pool_store.claim_lease(worker_id="worker-a", lease_seconds=60)
    second = pool_store.claim_lease(worker_id="worker-a", lease_seconds=60)
    third = pool_store.claim_lease(worker_id="worker-b", lease_seconds=60)

    assert first is not None
    assert second is not None
    assert first.sample_id != second.sample_id
    assert third is None
    dsn = _get_dsn()
    assert dsn is not None
    assert _lease_worker(dsn, pool_store.schema, first.sample_id) == "worker-a"


def test_release_lease_makes_sample_claimable_again(pool_store: PoolStore) -> None:
    sample = _sample(dim_a="release", sample_id="release-1")
    assert pool_store.insert_sample(sample) is True
    claimed = pool_store.claim_lease(worker_id="worker-a", lease_seconds=60)
    assert claimed is not None

    assert (
        pool_store.release_lease(sample_id=claimed.sample_id, worker_id="wrong-worker")
        is False
    )
    assert (
        pool_store.release_lease(sample_id=claimed.sample_id, worker_id="worker-a")
        is True
    )
    reclaimed = pool_store.claim_lease(worker_id="worker-b", lease_seconds=60)
    assert reclaimed is not None
    assert reclaimed.sample_id == claimed.sample_id


def test_expire_leases_removes_expired_rows(pool_store: PoolStore) -> None:
    dsn = _get_dsn()
    assert dsn is not None
    sample = _sample(dim_a="expire", sample_id="expire-1")
    assert pool_store.insert_sample(sample) is True
    claimed = pool_store.claim_lease(worker_id="worker-a", lease_seconds=60)
    assert claimed is not None
    _expire_lease(dsn, pool_store.schema, claimed.sample_id)

    assert pool_store.expire_leases() == 1
    assert _lease_worker(dsn, pool_store.schema, claimed.sample_id) is None
    reclaimed = pool_store.claim_lease(worker_id="worker-b", lease_seconds=60)
    assert reclaimed is not None
    assert reclaimed.sample_id == claimed.sample_id


def test_claim_lease_reclaims_expired_lease(pool_store: PoolStore) -> None:
    dsn = _get_dsn()
    assert dsn is not None
    sample = _sample(dim_a="reclaim", sample_id="reclaim-1")
    assert pool_store.insert_sample(sample) is True
    claimed = pool_store.claim_lease(worker_id="worker-a", lease_seconds=60)
    assert claimed is not None
    _expire_lease(dsn, pool_store.schema, claimed.sample_id)

    reclaimed = pool_store.claim_lease(worker_id="worker-b", lease_seconds=60)

    assert reclaimed is not None
    assert reclaimed.sample_id == claimed.sample_id
    assert _lease_worker(dsn, pool_store.schema, claimed.sample_id) == "worker-b"


def test_completion_counts(pool_store: PoolStore) -> None:
    samples = [
        _sample(dim_a="counts", dim_b=1, sample_id="counts-1", sample_idx=0),
        _sample(dim_a="counts", dim_b=1, sample_id="counts-2", sample_idx=1),
        _sample(dim_a="counts", dim_b=2, sample_id="counts-3", sample_idx=0),
    ]
    assert pool_store.insert_samples(samples).inserted == 3
    assert pool_store.incomplete_count() == 3
    assert pool_store.complete_count() == 0

    assert (
        pool_store.complete_sample(
            sample_id="counts-1",
            response={"text": "done"},
            finish_reason=None,
            attempt_count=1,
        )
        is True
    )

    assert pool_store.incomplete_count() == 2
    assert pool_store.complete_count() == 1
    assert (
        pool_store.incomplete_count(
            key_filter=PoolKeyFilter.eq(dim_a="counts", dim_b=1)
        )
        == 1
    )


def test_claim_lease_respects_key_filter(pool_store: PoolStore) -> None:
    assert (
        pool_store.insert_samples(
            [
                _sample(dim_a="skip", dim_b=1, sample_id="filter-1"),
                _sample(dim_a="target", dim_b=1, sample_id="filter-2"),
            ]
        ).inserted
        == 2
    )

    claimed = pool_store.claim_lease(
        worker_id="worker-a",
        lease_seconds=60,
        key_filter=PoolKeyFilter.eq(dim_a="target"),
    )

    assert claimed is not None
    assert claimed.sample_id == "filter-2"
    assert claimed.key_values["dim_a"] == "target"


def test_auto_idx_insertion_allocates_sequential_indices(
    pool_store: PoolStore,
) -> None:
    samples = [
        _sample(dim_a="auto", dim_b=1, sample_idx=None),
        _sample(dim_a="auto", dim_b=1, sample_idx=None),
        _sample(dim_a="auto", dim_b=1, sample_idx=None),
    ]
    assert pool_store.insert_samples(samples).inserted == 3

    rows = pool_store.bulk_load(key_filter=PoolKeyFilter.eq(dim_a="auto", dim_b=1))
    assert sorted(row.sample_idx for row in rows) == [0, 1, 2]

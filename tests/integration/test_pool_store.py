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
from dr_llm.pool.call_stats import CallStats
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.errors import PoolSchemaError, PoolTopupError
from dr_llm.pool.key_filter import PoolKeyFilter
from dr_llm.pool.models import AcquireQuery
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatus
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_service import PoolService
from dr_llm.pool.pool_store import PoolStore


_TEST_SCHEMA = PoolSchema(
    name="itest",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)

_POOL_TABLES = (
    _TEST_SCHEMA.call_stats_table,
    _TEST_SCHEMA.metadata_table,
    _TEST_SCHEMA.claims_table,
    _TEST_SCHEMA.pending_table,
    _TEST_SCHEMA.samples_table,
)


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
        payload=kwargs.get("payload", {"data": "test"}),
        source_run_id=kwargs.get("source_run_id"),
        metadata=kwargs.get("metadata", {}),
        sample_idx=kwargs.get("sample_idx"),
    )


def _pending(dim_a: str = "a", dim_b: int = 1, **kwargs: Any) -> PendingSample:
    return PendingSample(
        key_values={"dim_a": dim_a, "dim_b": dim_b},
        sample_idx=kwargs.get("sample_idx", 0),
        payload=kwargs.get("payload", {"partial": True}),
        source_run_id=kwargs.get("source_run_id"),
        metadata=kwargs.get("metadata", {}),
        priority=kwargs.get("priority", 0),
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
    # Same key + sample_idx should be silently skipped
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
            )
        )

        index_name = f"uq_{schema.samples_table}_cell"
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
                )
            )
            is False
        )
    finally:
        with psycopg.connect(dsn) as conn:
            for table_name in (
                schema.metadata_table,
                schema.claims_table,
                schema.pending_table,
                schema.samples_table,
            ):
                conn.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier("public", table_name)
                    )
                )
            conn.commit()
        runtime.close()


# --- Acquire ---


@pytest.mark.integration
def test_acquire_basic(pool_store: PoolStore) -> None:
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
    result = pool_store.acquire(q)
    assert result.claimed == 2
    assert len(result.samples) == 2
    assert result.samples[0].sample_idx == 0
    assert result.samples[1].sample_idx == 1
    assert result.samples[0].payload == {"data": "primary"}
    assert result.samples[0].source_run_id == "seed-run"
    assert result.samples[0].metadata == {"kind": "primary"}
    assert result.samples[0].status.value == "active"


@pytest.mark.integration
def test_acquire_no_replacement(pool_store: PoolStore) -> None:
    for i in range(3):
        pool_store.insert_sample(_sample(dim_a="norep", dim_b=1, sample_idx=i))

    q1 = AcquireQuery(run_id="run_nr", key_values={"dim_a": "norep", "dim_b": 1}, n=2)
    r1 = pool_store.acquire(q1)
    assert r1.claimed == 2

    # Second acquire in same run should get remaining sample
    q2 = AcquireQuery(run_id="run_nr", key_values={"dim_a": "norep", "dim_b": 1}, n=2)
    r2 = pool_store.acquire(q2)
    assert r2.claimed == 1

    # Third acquire exhausts pool
    q3 = AcquireQuery(run_id="run_nr", key_values={"dim_a": "norep", "dim_b": 1}, n=2)
    r3 = pool_store.acquire(q3)
    assert r3.claimed == 0


@pytest.mark.integration
def test_remaining(pool_store: PoolStore) -> None:
    for i in range(3):
        pool_store.insert_sample(_sample(dim_a="rem", dim_b=1, sample_idx=i))

    assert (
        pool_store.remaining(run_id="run_rem", key_values={"dim_a": "rem", "dim_b": 1})
        == 3
    )

    pool_store.acquire(
        AcquireQuery(run_id="run_rem", key_values={"dim_a": "rem", "dim_b": 1}, n=1)
    )
    assert (
        pool_store.remaining(run_id="run_rem", key_values={"dim_a": "rem", "dim_b": 1})
        == 2
    )


# --- Pending ---


@pytest.mark.integration
def test_pending_insert_and_count(pool_store: PoolStore) -> None:
    p = _pending(dim_a="pend", dim_b=1)
    assert pool_store.pending.insert(p) is True
    assert (
        pool_store.pending.count_in_flight(key_values={"dim_a": "pend", "dim_b": 1})
        == 1
    )


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


@pytest.mark.integration
def test_pending_claim_and_promote(pool_store: PoolStore) -> None:
    p = _pending(
        dim_a="promote",
        dim_b=1,
        sample_idx=0,
        payload={"partial": True, "draft": 1},
        source_run_id="pending-run",
        metadata={"source": "seed"},
    )
    pool_store.pending.insert(p)

    claimed = pool_store.pending.claim(
        worker_id="w1",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="promote", dim_b=1),
    )
    assert claimed is not None
    assert claimed.status == PendingStatus.leased
    assert claimed.worker_id == "w1"
    assert claimed.payload == {"partial": True, "draft": 1}
    assert claimed.source_run_id == "pending-run"
    assert claimed.metadata == {"source": "seed"}
    assert claimed.attempt_count == 1

    # Promote with final payload
    sample = pool_store.pending.promote(
        pending_id=claimed.pending_id,
        worker_id="w1",
        payload={"final": True, "score": 0.95},
    )
    assert sample is not None
    assert sample.payload["final"] is True
    assert sample.source_run_id == "pending-run"
    assert sample.metadata == {"source": "seed"}
    assert pool_store.cell_depth(key_values={"dim_a": "promote", "dim_b": 1}) == 1


@pytest.mark.integration
def test_pending_claim_rejects_non_positive_lease(pool_store: PoolStore) -> None:
    with pytest.raises(ValueError, match="lease_seconds must be a positive integer"):
        pool_store.pending.claim(
            worker_id="w1",
            lease_seconds=0,
            key_filter=_eq_filter(dim_a="noop", dim_b=1),
        )


@pytest.mark.integration
def test_pending_fail(pool_store: PoolStore) -> None:
    p = _pending(dim_a="fail", dim_b=1, sample_idx=0)
    pool_store.pending.insert(p)

    claimed = pool_store.pending.claim(
        worker_id="w1",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="fail", dim_b=1),
    )
    assert claimed is not None
    pool_store.pending.fail(
        pending_id=claimed.pending_id, worker_id="w1", reason="docker timeout"
    )

    # Failed samples should not be counted as pending
    assert (
        pool_store.pending.count_in_flight(key_values={"dim_a": "fail", "dim_b": 1})
        == 0
    )


@pytest.mark.integration
def test_pending_claim_supports_in_filter(pool_store: PoolStore) -> None:
    pool_store.pending.insert(_pending(dim_a="claim_in_a", dim_b=1, sample_idx=0))
    pool_store.pending.insert(_pending(dim_a="claim_in_b", dim_b=1, sample_idx=0))
    pool_store.pending.insert(_pending(dim_a="claim_in_c", dim_b=1, sample_idx=0))

    claimed = pool_store.pending.claim(
        worker_id="w-in",
        lease_seconds=60,
        key_filter=_in_filter(dim_a=["claim_in_a", "claim_in_b"]),
    )

    assert claimed is not None
    assert claimed.key_values["dim_a"] in {"claim_in_a", "claim_in_b"}


@pytest.mark.integration
def test_pending_status_counts_support_in_filter(pool_store: PoolStore) -> None:
    pool_store.pending.insert(_pending(dim_a="counts_in_a", dim_b=1, sample_idx=0))
    pool_store.pending.insert(_pending(dim_a="counts_in_b", dim_b=1, sample_idx=0))
    pool_store.pending.insert(_pending(dim_a="counts_in_c", dim_b=1, sample_idx=0))

    counts = pool_store.pending.status_counts(
        key_filter=_in_filter(dim_a=["counts_in_a", "counts_in_b"])
    )

    assert counts.pending == 2
    assert counts.leased == 0
    assert counts.promoted == 0
    assert counts.failed == 0


@pytest.mark.integration
def test_requeue_failed_supports_filtered_subset(pool_store: PoolStore) -> None:
    failed_a = _pending(
        dim_a="requeue_in_a",
        dim_b=1,
        sample_idx=0,
        metadata={"seed": "a"},
    )
    failed_b = _pending(
        dim_a="requeue_in_b",
        dim_b=1,
        sample_idx=0,
        metadata={"seed": "b"},
    )
    failed_c = _pending(
        dim_a="requeue_in_c",
        dim_b=1,
        sample_idx=0,
        metadata={"seed": "c"},
    )
    pool_store.pending.insert_many([failed_a, failed_b, failed_c])

    for worker_id, dim_a in [
        ("w-rq-a", "requeue_in_a"),
        ("w-rq-b", "requeue_in_b"),
        ("w-rq-c", "requeue_in_c"),
    ]:
        claimed = pool_store.pending.claim(
            worker_id=worker_id,
            lease_seconds=60,
            key_filter=_eq_filter(dim_a=dim_a, dim_b=1),
        )
        assert claimed is not None
        pool_store.pending.fail(
            pending_id=claimed.pending_id,
            worker_id=worker_id,
            reason=f"failed-{dim_a}",
        )

    updated = pool_store.pending.requeue_failed(
        key_filter=_in_filter(dim_a=["requeue_in_a", "requeue_in_b"])
    )

    assert updated == 2
    pending_rows = pool_store.pending.bulk_load(
        key_filter=_in_filter(dim_a=["requeue_in_a", "requeue_in_b"]),
        status=PendingStatus.pending,
    )
    assert len(pending_rows) == 2
    assert all(row.attempt_count == 0 for row in pending_rows)
    assert all("fail_reason" not in row.metadata for row in pending_rows)

    failed_rows = pool_store.pending.bulk_load(
        key_filter=_eq_filter(dim_a="requeue_in_c"),
        status=PendingStatus.failed,
    )
    assert len(failed_rows) == 1
    assert failed_rows[0].metadata["fail_reason"] == "failed-requeue_in_c"


@pytest.mark.integration
def test_clear_pending_deletes_only_pending_rows(pool_store: PoolStore) -> None:
    pending_only = _pending(dim_a="clear_all_pending", dim_b=1, sample_idx=0)
    leased_row = _pending(dim_a="clear_all_leased", dim_b=1, sample_idx=0)
    failed_row = _pending(dim_a="clear_all_failed", dim_b=1, sample_idx=0)
    promoted_row = _pending(dim_a="clear_all_promoted", dim_b=1, sample_idx=0)
    pool_store.pending.insert_many(
        [pending_only, leased_row, failed_row, promoted_row]
    )

    leased = pool_store.pending.claim(
        worker_id="w-clear-leased",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="clear_all_leased", dim_b=1),
    )
    assert leased is not None

    failed = pool_store.pending.claim(
        worker_id="w-clear-failed",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="clear_all_failed", dim_b=1),
    )
    assert failed is not None
    assert pool_store.pending.fail(
        pending_id=failed.pending_id,
        worker_id="w-clear-failed",
        reason="expected-failure",
    )

    promoted = pool_store.pending.claim(
        worker_id="w-clear-promoted",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="clear_all_promoted", dim_b=1),
    )
    assert promoted is not None
    assert (
        pool_store.pending.promote(
            pending_id=promoted.pending_id, worker_id="w-clear-promoted"
        )
        is not None
    )

    cleared = pool_store.pending.clear_pending(
        key_filter=_in_filter(
            dim_a=[
                "clear_all_pending",
                "clear_all_leased",
                "clear_all_failed",
                "clear_all_promoted",
            ]
        )
    )

    assert cleared == 1
    assert pool_store.pending.bulk_load(
        key_filter=_eq_filter(dim_a="clear_all_pending"),
        status=PendingStatus.pending,
    ) == []
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_all_leased"),
            status=PendingStatus.leased,
        )
    ) == 1
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_all_failed"),
            status=PendingStatus.failed,
        )
    ) == 1
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_all_promoted"),
            status=PendingStatus.promoted,
        )
    ) == 1


@pytest.mark.integration
def test_clear_pending_supports_filtered_subset(pool_store: PoolStore) -> None:
    pending_a = _pending(dim_a="clear_in_a", dim_b=1, sample_idx=0)
    pending_b = _pending(dim_a="clear_in_b", dim_b=1, sample_idx=0)
    pending_c = _pending(dim_a="clear_in_c", dim_b=1, sample_idx=0)
    pool_store.pending.insert_many([pending_a, pending_b, pending_c])

    failed = pool_store.pending.claim(
        worker_id="w-clear-in-b",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="clear_in_b", dim_b=1),
    )
    assert failed is not None
    assert pool_store.pending.fail(
        pending_id=failed.pending_id,
        worker_id="w-clear-in-b",
        reason="stay-failed",
    )

    cleared = pool_store.pending.clear_pending(
        key_filter=_in_filter(dim_a=["clear_in_a", "clear_in_b"])
    )

    assert cleared == 1
    assert pool_store.pending.bulk_load(
        key_filter=_eq_filter(dim_a="clear_in_a"),
        status=PendingStatus.pending,
    ) == []
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_in_b"),
            status=PendingStatus.failed,
        )
    ) == 1
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_in_c"),
            status=PendingStatus.pending,
        )
    ) == 1


@pytest.mark.integration
def test_clear_pending_without_filter_deletes_all_pending_rows(
    pool_store: PoolStore,
) -> None:
    before_counts = pool_store.pending.status_counts()

    pending_a = _pending(dim_a="clear_unfiltered_pending_a", dim_b=1, sample_idx=0)
    pending_b = _pending(dim_a="clear_unfiltered_pending_b", dim_b=1, sample_idx=0)
    leased_row = _pending(dim_a="clear_unfiltered_leased", dim_b=1, sample_idx=0)
    failed_row = _pending(dim_a="clear_unfiltered_failed", dim_b=1, sample_idx=0)
    promoted_row = _pending(dim_a="clear_unfiltered_promoted", dim_b=1, sample_idx=0)
    pool_store.pending.insert_many(
        [pending_a, pending_b, leased_row, failed_row, promoted_row]
    )

    leased = pool_store.pending.claim(
        worker_id="w-clear-unfiltered-leased",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="clear_unfiltered_leased", dim_b=1),
    )
    assert leased is not None

    failed = pool_store.pending.claim(
        worker_id="w-clear-unfiltered-failed",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="clear_unfiltered_failed", dim_b=1),
    )
    assert failed is not None
    assert pool_store.pending.fail(
        pending_id=failed.pending_id,
        worker_id="w-clear-unfiltered-failed",
        reason="expected-failure",
    )

    promoted = pool_store.pending.claim(
        worker_id="w-clear-unfiltered-promoted",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="clear_unfiltered_promoted", dim_b=1),
    )
    assert promoted is not None
    assert (
        pool_store.pending.promote(
            pending_id=promoted.pending_id,
            worker_id="w-clear-unfiltered-promoted",
        )
        is not None
    )

    cleared = pool_store.pending.clear_pending()
    after_counts = pool_store.pending.status_counts()

    assert cleared == before_counts.pending + 2
    assert after_counts.pending == 0
    assert after_counts.leased == before_counts.leased + 1
    assert after_counts.failed == before_counts.failed + 1
    assert after_counts.promoted == before_counts.promoted + 1
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_unfiltered_leased"),
            status=PendingStatus.leased,
        )
    ) == 1
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_unfiltered_failed"),
            status=PendingStatus.failed,
        )
    ) == 1
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_unfiltered_promoted"),
            status=PendingStatus.promoted,
        )
    ) == 1
    assert pool_store.pending.bulk_load(
        key_filter=_in_filter(
            dim_a=[
                "clear_unfiltered_pending_a",
                "clear_unfiltered_pending_b",
            ]
        ),
        status=PendingStatus.pending,
    ) == []


@pytest.mark.integration
def test_clear_pending_returns_zero_when_no_matching_pending_rows(
    pool_store: PoolStore,
) -> None:
    pending = _pending(dim_a="clear_none_pending", dim_b=1, sample_idx=0)
    pool_store.pending.insert(pending)

    leased = pool_store.pending.claim(
        worker_id="w-clear-none",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="clear_none_pending", dim_b=1),
    )
    assert leased is not None

    cleared = pool_store.pending.clear_pending(
        key_filter=_eq_filter(dim_a="clear_none_pending")
    )

    assert cleared == 0
    assert len(
        pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="clear_none_pending"),
            status=PendingStatus.leased,
        )
    ) == 1


@pytest.mark.integration
def test_bump_pending_priority(pool_store: PoolStore) -> None:
    p = _pending(dim_a="bump", dim_b=1, sample_idx=0, priority=5)
    pool_store.pending.insert(p)

    updated = pool_store.pending.bump_priority(
        key_values={"dim_a": "bump", "dim_b": 1}, priority=50
    )
    assert updated == 1


@pytest.mark.integration
def test_shuffle_priorities_randomizes_pending_rows(
    pool_store: PoolStore,
) -> None:
    samples = [
        _pending(dim_a="shuf", dim_b=1, sample_idx=i, priority=0) for i in range(20)
    ]
    pool_store.pending.insert_many(samples)

    updated = pool_store.pending.shuffle_priorities(
        key_filter=_eq_filter(dim_a="shuf")
    )
    assert updated == 20

    rows = pool_store.pending.bulk_load(key_filter=_eq_filter(dim_a="shuf"))
    priorities = [row.priority for row in rows]
    # With a 1e9 upper bound and 20 rows, ties should be vanishingly rare.
    assert len(set(priorities)) >= 18


@pytest.mark.integration
def test_shuffle_priorities_skips_leased_rows(pool_store: PoolStore) -> None:
    samples = [
        _pending(dim_a="shufleased", dim_b=1, sample_idx=i, priority=0)
        for i in range(4)
    ]
    pool_store.pending.insert_many(samples)

    leased = pool_store.pending.claim(
        worker_id="w-shuffle",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="shufleased"),
    )
    assert leased is not None

    updated = pool_store.pending.shuffle_priorities(
        key_filter=_eq_filter(dim_a="shufleased")
    )
    # 3 of 4 rows are still pending; the 1 leased row is skipped.
    assert updated == 3

    leased_row = next(
        row
        for row in pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="shufleased")
        )
        if row.pending_id == leased.pending_id
    )
    assert leased_row.priority == 0


@pytest.mark.integration
def test_shuffle_priorities_is_reproducible_with_seed(
    pool_store: PoolStore,
) -> None:
    samples = [
        _pending(dim_a="shufseed", dim_b=1, sample_idx=i, priority=0)
        for i in range(10)
    ]
    pool_store.pending.insert_many(samples)

    pool_store.pending.shuffle_priorities(
        seed=42, key_filter=_eq_filter(dim_a="shufseed")
    )
    first_pass = {
        row.pending_id: row.priority
        for row in pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="shufseed")
        )
    }

    pool_store.pending.shuffle_priorities(
        seed=42, key_filter=_eq_filter(dim_a="shufseed")
    )
    second_pass = {
        row.pending_id: row.priority
        for row in pool_store.pending.bulk_load(
            key_filter=_eq_filter(dim_a="shufseed")
        )
    }

    assert first_pass == second_pass


# --- Metadata ---


@pytest.mark.integration
def test_metadata_upsert_and_get(pool_store: PoolStore) -> None:
    pool_store.metadata.upsert("prompt_config_abc", {"blocks": ["role", "task"]})
    result = pool_store.metadata.get("prompt_config_abc")
    assert result is not None
    assert result["blocks"] == ["role", "task"]

    # Update
    pool_store.metadata.upsert(
        "prompt_config_abc", {"blocks": ["role", "task", "goal"]}
    )
    result = pool_store.metadata.get("prompt_config_abc")
    assert result is not None
    assert len(result["blocks"]) == 3


@pytest.mark.integration
def test_metadata_get_missing(pool_store: PoolStore) -> None:
    assert pool_store.metadata.get("nonexistent_key") is None


# --- Coverage ---


@pytest.mark.integration
def test_coverage(pool_store: PoolStore) -> None:
    cov = pool_store.coverage()
    assert isinstance(cov, list)
    assert all(hasattr(c, "key_values") and hasattr(c, "count") for c in cov)


# --- Bulk Load ---


@pytest.mark.integration
def test_bulk_load(pool_store: PoolStore) -> None:
    samples = pool_store.bulk_load()
    assert isinstance(samples, list)


@pytest.mark.integration
def test_bulk_load_with_filter(pool_store: PoolStore) -> None:
    # Insert some known samples
    for i in range(3):
        pool_store.insert_sample(
            _sample(
                dim_a="bload",
                dim_b=99,
                sample_idx=i,
                payload={"loaded": i},
                source_run_id=f"bulk-run-{i}",
                metadata={"batch": i},
            )
        )

    filtered = pool_store.bulk_load(key_filter=_eq_filter(dim_a="bload", dim_b=99))
    assert len(filtered) == 3
    assert all(s.key_values["dim_a"] == "bload" for s in filtered)
    assert filtered[0].payload == {"loaded": 0}
    assert filtered[0].source_run_id == "bulk-run-0"
    assert filtered[0].metadata == {"batch": 0}


@pytest.mark.integration
def test_bulk_load_pending(pool_store: PoolStore) -> None:
    pool_store.pending.insert(
        _pending(
            dim_a="blp",
            dim_b=70,
            sample_idx=0,
            payload={"partial": "first"},
            source_run_id="pending-seed-0",
            metadata={"batch": 0},
        )
    )
    pool_store.pending.insert(_pending(dim_a="blp", dim_b=70, sample_idx=1))

    results = pool_store.pending.bulk_load(
        key_filter=_eq_filter(dim_a="blp", dim_b=70)
    )
    assert len(results) == 2
    assert all(isinstance(r, PendingSample) for r in results)
    assert all(r.key_values["dim_a"] == "blp" for r in results)
    assert results[0].payload == {"partial": "first"}
    assert results[0].source_run_id == "pending-seed-0"
    assert results[0].metadata == {"batch": 0}


@pytest.mark.integration
def test_bulk_load_pending_with_filter(pool_store: PoolStore) -> None:
    pool_store.pending.insert(_pending(dim_a="blpf_a", dim_b=71, sample_idx=0))
    pool_store.pending.insert(_pending(dim_a="blpf_b", dim_b=71, sample_idx=0))

    filtered = pool_store.pending.bulk_load(key_filter=_eq_filter(dim_a="blpf_a"))
    assert len(filtered) == 1
    assert filtered[0].key_values["dim_a"] == "blpf_a"


@pytest.mark.integration
def test_bulk_load_pending_excludes_promoted(pool_store: PoolStore) -> None:
    # Insert a pending sample and promote it
    pool_store.pending.insert(_pending(dim_a="blpe", dim_b=72, sample_idx=0))
    claimed = pool_store.pending.claim(
        worker_id="w1",
        lease_seconds=300,
        key_filter=_eq_filter(dim_a="blpe", dim_b=72),
    )
    assert claimed is not None
    pool_store.pending.promote(pending_id=claimed.pending_id, worker_id="w1")

    # Insert another that stays pending
    pool_store.pending.insert(_pending(dim_a="blpe", dim_b=72, sample_idx=1))

    results = pool_store.pending.bulk_load(
        key_filter=_eq_filter(dim_a="blpe", dim_b=72)
    )
    assert len(results) == 1
    assert results[0].sample_idx == 1


# --- Validation ---


@pytest.mark.integration
def test_missing_key_raises(pool_store: PoolStore) -> None:
    with pytest.raises(PoolSchemaError, match="Missing key columns"):
        pool_store.insert_sample(PoolSample(key_values={"dim_a": "x"}))


# --- Service ---


@pytest.mark.integration
def test_service_acquire_or_generate(pool_store: PoolStore) -> None:
    service = PoolService(pool_store)

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
    result = service.acquire_or_generate(q, generator_fn=generator)
    assert len(result.samples) == 3

    # Samples should now be in the pool for future runs
    depth = pool_store.cell_depth(key_values={"dim_a": "svc", "dim_b": 42})
    assert depth >= 3


@pytest.mark.integration
def test_service_generator_error_wraps_as_topup_error(pool_store: PoolStore) -> None:
    service = PoolService(pool_store)

    def failing_generator(key_values: dict[str, Any], deficit: int) -> list[PoolSample]:
        raise RuntimeError("LLM call timed out")

    q = AcquireQuery(
        run_id="svc_fail",
        key_values={"dim_a": "svc_err", "dim_b": 99},
        n=3,
    )
    with pytest.raises(PoolTopupError, match="Top-up generation failed"):
        service.acquire_or_generate(q, generator_fn=failing_generator)


# --- Call stats ---


@pytest.mark.integration
@pytest.mark.usefixtures("pool_store")
def test_call_stats_table_created() -> None:
    dsn = _get_dsn()
    assert dsn is not None
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
            """,
            [_TEST_SCHEMA.call_stats_table],
        ).fetchone()
    assert row is not None
    assert row[0] is True


@pytest.mark.integration
def test_insert_call_stats(pool_store: PoolStore) -> None:
    s = _sample(dim_a="cs_insert", dim_b=1, sample_idx=0)
    pool_store.insert_sample(s)

    samples = pool_store.bulk_load(
        key_filter=_eq_filter(dim_a="cs_insert", dim_b=1)
    )
    assert len(samples) == 1
    sample_id = samples[0].sample_id

    stats = CallStats(
        sample_id=sample_id,
        latency_ms=1500,
        total_cost_usd=0.01,
        prompt_tokens=200,
        completion_tokens=100,
        reasoning_tokens=30,
        total_tokens=300,
        attempt_count=2,
        finish_reason="stop",
    )
    pool_store.insert_call_stats(stats)
    pool_store.insert_call_stats(stats)

    dsn = _get_dsn()
    assert dsn is not None
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            sql.SQL(
                "SELECT latency_ms, total_cost_usd, prompt_tokens, completion_tokens, "
                "reasoning_tokens, total_tokens, attempt_count, finish_reason, COUNT(*) OVER () "
                "FROM {} WHERE sample_id = %s"
            ).format(sql.Identifier("public", _TEST_SCHEMA.call_stats_table)),
            [sample_id],
        ).fetchone()
    assert row is not None
    assert row[0] == 1500
    assert row[1] == pytest.approx(0.01)
    assert row[2] == 200
    assert row[3] == 100
    assert row[4] == 30
    assert row[5] == 300
    assert row[6] == 2
    assert row[7] == "stop"
    assert row[8] == 1


@pytest.mark.integration
def test_call_stats_full_flow(pool_store: PoolStore) -> None:
    """Seed → claim → promote → insert_call_stats → verify the stored row."""
    p = _pending(
        dim_a="cs_flow",
        dim_b=1,
        sample_idx=0,
        payload={"partial": True},
    )
    pool_store.pending.insert(p)

    claimed = pool_store.pending.claim(
        worker_id="w1",
        lease_seconds=60,
        key_filter=_eq_filter(dim_a="cs_flow", dim_b=1),
    )
    assert claimed is not None

    response_payload = {
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
    }

    promoted = pool_store.pending.promote(
        pending_id=claimed.pending_id,
        worker_id="w1",
        payload=response_payload,
    )
    assert promoted is not None

    stats = CallStats.from_response(
        sample_id=promoted.sample_id,
        response=response_payload,
        attempt_count=claimed.attempt_count,
    )
    pool_store.insert_call_stats(stats)

    dsn = _get_dsn()
    assert dsn is not None
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            sql.SQL(
                "SELECT latency_ms, prompt_tokens, attempt_count, finish_reason, "
                "reasoning_tokens, total_cost_usd "
                "FROM {} WHERE sample_id = %s"
            ).format(sql.Identifier("public", _TEST_SCHEMA.call_stats_table)),
            [promoted.sample_id],
        ).fetchone()
    assert row is not None
    assert row[0] == 800
    assert row[1] == 50
    assert row[2] == 1
    assert row[3] == "stop"
    assert row[4] is None  # reasoning_tokens=0 not stored
    assert row[5] == pytest.approx(0.003)

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
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.pool.store_ops.request_update import RequestUpdate

pytestmark = pytest.mark.integration


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv(
        "DR_LLM_DATABASE_URL"
    )


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
        conn.execute(
            "DELETE FROM pool_catalog WHERE pool_name = %s", [schema.name]
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
                sql.Identifier(
                    "public", schema.table_name(PoolTableType.LEASES)
                )
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
            ).format(
                sql.Identifier(
                    "public", schema.table_name(PoolTableType.LEASES)
                )
            ),
            [sample_id],
        )
        conn.commit()


@pytest.fixture
def pool_store() -> Generator[PoolStore, None, None]:
    dsn = _get_dsn()
    if not dsn:
        pytest.skip(
            "Set DR_LLM_TEST_DATABASE_URL to run pool integration tests"
        )

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


def test_ensure_schema_creates_tables_idempotently(
    pool_store: PoolStore,
) -> None:
    dsn = _get_dsn()
    assert dsn is not None

    pool_store.ensure_schema()
    pool_store.ensure_schema()

    assert (
        _table_exists(dsn, pool_store.schema.table_name(PoolTableType.SAMPLES))
        is True
    )
    assert (
        _table_exists(dsn, pool_store.schema.table_name(PoolTableType.LEASES))
        is True
    )


def test_insert_samples_and_complete_sample(pool_store: PoolStore) -> None:
    sample = _sample(dim_a="complete", dim_b=1, sample_id="complete-1")
    assert pool_store.insert_sample(sample) is True
    [stored] = pool_store.bulk_load(
        key_filter=PoolKeyFilter.eq(dim_a="complete")
    )
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

    [completed] = pool_store.bulk_load(
        key_filter=PoolKeyFilter.eq(dim_a="complete")
    )
    assert completed.is_complete is True
    assert completed.response == {"text": "done"}
    assert completed.finish_reason == "stop"
    assert completed.attempt_count == 2


def test_complete_sample_with_lease_owner_requires_active_owner(
    pool_store: PoolStore,
) -> None:
    sample = _sample(dim_a="lease_complete", sample_id="lease-complete-1")
    assert pool_store.insert_sample(sample) is True
    claimed = pool_store.claim_lease(worker_id="worker-a", lease_seconds=60)
    assert claimed is not None

    assert (
        pool_store.complete_sample(
            sample_id=claimed.sample_id,
            response={"text": "wrong worker"},
            finish_reason="stop",
            attempt_count=claimed.attempt_count,
            lease_owner="worker-b",
        )
        is False
    )
    assert (
        pool_store.complete_sample(
            sample_id=claimed.sample_id,
            response={"text": "done"},
            finish_reason="stop",
            attempt_count=claimed.attempt_count,
            lease_owner="worker-a",
        )
        is True
    )

    [completed] = pool_store.bulk_load(
        key_filter=PoolKeyFilter.eq(dim_a="lease_complete")
    )
    assert completed.response == {"text": "done"}
    assert completed.attempt_count == 1


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
    assert first.attempt_count == 1
    assert second.attempt_count == 1
    assert first.sample_id != second.sample_id
    assert third is None
    dsn = _get_dsn()
    assert dsn is not None
    assert _lease_worker(dsn, pool_store.schema, first.sample_id) == "worker-a"


def test_release_lease_makes_sample_claimable_again(
    pool_store: PoolStore,
) -> None:
    sample = _sample(dim_a="release", sample_id="release-1")
    assert pool_store.insert_sample(sample) is True
    claimed = pool_store.claim_lease(worker_id="worker-a", lease_seconds=60)
    assert claimed is not None

    assert (
        pool_store.release_lease(
            sample_id=claimed.sample_id, worker_id="wrong-worker"
        )
        is False
    )
    assert (
        pool_store.release_lease(
            sample_id=claimed.sample_id, worker_id="worker-a"
        )
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
    assert (
        _lease_worker(dsn, pool_store.schema, claimed.sample_id) == "worker-b"
    )


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

    rows = pool_store.bulk_load(
        key_filter=PoolKeyFilter.eq(dim_a="auto", dim_b=1)
    )
    assert sorted(row.sample_idx for row in rows) == [0, 1, 2]


# --- Progress / error / leased count tests ---


def test_progress_all_states(pool_store: PoolStore) -> None:
    samples = [
        _sample(dim_a="prog", dim_b=1, sample_id="prog-1", sample_idx=0),
        _sample(dim_a="prog", dim_b=1, sample_id="prog-2", sample_idx=1),
        _sample(dim_a="prog", dim_b=1, sample_id="prog-3", sample_idx=2),
        _sample(dim_a="prog", dim_b=2, sample_id="prog-4", sample_idx=0),
    ]
    assert pool_store.insert_samples(samples).inserted == 4

    pool_store.complete_sample(
        sample_id="prog-1",
        response={"text": "ok"},
        finish_reason="stop",
        attempt_count=1,
    )
    pool_store.complete_sample(
        sample_id="prog-2",
        response={"error": "boom"},
        finish_reason="error",
        attempt_count=3,
    )
    pool_store.claim_lease(worker_id="w1", lease_seconds=300)

    p = pool_store.progress()
    assert p.total == 4
    assert p.complete == 2
    assert p.incomplete == 2
    assert p.error == 1
    assert p.leased == 1


def test_error_count_with_key_filter(pool_store: PoolStore) -> None:
    samples = [
        _sample(dim_a="errcnt", dim_b=1, sample_id="errcnt-1", sample_idx=0),
        _sample(dim_a="errcnt", dim_b=2, sample_id="errcnt-2", sample_idx=0),
    ]
    assert pool_store.insert_samples(samples).inserted == 2
    pool_store.complete_sample(
        sample_id="errcnt-1",
        response={"error": "fail"},
        finish_reason="error",
        attempt_count=1,
    )
    pool_store.complete_sample(
        sample_id="errcnt-2",
        response={"error": "fail"},
        finish_reason="error",
        attempt_count=1,
    )

    assert pool_store.error_count() == 2
    assert (
        pool_store.error_count(
            key_filter=PoolKeyFilter.eq(dim_a="errcnt", dim_b=1)
        )
        == 1
    )


def test_leased_count_reflects_active_leases(pool_store: PoolStore) -> None:
    dsn = _get_dsn()
    assert dsn is not None
    samples = [
        _sample(dim_a="lcnt", dim_b=1, sample_id="lcnt-1", sample_idx=0),
        _sample(dim_a="lcnt", dim_b=1, sample_id="lcnt-2", sample_idx=1),
    ]
    assert pool_store.insert_samples(samples).inserted == 2

    pool_store.claim_lease(worker_id="w1", lease_seconds=300)
    claimed2 = pool_store.claim_lease(worker_id="w2", lease_seconds=300)
    assert claimed2 is not None
    assert pool_store.leased_count() == 2

    _expire_lease(dsn, pool_store.schema, claimed2.sample_id)
    assert pool_store.leased_count() == 1


# --- Completion filter tests ---


def test_bulk_load_completion_filter(pool_store: PoolStore) -> None:
    samples = [
        _sample(dim_a="cf", dim_b=1, sample_id="cf-1", sample_idx=0),
        _sample(dim_a="cf", dim_b=1, sample_id="cf-2", sample_idx=1),
        _sample(dim_a="cf", dim_b=1, sample_id="cf-3", sample_idx=2),
    ]
    assert pool_store.insert_samples(samples).inserted == 3
    pool_store.complete_sample(
        sample_id="cf-1",
        response={"text": "ok"},
        finish_reason="stop",
        attempt_count=1,
    )
    pool_store.complete_sample(
        sample_id="cf-2",
        response={"error": "fail"},
        finish_reason="error",
        attempt_count=2,
    )

    kf = PoolKeyFilter.eq(dim_a="cf")
    assert len(pool_store.bulk_load(key_filter=kf, completion="all")) == 3
    assert len(pool_store.bulk_load(key_filter=kf, completion="complete")) == 2
    assert (
        len(pool_store.bulk_load(key_filter=kf, completion="incomplete")) == 1
    )
    errors = pool_store.bulk_load(key_filter=kf, completion="error")
    assert len(errors) == 1
    assert errors[0].sample_id == "cf-2"


def test_iter_samples_completion_filter(pool_store: PoolStore) -> None:
    samples = [
        _sample(dim_a="cfi", dim_b=1, sample_id="cfi-1", sample_idx=0),
        _sample(dim_a="cfi", dim_b=1, sample_id="cfi-2", sample_idx=1),
    ]
    assert pool_store.insert_samples(samples).inserted == 2
    pool_store.complete_sample(
        sample_id="cfi-1",
        response={"text": "ok"},
        finish_reason="stop",
        attempt_count=1,
    )

    kf = PoolKeyFilter.eq(dim_a="cfi")
    incomplete = list(
        pool_store.iter_samples(key_filter=kf, completion="incomplete")
    )
    assert len(incomplete) == 1
    assert incomplete[0].sample_id == "cfi-2"


# --- Request update tests ---


def test_update_incomplete_request_succeeds(pool_store: PoolStore) -> None:
    sample = _sample(dim_a="upd", dim_b=1, sample_id="upd-1")
    pool_store.insert_sample(sample)

    assert pool_store.update_incomplete_request(
        sample_id="upd-1", request={"prompt": "new"}
    )
    [s] = pool_store.bulk_load(key_filter=PoolKeyFilter.eq(dim_a="upd"))
    assert s.request == {"prompt": "new"}


def test_update_incomplete_request_noop_for_complete(
    pool_store: PoolStore,
) -> None:
    sample = _sample(dim_a="updc", dim_b=1, sample_id="updc-1")
    pool_store.insert_sample(sample)
    pool_store.complete_sample(
        sample_id="updc-1",
        response={"text": "done"},
        finish_reason="stop",
        attempt_count=1,
    )

    assert not pool_store.update_incomplete_request(
        sample_id="updc-1", request={"prompt": "should not apply"}
    )


def test_update_incomplete_requests_batch(pool_store: PoolStore) -> None:
    samples = [
        _sample(dim_a="batch", dim_b=1, sample_id="batch-1", sample_idx=0),
        _sample(dim_a="batch", dim_b=1, sample_id="batch-2", sample_idx=1),
    ]
    pool_store.insert_samples(samples)

    count = pool_store.update_incomplete_requests(
        [
            RequestUpdate(sample_id="batch-1", request={"v": 1}),
            RequestUpdate(sample_id="batch-2", request={"v": 2}),
        ]
    )
    assert count == 2


def test_update_incomplete_request_with_metadata(
    pool_store: PoolStore,
) -> None:
    sample = _sample(dim_a="updm", dim_b=1, sample_id="updm-1")
    pool_store.insert_sample(sample)

    pool_store.update_incomplete_request(
        sample_id="updm-1",
        request={"prompt": "new"},
        metadata={"source": "repair"},
    )
    [s] = pool_store.bulk_load(key_filter=PoolKeyFilter.eq(dim_a="updm"))
    assert s.request == {"prompt": "new"}
    assert s.metadata == {"source": "repair"}


# --- Catalog tests ---


def test_ensure_schema_creates_catalog_entry(pool_store: PoolStore) -> None:
    from dr_llm.pool.db.catalog import load_schema

    loaded = load_schema(pool_store._runtime, pool_store.schema.name)
    assert loaded is not None
    assert loaded == pool_store.schema


def test_catalog_load_schema_round_trip(pool_store: PoolStore) -> None:
    from dr_llm.pool.db.catalog import load_schema

    loaded = load_schema(pool_store._runtime, pool_store.schema.name)
    assert loaded is not None
    assert loaded.name == pool_store.schema.name
    assert loaded.key_columns == pool_store.schema.key_columns


def test_catalog_list_pool_names(pool_store: PoolStore) -> None:
    from dr_llm.pool.db.catalog import list_pool_names

    names = list_pool_names(pool_store._runtime)
    assert pool_store.schema.name in names


# --- Requeue / reset tests ---


def test_requeue_errors_clears_response_and_leases(
    pool_store: PoolStore,
) -> None:
    sample = _sample(dim_a="rq", dim_b=1, sample_id="rq-1")
    pool_store.insert_sample(sample)
    pool_store.complete_sample(
        sample_id="rq-1",
        response={"error": "boom"},
        finish_reason="error",
        attempt_count=3,
    )
    assert pool_store.error_count() >= 1

    requeued = pool_store.requeue_errors()
    assert requeued >= 1

    [s] = pool_store.bulk_load(key_filter=PoolKeyFilter.eq(dim_a="rq"))
    assert s.response is None
    assert s.finish_reason is None
    assert s.attempt_count == 0


def test_requeue_errors_respects_key_filter(pool_store: PoolStore) -> None:
    samples = [
        _sample(dim_a="rqf", dim_b=1, sample_id="rqf-1", sample_idx=0),
        _sample(dim_a="rqf", dim_b=2, sample_id="rqf-2", sample_idx=0),
    ]
    pool_store.insert_samples(samples)
    for sid in ("rqf-1", "rqf-2"):
        pool_store.complete_sample(
            sample_id=sid,
            response={"error": "fail"},
            finish_reason="error",
            attempt_count=1,
        )

    requeued = pool_store.requeue_errors(
        key_filter=PoolKeyFilter.eq(dim_a="rqf", dim_b=1)
    )
    assert requeued == 1
    assert (
        pool_store.error_count(key_filter=PoolKeyFilter.eq(dim_a="rqf")) == 1
    )


def test_requeue_errors_skips_successful_completions(
    pool_store: PoolStore,
) -> None:
    sample = _sample(dim_a="rqs", dim_b=1, sample_id="rqs-1")
    pool_store.insert_sample(sample)
    pool_store.complete_sample(
        sample_id="rqs-1",
        response={"text": "ok"},
        finish_reason="stop",
        attempt_count=1,
    )

    assert (
        pool_store.requeue_errors(key_filter=PoolKeyFilter.eq(dim_a="rqs"))
        == 0
    )


def test_reset_samples_by_id(pool_store: PoolStore) -> None:
    sample = _sample(dim_a="rst", dim_b=1, sample_id="rst-1")
    pool_store.insert_sample(sample)
    pool_store.complete_sample(
        sample_id="rst-1",
        response={"text": "done"},
        finish_reason="stop",
        attempt_count=1,
    )

    reset = pool_store.reset_samples(sample_ids=["rst-1"])
    assert reset == 1

    [s] = pool_store.bulk_load(key_filter=PoolKeyFilter.eq(dim_a="rst"))
    assert s.response is None
    assert s.attempt_count == 0


def test_reset_samples_with_new_request(pool_store: PoolStore) -> None:
    sample = _sample(dim_a="rstn", dim_b=1, sample_id="rstn-1")
    pool_store.insert_sample(sample)
    pool_store.complete_sample(
        sample_id="rstn-1",
        response={"text": "done"},
        finish_reason="stop",
        attempt_count=1,
    )

    pool_store.reset_samples(
        sample_ids=["rstn-1"],
        reset_request={"prompt": "retry"},
    )

    [s] = pool_store.bulk_load(key_filter=PoolKeyFilter.eq(dim_a="rstn"))
    assert s.request == {"prompt": "retry"}
    assert s.response is None

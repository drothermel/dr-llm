"""Integration tests for pool store operations (requires PostgreSQL)."""

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
from dr_llm.pool.errors import PoolSchemaError, PoolTopupError
from dr_llm.pool.models import AcquireQuery
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatus
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_service import PoolService
from dr_llm.pool.sample_store import PoolStore


_TEST_SCHEMA = PoolSchema(
    name="itest",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)

_POOL_TABLES = (
    _TEST_SCHEMA.metadata_table,
    _TEST_SCHEMA.claims_table,
    _TEST_SCHEMA.pending_table,
    _TEST_SCHEMA.samples_table,
)

_LEGACY_TABLES = (
    "artifacts",
    "llm_call_responses",
    "llm_call_requests",
    "llm_calls",
    "run_parameters",
    "runs",
    "tool_call_dead_letters",
    "tool_results",
    "tool_calls",
    "session_events",
    "session_turns",
    "sessions",
)


def _drop_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        for tbl in _POOL_TABLES:
            conn.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(tbl))
            )
        conn.commit()


def _get_dsn() -> str | None:
    return os.getenv("DR_LLM_TEST_DATABASE_URL") or os.getenv("DR_LLM_DATABASE_URL")


def _create_legacy_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        for table_name in _LEGACY_TABLES:
            conn.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} (id TEXT)").format(
                    sql.Identifier(table_name)
                )
            )
        conn.commit()


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
        store.init_schema()
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
def test_store_init_drops_legacy_call_recorder_tables(pool_store: PoolStore) -> None:
    dsn = _get_dsn()
    assert dsn is not None
    _create_legacy_tables(dsn)

    fresh_runtime = DbRuntime(
        DbConfig(
            dsn=dsn,
            min_pool_size=1,
            max_pool_size=4,
            application_name="pool_tests_cleanup",
        )
    )
    fresh_store = PoolStore(_TEST_SCHEMA, fresh_runtime)
    try:
        fresh_store.init_schema()
    finally:
        fresh_runtime.close()

    with psycopg.connect(dsn) as conn:
        for table_name in _LEGACY_TABLES:
            exists = conn.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """,
                [table_name],
            ).fetchone()
            assert exists is not None
            assert exists[0] is False


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

    assert pool_store.remaining(run_id="run_rem", key_values={"dim_a": "rem", "dim_b": 1}) == 3

    pool_store.acquire(AcquireQuery(run_id="run_rem", key_values={"dim_a": "rem", "dim_b": 1}, n=1))
    assert pool_store.remaining(run_id="run_rem", key_values={"dim_a": "rem", "dim_b": 1}) == 2


# --- Pending ---


@pytest.mark.integration
def test_pending_insert_and_count(pool_store: PoolStore) -> None:
    p = _pending(dim_a="pend", dim_b=1)
    assert pool_store.pending.insert_pending(p) is True
    assert pool_store.pending.pending_counts(key_values={"dim_a": "pend", "dim_b": 1}) == 1


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
    pool_store.pending.insert_pending(p)

    claimed = pool_store.pending.claim_pending(
        worker_id="w1", limit=1, lease_seconds=60,
        key_filter={"dim_a": "promote", "dim_b": 1},
    )
    assert len(claimed) == 1
    assert claimed[0].status == PendingStatus.leased
    assert claimed[0].worker_id == "w1"
    assert claimed[0].payload == {"partial": True, "draft": 1}
    assert claimed[0].source_run_id == "pending-run"
    assert claimed[0].metadata == {"source": "seed"}
    assert claimed[0].attempt_count == 1

    # Promote with final payload
    sample = pool_store.pending.promote_pending(
        pending_id=claimed[0].pending_id,
        payload={"final": True, "score": 0.95},
    )
    assert sample is not None
    assert sample.payload["final"] is True
    assert sample.source_run_id == "pending-run"
    assert sample.metadata == {"source": "seed"}
    assert pool_store.cell_depth(key_values={"dim_a": "promote", "dim_b": 1}) == 1


@pytest.mark.integration
def test_pending_fail(pool_store: PoolStore) -> None:
    p = _pending(dim_a="fail", dim_b=1, sample_idx=0)
    pool_store.pending.insert_pending(p)

    claimed = pool_store.pending.claim_pending(
        worker_id="w1", limit=1, lease_seconds=60,
        key_filter={"dim_a": "fail", "dim_b": 1},
    )
    pool_store.pending.fail_pending(pending_id=claimed[0].pending_id, worker_id="w1", reason="docker timeout")

    # Failed samples should not be counted as pending
    assert pool_store.pending.pending_counts(key_values={"dim_a": "fail", "dim_b": 1}) == 0


@pytest.mark.integration
def test_bump_pending_priority(pool_store: PoolStore) -> None:
    p = _pending(dim_a="bump", dim_b=1, sample_idx=0, priority=5)
    pool_store.pending.insert_pending(p)

    updated = pool_store.pending.bump_pending_priority(
        key_values={"dim_a": "bump", "dim_b": 1}, priority=50
    )
    assert updated == 1


# --- Metadata ---


@pytest.mark.integration
def test_metadata_upsert_and_get(pool_store: PoolStore) -> None:
    pool_store.metadata.upsert_metadata("prompt_config_abc", {"blocks": ["role", "task"]})
    result = pool_store.metadata.get_metadata("prompt_config_abc")
    assert result is not None
    assert result["blocks"] == ["role", "task"]

    # Update
    pool_store.metadata.upsert_metadata("prompt_config_abc", {"blocks": ["role", "task", "goal"]})
    result = pool_store.metadata.get_metadata("prompt_config_abc")
    assert result is not None
    assert len(result["blocks"]) == 3


@pytest.mark.integration
def test_metadata_get_missing(pool_store: PoolStore) -> None:
    assert pool_store.metadata.get_metadata("nonexistent_key") is None


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

    filtered = pool_store.bulk_load(key_filter={"dim_a": "bload", "dim_b": 99})
    assert len(filtered) == 3
    assert all(s.key_values["dim_a"] == "bload" for s in filtered)
    assert filtered[0].payload == {"loaded": 0}
    assert filtered[0].source_run_id == "bulk-run-0"
    assert filtered[0].metadata == {"batch": 0}


@pytest.mark.integration
def test_bulk_load_pending(pool_store: PoolStore) -> None:
    pool_store.pending.insert_pending(
        _pending(
            dim_a="blp",
            dim_b=70,
            sample_idx=0,
            payload={"partial": "first"},
            source_run_id="pending-seed-0",
            metadata={"batch": 0},
        )
    )
    pool_store.pending.insert_pending(_pending(dim_a="blp", dim_b=70, sample_idx=1))

    results = pool_store.pending.bulk_load_pending(key_filter={"dim_a": "blp", "dim_b": 70})
    assert len(results) == 2
    assert all(isinstance(r, PendingSample) for r in results)
    assert all(r.key_values["dim_a"] == "blp" for r in results)
    assert results[0].payload == {"partial": "first"}
    assert results[0].source_run_id == "pending-seed-0"
    assert results[0].metadata == {"batch": 0}


@pytest.mark.integration
def test_bulk_load_pending_with_filter(pool_store: PoolStore) -> None:
    pool_store.pending.insert_pending(_pending(dim_a="blpf_a", dim_b=71, sample_idx=0))
    pool_store.pending.insert_pending(_pending(dim_a="blpf_b", dim_b=71, sample_idx=0))

    filtered = pool_store.pending.bulk_load_pending(key_filter={"dim_a": "blpf_a"})
    assert len(filtered) == 1
    assert filtered[0].key_values["dim_a"] == "blpf_a"


@pytest.mark.integration
def test_bulk_load_pending_excludes_promoted(pool_store: PoolStore) -> None:
    # Insert a pending sample and promote it
    pool_store.pending.insert_pending(_pending(dim_a="blpe", dim_b=72, sample_idx=0))
    claimed = pool_store.pending.claim_pending(
        worker_id="w1", limit=1, lease_seconds=300,
        key_filter={"dim_a": "blpe", "dim_b": 72},
    )
    assert len(claimed) == 1
    pool_store.pending.promote_pending(pending_id=claimed[0].pending_id)

    # Insert another that stays pending
    pool_store.pending.insert_pending(_pending(dim_a="blpe", dim_b=72, sample_idx=1))

    results = pool_store.pending.bulk_load_pending(key_filter={"dim_a": "blpe", "dim_b": 72})
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

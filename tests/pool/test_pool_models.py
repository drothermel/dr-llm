"""Unit tests for pool models."""

from __future__ import annotations

from datetime import UTC, datetime
from pydantic import BaseModel

import pytest

from dr_llm.pool.db.schema import ColumnType, KeyColumn, PoolSchema
from dr_llm.pool.models import (
    AcquireQuery,
    AcquireResult,
    InsertResult,
)
from dr_llm.pool.pending.backend import PoolPendingBackendState
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatus, PendingStatusCounts
from dr_llm.pool.pool_sample import PoolSample, SampleStatus
from dr_llm.workers import WorkerSnapshot
from pydantic import ValidationError

_TEST_SCHEMA = PoolSchema(
    name="modeltest",
    key_columns=[
        KeyColumn(name="dim_a"),
        KeyColumn(name="dim_b", type=ColumnType.integer),
    ],
)


def test_pool_sample_defaults() -> None:
    s = PoolSample(key_values={"x": "a"})
    assert s.sample_id  # auto-generated
    assert s.sample_idx is None
    assert s.payload == {}
    assert s.status == SampleStatus.active


def test_acquire_result_deficit() -> None:
    r = AcquireResult(samples=[])
    assert r.claimed == 0
    assert r.deficit(5) == 5

    r2 = AcquireResult(samples=[PoolSample(key_values={"x": "a"})] * 3)
    assert r2.claimed == 3
    assert r2.deficit(5) == 2
    assert r2.deficit(3) == 0
    assert r2.deficit(1) == 0


def test_pending_sample_defaults() -> None:
    p = PendingSample(key_values={"x": "a"})
    assert p.status == PendingStatus.pending
    assert p.priority == 0
    assert p.attempt_count == 0


def test_pool_sample_to_db_insert_row_splats_key_values() -> None:
    sample = PoolSample(
        sample_id="sample-1",
        sample_idx=7,
        key_values={"dim_b": 3, "dim_a": "alpha"},
        payload={"score": 0.9},
        source_run_id="run-1",
        metadata={"source": "test"},
        status=SampleStatus.superseded,
    )

    row = sample.to_db_insert_row()

    assert set(row.keys()) == {
        "sample_id",
        "dim_a",
        "dim_b",
        "sample_idx",
        "payload_json",
        "source_run_id",
        "metadata_json",
        "status",
    }
    assert row["sample_id"] == "sample-1"
    assert row["sample_idx"] == 7
    assert row["dim_a"] == "alpha"
    assert row["dim_b"] == 3
    assert row["payload_json"] == {"score": 0.9}
    assert row["source_run_id"] == "run-1"
    assert row["metadata_json"] == {"source": "test"}
    assert row["status"] == SampleStatus.superseded.value


def test_pool_sample_to_db_insert_row_json_serializes_nested_values() -> None:
    class RichPayload(BaseModel):
        when: datetime

    sample = PoolSample(
        key_values={"dim_a": "alpha", "dim_b": 3},
        payload={"rich": RichPayload(when=datetime(2024, 1, 2, tzinfo=UTC))},
        metadata={"created_at": datetime(2024, 1, 3, tzinfo=UTC)},
    )

    row = sample.to_db_insert_row()

    assert row["payload_json"] == {"rich": {"when": "2024-01-02T00:00:00Z"}}
    assert row["metadata_json"] == {"created_at": "2024-01-03T00:00:00Z"}


def test_pool_sample_from_db_row_parses_dynamic_columns_and_json() -> None:
    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    sample = PoolSample.from_db_row(
        _TEST_SCHEMA,
        {
            "sample_id": "sample-1",
            "dim_a": "alpha",
            "dim_b": 3,
            "sample_idx": 7,
            "payload_json": {"score": 0.9},
            "source_run_id": "run-1",
            "metadata_json": {"source": "test"},
            "status": "superseded",
            "created_at": created_at,
        },
    )

    assert sample.sample_id == "sample-1"
    assert sample.key_values == {"dim_a": "alpha", "dim_b": 3}
    assert sample.payload == {"score": 0.9}
    assert sample.metadata == {"source": "test"}
    assert sample.status == SampleStatus.superseded
    assert sample.created_at == created_at


def test_pending_sample_to_db_insert_row_splats_key_values() -> None:
    sample = PendingSample(
        pending_id="pending-1",
        key_values={"dim_b": 4, "dim_a": "beta"},
        sample_idx=2,
        payload={"partial": True},
        source_run_id="run-2",
        metadata={"attempt": 1},
        priority=9,
        status=PendingStatus.leased,
    )

    row = sample.to_db_insert_row()

    assert set(row.keys()) == {
        "pending_id",
        "dim_a",
        "dim_b",
        "sample_idx",
        "payload_json",
        "source_run_id",
        "metadata_json",
        "priority",
        "status",
    }
    assert row["pending_id"] == "pending-1"
    assert row["sample_idx"] == 2
    assert row["dim_a"] == "beta"
    assert row["dim_b"] == 4
    assert row["payload_json"] == {"partial": True}
    assert row["source_run_id"] == "run-2"
    assert row["metadata_json"] == {"attempt": 1}
    assert row["priority"] == 9
    assert row["status"] == PendingStatus.leased.value


def test_pending_sample_to_db_insert_row_json_serializes_nested_values() -> None:
    class RichPayload(BaseModel):
        when: datetime

    sample = PendingSample(
        key_values={"dim_a": "beta", "dim_b": 4},
        payload={"rich": RichPayload(when=datetime(2024, 6, 7, tzinfo=UTC))},
        metadata={"updated_at": datetime(2024, 6, 8, tzinfo=UTC)},
    )

    row = sample.to_db_insert_row()

    assert row["payload_json"] == {"rich": {"when": "2024-06-07T00:00:00Z"}}
    assert row["metadata_json"] == {"updated_at": "2024-06-08T00:00:00Z"}


def test_pending_sample_from_db_row_parses_dynamic_columns_and_json() -> None:
    created_at = datetime(2024, 6, 7, 8, 9, 10, tzinfo=UTC)
    lease_expires_at = datetime(2024, 6, 7, 8, 14, 10, tzinfo=UTC)
    sample = PendingSample.from_db_row(
        _TEST_SCHEMA,
        {
            "pending_id": "pending-1",
            "dim_a": "beta",
            "dim_b": 4,
            "sample_idx": 2,
            "payload_json": {"partial": True},
            "source_run_id": "run-2",
            "metadata_json": {"attempt": 1},
            "priority": 9,
            "status": "leased",
            "worker_id": "worker-1",
            "lease_expires_at": lease_expires_at,
            "attempt_count": 3,
            "created_at": created_at,
        },
    )

    assert sample.pending_id == "pending-1"
    assert sample.key_values == {"dim_a": "beta", "dim_b": 4}
    assert sample.payload == {"partial": True}
    assert sample.metadata == {"attempt": 1}
    assert sample.priority == 9
    assert sample.status == PendingStatus.leased
    assert sample.worker_id == "worker-1"
    assert sample.lease_expires_at == lease_expires_at
    assert sample.attempt_count == 3
    assert sample.created_at == created_at


def test_insert_result_defaults() -> None:
    r = InsertResult()
    assert r.inserted == 0
    assert r.skipped == 0
    assert r.failed == 0


def test_pending_status_counts_total() -> None:
    counts = PendingStatusCounts(pending=1, leased=2, promoted=3, failed=4)
    assert counts.total == 10


def test_pending_status_counts_in_flight() -> None:
    counts = PendingStatusCounts(pending=3, leased=2, promoted=10, failed=1)
    assert counts.in_flight == 5


def test_pending_status_counts_from_rows() -> None:
    rows = [
        {"status": "pending", "cnt": 3},
        {"status": "leased", "cnt": 2},
        {"status": "promoted", "cnt": 5},
        {"status": "failed", "cnt": 1},
    ]
    counts = PendingStatusCounts.from_rows(rows)
    assert counts.pending == 3
    assert counts.leased == 2
    assert counts.promoted == 5
    assert counts.failed == 1


def test_pending_status_counts_from_rows_handles_partial_and_unknown() -> None:
    rows = [
        {"status": "pending", "cnt": 4},
        {"status": "unknown_status", "cnt": 99},
    ]
    counts = PendingStatusCounts.from_rows(rows)
    assert counts.pending == 4
    assert counts.leased == 0
    assert counts.promoted == 0
    assert counts.failed == 0


def test_acquire_query_auto_request_id() -> None:
    q = AcquireQuery(run_id="r1", key_values={"x": "a"}, n=5)
    assert q.request_id  # auto-generated


def test_acquire_query_rejects_negative_n() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AcquireQuery(run_id="r1", key_values={"x": "a"}, n=-1)


def test_worker_snapshot_defaults() -> None:
    snapshot = WorkerSnapshot[PoolPendingBackendState](
        worker_count=2,
        backend_state=PoolPendingBackendState(),
    )
    assert snapshot.stop_requested is False
    assert snapshot.counts.claimed == 0
    assert snapshot.backend_state is not None
    assert snapshot.backend_state.status_counts.total == 0


def test_pool_root_re_exports_admin_models_and_services() -> None:
    import dr_llm.pool as pool

    assert not hasattr(pool, "PoolStore")
    assert not hasattr(pool, "PoolService")
    assert not hasattr(pool, "PoolSchema")
    assert hasattr(pool, "AcquireQuery")
    assert hasattr(pool, "AcquireResult")
    assert hasattr(pool, "CreatePoolRequest")
    assert hasattr(pool, "PoolInspection")
    assert hasattr(pool, "assess_pool_creation")
    assert hasattr(pool, "create_pool")
    assert hasattr(pool, "discover_pools")
    assert not hasattr(pool, "PendingSample")
    assert not hasattr(pool, "PoolDb")


def test_pending_and_db_packages_have_no_re_exports() -> None:
    import dr_llm.pool.db as pool_db
    import dr_llm.pool.pending as pending

    assert not hasattr(pool_db, "PoolDb")
    assert not hasattr(pool_db, "DbConfig")
    assert not hasattr(pool_db, "RunStatus")
    assert not hasattr(pending, "PendingSample")
    assert not hasattr(pending, "PendingStore")

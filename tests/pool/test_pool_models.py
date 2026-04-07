"""Unit tests for pool models."""

from __future__ import annotations

from datetime import UTC, datetime

from dr_llm.pool.db import RecordedCall, RunStatus
from dr_llm.pool.models import AcquireQuery, AcquireResult, PoolSample, SampleStatus
from dr_llm.pool.pending import PendingSample, PendingStatus, PendingStatusCounts, WorkerSnapshot
from dr_llm.pool.results import InsertResult
from dr_llm.providers.models import CallMode


def test_pool_sample_defaults() -> None:
    s = PoolSample(key_values={"x": "a"})
    assert s.sample_id  # auto-generated
    assert s.sample_idx is None
    assert s.payload == {}
    assert s.status == SampleStatus.active


def test_acquire_result_deficit() -> None:
    r = AcquireResult(samples=[], claimed=0)
    assert r.deficit(5) == 5

    r2 = AcquireResult(
        samples=[PoolSample(key_values={"x": "a"})] * 3,
        claimed=3,
    )
    assert r2.deficit(5) == 2
    assert r2.deficit(3) == 0
    assert r2.deficit(1) == 0


def test_pending_sample_defaults() -> None:
    p = PendingSample(key_values={"x": "a"})
    assert p.status == PendingStatus.pending
    assert p.priority == 0
    assert p.attempt_count == 0


def test_insert_result_defaults() -> None:
    r = InsertResult()
    assert r.inserted == 0
    assert r.skipped == 0
    assert r.failed == 0


def test_pending_status_counts_total() -> None:
    counts = PendingStatusCounts(pending=1, leased=2, promoted=3, failed=4)
    assert counts.total == 10


def test_acquire_query_auto_request_id() -> None:
    q = AcquireQuery(run_id="r1", key_values={"x": "a"}, n=5)
    assert q.request_id  # auto-generated


def test_worker_snapshot_defaults() -> None:
    snapshot = WorkerSnapshot(worker_count=2)
    assert snapshot.stop_requested is False
    assert snapshot.status_counts.total == 0


def test_recorded_call_coerces_status_to_enum() -> None:
    call = RecordedCall(
        call_id="call_123",
        run_id="run_123",
        provider="openai",
        model="gpt-4.1",
        mode=CallMode.api,
        status="success",  # type: ignore[arg-type]  # raw string, not enum
        created_at=datetime(2026, 3, 28, 12, 0, tzinfo=UTC),
        latency_ms=12,
        error_text=None,
        request={},
        response=None,
    )
    assert call.status is RunStatus.success


def test_pool_root_exports_are_narrow() -> None:
    import dr_llm.pool as pool

    assert hasattr(pool, "PoolStore")
    assert hasattr(pool, "PoolService")
    assert hasattr(pool, "PoolSchema")
    assert hasattr(pool, "AcquireQuery")
    assert not hasattr(pool, "PendingSample")
    assert not hasattr(pool, "PoolDb")


def test_pending_and_db_package_exports() -> None:
    import dr_llm.pool.db as pool_db
    import dr_llm.pool.pending as pending

    assert hasattr(pool_db, "PoolDb")
    assert hasattr(pool_db, "DbConfig")
    assert hasattr(pool_db, "RunStatus")
    assert hasattr(pending, "PendingSample")
    assert hasattr(pending, "PendingStore")
    assert not hasattr(pending, "seed_pending")

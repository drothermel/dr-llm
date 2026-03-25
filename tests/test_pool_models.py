"""Unit tests for pool models."""

from __future__ import annotations

from dr_llm.pool.models import (
    AcquireQuery,
    AcquireResult,
    InsertResult,
    PendingSample,
    PendingStatus,
    PoolSample,
    SampleStatus,
)


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


def test_acquire_query_auto_request_id() -> None:
    q = AcquireQuery(run_id="r1", key_values={"x": "a"}, n=5)
    assert q.request_id  # auto-generated

"""Unit tests for sampling acquisition models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_llm.pool.pool_sample import PoolSample
from dr_llm.sampling.acquisition import AcquireQuery, AcquireResult


def test_acquire_result_deficit() -> None:
    r = AcquireResult(samples=[])
    assert r.claimed == 0
    assert r.deficit(5) == 5

    r2 = AcquireResult(samples=[PoolSample(key_values={"x": "a"})] * 3)
    assert r2.claimed == 3
    assert r2.deficit(5) == 2
    assert r2.deficit(3) == 0
    assert r2.deficit(1) == 0


def test_acquire_query_auto_request_id() -> None:
    q = AcquireQuery(run_id="r1", key_values={"x": "a"}, n=5)
    assert q.request_id  # auto-generated


def test_acquire_query_rejects_negative_n() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AcquireQuery(run_id="r1", key_values={"x": "a"}, n=-1)

from __future__ import annotations

from unittest.mock import MagicMock

from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.insert_result import InsertResult
from dr_llm.sampling.acquisition import AcquireQuery, AcquireResult
from dr_llm.sampling.pool_service import PoolService


def _make_service() -> tuple[PoolService, MagicMock, MagicMock]:
    store = MagicMock()
    service = PoolService(store)
    sampling = MagicMock()
    service._sampling = sampling
    return service, store, sampling


def test_acquire_or_generate_returns_early_when_satisfied() -> None:
    service, store, sampling = _make_service()
    sample = PoolSample(
        key_values={"dim_a": "x", "dim_b": 1},
        request={"prompt": "hi"},
        response={"text": "ok"},
        finish_reason="stop",
        sample_idx=0,
    )
    sampling.acquire.return_value = AcquireResult(samples=[sample])
    query = AcquireQuery(run_id="r1", key_values={"dim_a": "x", "dim_b": 1}, n=1)

    result = service.acquire_or_generate(
        query, consumer_id="c1", generator_fn=lambda kv, n: []
    )

    assert result.claimed == 1
    store.insert_samples.assert_not_called()


def test_acquire_or_generate_calls_generator_on_deficit() -> None:
    service, store, sampling = _make_service()
    empty_result = AcquireResult()
    generated_sample = PoolSample(
        key_values={"dim_a": "x", "dim_b": 1},
        request={"prompt": "hi"},
        response={"text": "ok"},
        finish_reason="stop",
        sample_idx=0,
    )
    reacquired_result = AcquireResult(samples=[generated_sample])
    sampling.acquire.side_effect = [empty_result, reacquired_result]
    store.insert_samples.return_value = InsertResult(inserted=1, skipped=0)
    query = AcquireQuery(run_id="r1", key_values={"dim_a": "x", "dim_b": 1}, n=1)

    def gen(kv: dict, n: int) -> list[PoolSample]:
        return [generated_sample]

    result = service.acquire_or_generate(query, consumer_id="c1", generator_fn=gen)

    assert result.claimed == 1
    store.insert_samples.assert_called_once()


def test_acquire_or_generate_skips_reacquire_when_nothing_inserted() -> None:
    service, store, sampling = _make_service()
    empty_result = AcquireResult()
    generated_sample = PoolSample(
        key_values={"dim_a": "x", "dim_b": 1},
        request={"prompt": "hi"},
        response={"text": "ok"},
        finish_reason="stop",
        sample_idx=0,
    )
    sampling.acquire.return_value = empty_result
    store.insert_samples.return_value = InsertResult(inserted=0, skipped=1, failed=0)
    query = AcquireQuery(run_id="r1", key_values={"dim_a": "x", "dim_b": 1}, n=1)

    result = service.acquire_or_generate(
        query,
        consumer_id="c1",
        generator_fn=lambda kv, n: [generated_sample],
    )

    assert result.claimed == 0
    sampling.acquire.assert_called_once()
    store.insert_samples.assert_called_once()

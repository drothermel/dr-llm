"""Unit tests for RoundRobinClaimer (mock-based, no DB)."""

from __future__ import annotations

from unittest.mock import MagicMock

from dr_llm.pool.claim_strategy import ClaimOrder, RoundRobinClaimer, _merge_filter
from dr_llm.pool.db.key_filter import PoolKeyEqClause, PoolKeyFilter
from dr_llm.pool.pool_sample import PoolSample


def _make_sample(dim_a: str) -> PoolSample:
    return PoolSample(
        key_values={"dim_a": dim_a},
        sample_idx=0,
        request={"prompt": dim_a},
    )


def test_cycles_through_values() -> None:
    store = MagicMock()
    samples = {"x": _make_sample("x"), "y": _make_sample("y"), "z": _make_sample("z")}
    store.claim_lease.side_effect = lambda **kw: samples.get(
        kw["key_filter"].root["dim_a"].value
    )

    claimer = RoundRobinClaimer(
        store,
        round_robin_key="dim_a",
        round_robin_values=["x", "y", "z"],
    )

    first = claimer.claim(worker_id="w1", lease_seconds=60)
    second = claimer.claim(worker_id="w1", lease_seconds=60)
    third = claimer.claim(worker_id="w1", lease_seconds=60)

    assert first is not None and first.key_values["dim_a"] == "x"
    assert second is not None and second.key_values["dim_a"] == "y"
    assert third is not None and third.key_values["dim_a"] == "z"


def test_returns_none_when_all_exhausted() -> None:
    store = MagicMock()
    store.claim_lease.return_value = None

    claimer = RoundRobinClaimer(
        store,
        round_robin_key="dim_a",
        round_robin_values=["x", "y"],
    )
    result = claimer.claim(worker_id="w1", lease_seconds=60)
    assert result is None
    assert store.claim_lease.call_count == 2


def test_skips_exhausted_value_and_tries_next() -> None:
    store = MagicMock()
    sample_y = _make_sample("y")
    store.claim_lease.side_effect = [None, sample_y]

    claimer = RoundRobinClaimer(
        store,
        round_robin_key="dim_a",
        round_robin_values=["x", "y"],
    )
    result = claimer.claim(worker_id="w1", lease_seconds=60)
    assert result is sample_y
    assert store.claim_lease.call_count == 2


def test_random_shuffles_values() -> None:
    store = MagicMock()
    store.claim_lease.return_value = None

    claimer = RoundRobinClaimer(
        store,
        round_robin_key="dim_a",
        round_robin_values=["a", "b", "c", "d", "e"],
        order=ClaimOrder(kind="random", seed=42),
    )
    claimer.claim(worker_id="w1", lease_seconds=60)

    attempted_values = [
        c.kwargs["key_filter"].root["dim_a"].value
        for c in store.claim_lease.call_args_list
    ]
    assert sorted(attempted_values) == ["a", "b", "c", "d", "e"]
    assert attempted_values != ["a", "b", "c", "d", "e"]


def test_merges_base_filter() -> None:
    store = MagicMock()
    store.claim_lease.return_value = None

    base = PoolKeyFilter.eq(dim_b=1)
    claimer = RoundRobinClaimer(
        store,
        round_robin_key="dim_a",
        round_robin_values=["x"],
        base_key_filter=base,
    )
    claimer.claim(worker_id="w1", lease_seconds=60)

    kf = store.claim_lease.call_args.kwargs["key_filter"]
    assert "dim_a" in kf.root
    assert "dim_b" in kf.root
    assert kf.root["dim_b"].value == 1


def test_empty_values_returns_none() -> None:
    store = MagicMock()
    claimer = RoundRobinClaimer(
        store,
        round_robin_key="dim_a",
        round_robin_values=[],
    )
    assert claimer.claim(worker_id="w1", lease_seconds=60) is None
    store.claim_lease.assert_not_called()


def test_merge_filter_without_base() -> None:
    result = _merge_filter(None, "dim_a", "x")
    clause = result.root["dim_a"]
    assert isinstance(clause, PoolKeyEqClause)
    assert clause.value == "x"
    assert len(result.root) == 1


def test_merge_filter_with_base() -> None:
    base = PoolKeyFilter.eq(dim_b=1, dim_c="hello")
    result = _merge_filter(base, "dim_a", "x")
    dim_a = result.root["dim_a"]
    dim_b = result.root["dim_b"]
    dim_c = result.root["dim_c"]
    assert isinstance(dim_a, PoolKeyEqClause) and dim_a.value == "x"
    assert isinstance(dim_b, PoolKeyEqClause) and dim_b.value == 1
    assert isinstance(dim_c, PoolKeyEqClause) and dim_c.value == "hello"

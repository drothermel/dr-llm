from __future__ import annotations

from typing import Any, cast

import pytest

from dr_llm.pool.errors import PoolSchemaError
from dr_llm.pool.pending.fill_pending import seed_pending
from dr_llm.pool.sample_store import PoolStore


class _FakeSchema:
    def __init__(self, key_column_names: list[str]) -> None:
        self.key_column_names = key_column_names


class _FakePendingStore:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._seen: set[tuple[tuple[str, Any], ...]] = set()

    def insert_pending(self, sample: Any, *, ignore_conflicts: bool = True) -> bool:
        key = tuple(sorted(sample.key_values.items())) + (("sample_idx", sample.sample_idx),)
        if ignore_conflicts and key in self._seen:
            return False
        self._seen.add(key)
        self.samples.append(
            {
                "key_values": dict(sample.key_values),
                "sample_idx": sample.sample_idx,
                "priority": sample.priority,
            }
        )
        return True


class _FakeStore:
    def __init__(self, key_column_names: list[str]) -> None:
        self.schema = _FakeSchema(key_column_names)
        self.pending = _FakePendingStore()
        self.init_calls = 0

    def init_schema(self) -> None:
        self.init_calls += 1


def test_seed_pending_expands_cartesian_product() -> None:
    store = _FakeStore(["model", "prompt"])
    typed_store = cast(PoolStore, store)

    result = seed_pending(
        typed_store,
        key_grid={
            "model": ["m1", "m2"],
            "prompt": ["p1", "p2"],
        },
        n=2,
        priority=7,
    )

    assert store.init_calls == 1
    assert result.inserted == 8
    assert result.skipped == 0
    assert [(row["key_values"]["model"], row["key_values"]["prompt"], row["sample_idx"]) for row in store.pending.samples] == [
        ("m1", "p1", 0),
        ("m1", "p1", 1),
        ("m1", "p2", 0),
        ("m1", "p2", 1),
        ("m2", "p1", 0),
        ("m2", "p1", 1),
        ("m2", "p2", 0),
        ("m2", "p2", 1),
    ]
    assert all(row["priority"] == 7 for row in store.pending.samples)


def test_seed_pending_is_incremental_for_higher_n() -> None:
    store = _FakeStore(["model", "prompt"])
    typed_store = cast(PoolStore, store)

    first = seed_pending(
        typed_store,
        key_grid={"model": ["m1"], "prompt": ["p1"]},
        n=1,
    )
    second = seed_pending(
        typed_store,
        key_grid={"model": ["m1"], "prompt": ["p1"]},
        n=3,
    )

    assert first.inserted == 1
    assert second.inserted == 2
    assert second.skipped == 1
    assert [row["sample_idx"] for row in store.pending.samples] == [0, 1, 2]


def test_seed_pending_validates_key_grid_columns() -> None:
    store = _FakeStore(["model", "prompt"])
    typed_store = cast(PoolStore, store)

    with pytest.raises(PoolSchemaError, match="Missing key columns"):
        seed_pending(typed_store, key_grid={"model": ["m1"]}, n=1)

    with pytest.raises(PoolSchemaError, match="Unexpected key columns"):
        seed_pending(
            typed_store,
            key_grid={"model": ["m1"], "prompt": ["p1"], "extra": ["x"]},
            n=1,
        )


def test_seed_pending_rejects_negative_n() -> None:
    store = _FakeStore(["model"])
    typed_store = cast(PoolStore, store)

    with pytest.raises(ValueError, match="non-negative"):
        seed_pending(typed_store, key_grid={"model": ["m1"]}, n=-1)

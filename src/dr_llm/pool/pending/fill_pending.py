"""Helpers for seeding pending pool work."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any

from pydantic import BaseModel

from dr_llm.pool.db.sql_helpers import validate_key_values
from dr_llm.pool.models import InsertResult
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pool_store import PoolStore


type PendingGridValue = Iterable[Any] | dict[str, Any]


def seed_pending(
    store: PoolStore,
    *,
    key_grid: Mapping[str, PendingGridValue],
    n: int,
    priority: int = 0,
) -> InsertResult:
    """Seed the pending queue from a key-dimension cartesian product."""
    if n < 0:
        raise ValueError("n must be non-negative")

    validate_key_values(store.schema, dict(key_grid))
    if n == 0:
        return InsertResult()

    grid_keys, rich_columns = _parse_key_grid(store.schema.key_column_names, key_grid)
    samples = _build_pending_samples(
        column_names=store.schema.key_column_names,
        grid_keys=grid_keys,
        rich_columns=rich_columns,
        n=n,
        priority=priority,
    )
    return store.pending.insert_many(samples, ignore_conflicts=True)


def _parse_key_grid(
    column_names: list[str],
    key_grid: Mapping[str, PendingGridValue],
) -> tuple[list[list[Any]], dict[str, dict[str, Any]]]:
    """Parse key_grid into grid keys plus optional rich payload columns."""
    grid_keys: list[list[Any]] = []
    rich_columns: dict[str, dict[str, Any]] = {}

    for name in column_names:
        raw = key_grid[name]
        if isinstance(raw, dict):
            grid_keys.append(list(raw.keys()))
            rich_columns[name] = dict(raw)
            continue

        if isinstance(raw, (str, bytes)):
            raise TypeError(f"key_grid[{name!r}] must be an iterable of values")

        grid_keys.append(list(raw))

    return grid_keys, rich_columns


def _build_pending_samples(
    *,
    column_names: list[str],
    grid_keys: list[list[Any]],
    rich_columns: dict[str, dict[str, Any]],
    n: int,
    priority: int,
) -> list[PendingSample]:
    samples: list[PendingSample] = []
    for combination in product(*grid_keys):
        key_values = dict(zip(column_names, combination, strict=True))
        payload = _build_payload(key_values=key_values, rich_columns=rich_columns)
        samples.extend(
            PendingSample(
                key_values=key_values,
                sample_idx=sample_idx,
                payload=payload,
                priority=priority,
            )
            for sample_idx in range(n)
        )
    return samples


def _build_payload(
    *,
    key_values: dict[str, Any],
    rich_columns: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, col_value in key_values.items():
        if name in rich_columns:
            payload[name] = serialize_payload_value(rich_columns[name][col_value])
    return payload


def serialize_payload_value(value: Any) -> Any:
    """Convert common pending payload values into plain JSON-compatible shapes."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in value
        ]
    return value

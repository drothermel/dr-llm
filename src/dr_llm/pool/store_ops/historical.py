"""Migration-oriented insert helpers for historical pool data."""

from __future__ import annotations

from typing import Any, Final

from dr_llm.pool.db import DbRuntime, PoolSchema, PoolTables
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.store_ops import insert as insert_ops

UNAVAILABLE_REQUEST_SENTINEL: Final[dict[str, Any]] = {
    "unavailable": True,
    "reason": "historical_migration",
}


def insert_historical_sample(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    sample: PoolSample,
    *,
    allow_missing_request: bool = False,
) -> bool:
    if sample.request == {}:
        if not allow_missing_request:
            raise ValueError(
                f"Sample {sample.sample_id} has an empty request. "
                "Pass allow_missing_request=True to insert with a sentinel value."
            )
        sample = sample.model_copy(update={"request": UNAVAILABLE_REQUEST_SENTINEL})

    return insert_ops.insert_sample(
        runtime, schema, tables, sample, ignore_conflicts=True
    )

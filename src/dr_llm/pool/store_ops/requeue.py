"""Requeue error rows and administratively reset samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import delete, update

from dr_llm.pool.db import (
    DbRuntime,
    LeaseColumn,
    PoolSchema,
    PoolTables,
    PoolTableType,
    SampleColumn,
)
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.db.sql_helpers import partial_key_filter_clause


def requeue_errors(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    *,
    key_filter: PoolKeyFilter | None = None,
    reset_attempt_count: bool = True,
) -> int:
    samples_table = tables[PoolTableType.SAMPLES]
    leases_table = tables[PoolTableType.LEASES]

    values: dict[str, Any] = {
        SampleColumn.RESPONSE_JSON: None,
        SampleColumn.FINISH_REASON: None,
    }
    if reset_attempt_count:
        values[SampleColumn.ATTEMPT_COUNT] = 0

    stmt = (
        update(samples_table)
        .where(
            samples_table.c[SampleColumn.FINISH_REASON] == "error",
            samples_table.c[SampleColumn.RESPONSE_JSON].is_not(None),
        )
        .values(values)
        .returning(samples_table.c[SampleColumn.SAMPLE_ID])
    )
    partial_filter = partial_key_filter_clause(schema, samples_table, key_filter)
    if partial_filter is not None:
        stmt = stmt.where(partial_filter)

    with runtime.begin() as conn:
        result = conn.execute(stmt)
        requeued_ids = list(result.scalars())
        if requeued_ids:
            conn.execute(
                delete(leases_table).where(
                    leases_table.c[LeaseColumn.SAMPLE_ID].in_(requeued_ids)
                )
            )
    return len(requeued_ids)


def reset_samples(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    *,
    sample_ids: Sequence[str],
    reset_request: dict[str, Any] | None = None,
    reset_metadata: dict[str, Any] | None = None,
) -> int:
    if not sample_ids:
        return 0

    samples_table = tables[PoolTableType.SAMPLES]
    leases_table = tables[PoolTableType.LEASES]

    values: dict[str, Any] = {
        SampleColumn.RESPONSE_JSON: None,
        SampleColumn.FINISH_REASON: None,
        SampleColumn.ATTEMPT_COUNT: 0,
    }
    if reset_request is not None:
        values[SampleColumn.REQUEST_JSON] = to_jsonable_python(reset_request)
    if reset_metadata is not None:
        values[SampleColumn.METADATA_JSON] = to_jsonable_python(reset_metadata)

    stmt = (
        update(samples_table)
        .where(samples_table.c[SampleColumn.SAMPLE_ID].in_(list(sample_ids)))
        .values(values)
        .returning(samples_table.c[SampleColumn.SAMPLE_ID])
    )

    with runtime.begin() as conn:
        result = conn.execute(stmt)
        reset_ids = list(result.scalars())
        if reset_ids:
            conn.execute(
                delete(leases_table).where(
                    leases_table.c[LeaseColumn.SAMPLE_ID].in_(reset_ids)
                )
            )
    return len(reset_ids)

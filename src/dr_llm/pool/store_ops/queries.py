"""Read-only pool queries: counts, depth, and sample iteration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.sql.schema import Table

from dr_llm.pool.db import (
    DbRuntime,
    PoolSchema,
    PoolTables,
    PoolTableType,
    SampleColumn,
)
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.db.sql_helpers import (
    key_filter_clause,
    partial_key_filter_clause,
    stream_select_rows,
    validate_key_values,
)
from dr_llm.pool.pool_sample import PoolSample


def sample_count(runtime: DbRuntime, samples_table: Table) -> int:
    stmt = select(func.count()).select_from(samples_table)
    with runtime.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def incomplete_count(
    runtime: DbRuntime,
    schema: PoolSchema,
    samples_table: Table,
    *,
    key_filter: PoolKeyFilter | None = None,
) -> int:
    return _completion_count(
        runtime, schema, samples_table, is_complete=False, key_filter=key_filter
    )


def complete_count(
    runtime: DbRuntime,
    schema: PoolSchema,
    samples_table: Table,
    *,
    key_filter: PoolKeyFilter | None = None,
) -> int:
    return _completion_count(
        runtime, schema, samples_table, is_complete=True, key_filter=key_filter
    )


def _completion_count(
    runtime: DbRuntime,
    schema: PoolSchema,
    samples_table: Table,
    *,
    is_complete: bool,
    key_filter: PoolKeyFilter | None,
) -> int:
    response_predicate = (
        samples_table.c[SampleColumn.RESPONSE_JSON].is_not(None)
        if is_complete
        else samples_table.c[SampleColumn.RESPONSE_JSON].is_(None)
    )
    stmt = select(func.count()).select_from(samples_table).where(response_predicate)
    partial_filter = partial_key_filter_clause(schema, samples_table, key_filter)
    if partial_filter is not None:
        stmt = stmt.where(partial_filter)
    with runtime.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def cell_depth(
    runtime: DbRuntime,
    schema: PoolSchema,
    samples_table: Table,
    *,
    key_values: dict[str, Any],
) -> int:
    validate_key_values(schema, key_values)
    stmt = select(func.count()).where(
        key_filter_clause(schema, samples_table, key_values)
    )
    with runtime.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def bulk_load(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    *,
    key_filter: PoolKeyFilter | None = None,
) -> list[PoolSample]:
    return list(iter_samples(runtime, schema, tables, key_filter=key_filter))


def iter_samples(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    *,
    key_filter: PoolKeyFilter | None = None,
    chunk_size: int = 1000,
) -> Iterator[PoolSample]:
    samples_table = tables[PoolTableType.SAMPLES]
    rows = stream_select_rows(
        runtime,
        schema,
        samples_table,
        tables.select_columns(PoolTableType.SAMPLES),
        order_by=[samples_table.c.sample_idx.asc()],
        key_filter=key_filter,
        chunk_size=chunk_size,
    )
    for row in rows:
        yield tables.sample_from_row(row)

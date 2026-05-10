"""Read-only pool queries: counts, depth, and sample iteration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Table

from dr_llm.pool.db import (
    DbRuntime,
    LeaseColumn,
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
from dr_llm.pool.completion_filter import CompletionFilter
from dr_llm.pool.pool_progress import PoolProgress
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
        runtime,
        schema,
        samples_table,
        is_complete=False,
        key_filter=key_filter,
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
    stmt = (
        select(func.count())
        .select_from(samples_table)
        .where(response_predicate)
    )
    partial_filter = partial_key_filter_clause(
        schema, samples_table, key_filter
    )
    if partial_filter is not None:
        stmt = stmt.where(partial_filter)
    with runtime.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def error_count(
    runtime: DbRuntime,
    schema: PoolSchema,
    samples_table: Table,
    *,
    key_filter: PoolKeyFilter | None = None,
) -> int:
    stmt = (
        select(func.count())
        .select_from(samples_table)
        .where(
            samples_table.c[SampleColumn.RESPONSE_JSON].is_not(None),
            samples_table.c[SampleColumn.FINISH_REASON] == "error",
        )
    )
    partial_filter = partial_key_filter_clause(
        schema, samples_table, key_filter
    )
    if partial_filter is not None:
        stmt = stmt.where(partial_filter)
    with runtime.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def leased_count(
    runtime: DbRuntime,
    schema: PoolSchema,
    samples_table: Table,
    leases_table: Table,
    *,
    key_filter: PoolKeyFilter | None = None,
) -> int:
    stmt = (
        select(func.count())
        .select_from(
            samples_table.join(
                leases_table,
                samples_table.c[SampleColumn.SAMPLE_ID]
                == leases_table.c[LeaseColumn.SAMPLE_ID],
            )
        )
        .where(
            leases_table.c[LeaseColumn.LEASE_EXPIRES_AT] >= func.now(),
        )
    )
    partial_filter = partial_key_filter_clause(
        schema, samples_table, key_filter
    )
    if partial_filter is not None:
        stmt = stmt.where(partial_filter)
    with runtime.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def progress(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    *,
    key_filter: PoolKeyFilter | None = None,
) -> PoolProgress:
    samples_table = tables[PoolTableType.SAMPLES]
    leases_table = tables[PoolTableType.LEASES]

    response_col = samples_table.c[SampleColumn.RESPONSE_JSON]
    finish_col = samples_table.c[SampleColumn.FINISH_REASON]
    stmt = select(
        func.count().label("total"),
        func.count().filter(response_col.is_(None)).label("incomplete"),
        func.count().filter(response_col.is_not(None)).label("complete"),
        func.count()
        .filter(response_col.is_not(None), finish_col == "error")
        .label("error"),
    ).select_from(samples_table)

    partial_filter = partial_key_filter_clause(
        schema, samples_table, key_filter
    )
    if partial_filter is not None:
        stmt = stmt.where(partial_filter)

    with runtime.connect() as conn:
        row = conn.execute(stmt).mappings().one()
        leased = leased_count(
            runtime, schema, samples_table, leases_table, key_filter=key_filter
        )

    return PoolProgress(
        total=int(row["total"]),
        incomplete=int(row["incomplete"]),
        complete=int(row["complete"]),
        error=int(row["error"]),
        leased=leased,
    )


def _completion_predicate(
    samples_table: Table,
    completion: CompletionFilter,
) -> ColumnElement[bool] | None:
    response_col = samples_table.c[SampleColumn.RESPONSE_JSON]
    finish_col = samples_table.c[SampleColumn.FINISH_REASON]
    match completion:
        case "all":
            return None
        case "incomplete":
            return response_col.is_(None)
        case "complete":
            return response_col.is_not(None)
        case "error":
            return and_(response_col.is_not(None), finish_col == "error")


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
    completion: CompletionFilter = "all",
) -> list[PoolSample]:
    return list(
        iter_samples(
            runtime,
            schema,
            tables,
            key_filter=key_filter,
            completion=completion,
        )
    )


def iter_samples(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    *,
    key_filter: PoolKeyFilter | None = None,
    completion: CompletionFilter = "all",
    chunk_size: int = 1000,
) -> Iterator[PoolSample]:
    samples_table = tables[PoolTableType.SAMPLES]
    predicate = _completion_predicate(samples_table, completion)
    base_predicates = (predicate,) if predicate is not None else ()
    rows = stream_select_rows(
        runtime,
        schema,
        samples_table,
        tables.select_columns(PoolTableType.SAMPLES),
        base_predicates=base_predicates,
        order_by=[samples_table.c.sample_idx.asc()],
        key_filter=key_filter,
        chunk_size=chunk_size,
    )
    for row in rows:
        yield tables.sample_from_row(row)

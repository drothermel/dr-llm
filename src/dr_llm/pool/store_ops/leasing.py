"""Lease operations: claim, release, and expire sample leases."""

from __future__ import annotations

from sqlalchemy import delete, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
from dr_llm.pool.db.sql_helpers import partial_key_filter_clause
from dr_llm.pool.pool_sample import PoolSample


def claim_lease(
    runtime: DbRuntime,
    schema: PoolSchema,
    tables: PoolTables,
    *,
    worker_id: str,
    lease_seconds: int,
    key_filter: PoolKeyFilter | None = None,
) -> PoolSample | None:
    if lease_seconds <= 0:
        raise ValueError(
            f"lease_seconds must be a positive integer; got {lease_seconds}"
        )

    samples_table = tables[PoolTableType.SAMPLES]
    leases_table = tables[PoolTableType.LEASES]
    sample_columns = tables.select_columns(PoolTableType.SAMPLES)
    predicates = [
        samples_table.c[SampleColumn.RESPONSE_JSON].is_(None),
        or_(
            leases_table.c[LeaseColumn.SAMPLE_ID].is_(None),
            leases_table.c[LeaseColumn.LEASE_EXPIRES_AT] < func.now(),
        ),
    ]
    partial_filter = partial_key_filter_clause(
        schema, samples_table, key_filter
    )
    if partial_filter is not None:
        predicates.append(partial_filter)

    locked = (
        select(*sample_columns)
        .select_from(
            samples_table.outerjoin(
                leases_table,
                samples_table.c[SampleColumn.SAMPLE_ID]
                == leases_table.c[LeaseColumn.SAMPLE_ID],
            )
        )
        .where(*predicates)
        .order_by(samples_table.c[SampleColumn.CREATED_AT].asc())
        .limit(1)
        .with_for_update(of=samples_table, skip_locked=True)
        .cte("locked_sample")
    )
    lease_expires_at = func.now() + func.make_interval(
        0, 0, 0, 0, 0, 0, lease_seconds
    )
    leased = (
        pg_insert(leases_table)
        .from_select(
            [
                LeaseColumn.SAMPLE_ID,
                LeaseColumn.WORKER_ID,
                LeaseColumn.LEASE_EXPIRES_AT,
            ],
            select(
                locked.c[SampleColumn.SAMPLE_ID],
                literal(worker_id),
                lease_expires_at,
            ),
        )
        .on_conflict_do_update(
            index_elements=[leases_table.c[LeaseColumn.SAMPLE_ID]],
            set_={
                LeaseColumn.WORKER_ID: worker_id,
                LeaseColumn.LEASE_EXPIRES_AT: lease_expires_at,
            },
            where=leases_table.c[LeaseColumn.LEASE_EXPIRES_AT] < func.now(),
        )
        .returning(leases_table.c[LeaseColumn.SAMPLE_ID])
        .cte("leased_sample")
    )
    updated = (
        update(samples_table)
        .where(
            samples_table.c[SampleColumn.SAMPLE_ID]
            == select(leased.c[LeaseColumn.SAMPLE_ID]).scalar_subquery()
        )
        .values(
            {
                SampleColumn.ATTEMPT_COUNT: samples_table.c[
                    SampleColumn.ATTEMPT_COUNT
                ]
                + 1
            }
        )
        .returning(*sample_columns)
        .cte("updated_sample")
    )
    stmt = select(*(updated.c[column.name] for column in sample_columns))

    with runtime.begin() as conn:
        row = conn.execute(stmt).mappings().first()
    if row is None:
        return None
    return tables.sample_from_row(dict(row))


def release_lease(
    runtime: DbRuntime,
    leases_table: Table,
    *,
    sample_id: str,
    worker_id: str,
) -> bool:
    stmt = (
        delete(leases_table)
        .where(
            leases_table.c[LeaseColumn.SAMPLE_ID] == sample_id,
            leases_table.c[LeaseColumn.WORKER_ID] == worker_id,
        )
        .returning(leases_table.c[LeaseColumn.SAMPLE_ID])
    )
    with runtime.begin() as conn:
        return conn.execute(stmt).scalar_one_or_none() is not None


def expire_leases(runtime: DbRuntime, leases_table: Table) -> int:
    stmt = (
        delete(leases_table)
        .where(leases_table.c[LeaseColumn.LEASE_EXPIRES_AT] < func.now())
        .returning(leases_table.c[LeaseColumn.SAMPLE_ID])
    )
    with runtime.begin() as conn:
        return sum(1 for _ in conn.execute(stmt).scalars())

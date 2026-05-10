"""Consumer-scoped claims-based acquisition from a pool."""

from __future__ import annotations

from typing import Any, Self

from sqlalchemy import Text, exists, func, literal, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql.elements import ColumnElement

from dr_llm.pool.db import (
    DbRuntime,
    PoolSchema,
    PoolTableType,
    PoolTables,
    SampleColumn,
)
from dr_llm.pool.db.sql_helpers import key_filter_clause, validate_key_values
from dr_llm.pool.pool_store import PoolStore
from dr_llm.sampling.acquisition import AcquireQuery, AcquireResult
from dr_llm.sampling.db.claims_tables import ClaimsTables
from dr_llm.sampling.db.names import ClaimColumn


class SamplingStore:
    """Claims-based acquisition parameterized by pool schema and consumer.

    Each consumer gets its own claims table. Call :meth:`setup_consumer`
    before acquiring samples, and :meth:`teardown_consumer` when done.
    """

    def __init__(
        self, schema: PoolSchema, runtime: DbRuntime, pool_tables: PoolTables
    ) -> None:
        self._schema = schema
        self._runtime = runtime
        self._pool_tables = pool_tables
        self._consumers: dict[str, ClaimsTables] = {}

    @classmethod
    def from_pool_store(cls, store: PoolStore) -> Self:
        """Create a sampling store from an initialized pool store."""
        return cls(store.schema, store._runtime, store._tables)

    def setup_consumer(self, consumer_id: str) -> None:
        claims = ClaimsTables(self._schema.name, consumer_id)
        claims.ensure_table(self._runtime)
        self._consumers[consumer_id] = claims

    def teardown_consumer(self, consumer_id: str) -> None:
        claims = self._consumers.pop(consumer_id, None)
        if claims is not None:
            claims.drop_table(self._runtime)

    def _get_claims(self, consumer_id: str) -> ClaimsTables:
        claims = self._consumers.get(consumer_id)
        if claims is None:
            raise ValueError(
                f"Consumer {consumer_id!r} not set up. Call setup_consumer() first."
            )
        return claims

    def acquire(self, query: AcquireQuery, consumer_id: str) -> AcquireResult:
        """Acquire up to query.n completed, unclaimed samples for given keys.

        Single round-trip via data-modifying CTE: lock candidate sample rows
        with FOR UPDATE SKIP LOCKED, insert claim rows for them, then return
        the joined sample data.
        """
        validate_key_values(self._schema, query.key_values)
        if query.n == 0:
            return AcquireResult()

        claims = self._get_claims(consumer_id)
        samples_table = self._pool_tables[PoolTableType.SAMPLES]
        claims_table = claims.claims_table

        locked = (
            select(
                samples_table.c.sample_id,
                samples_table.c.sample_idx,
                samples_table.c.created_at,
            )
            .where(
                key_filter_clause(self._schema, samples_table, query.key_values),
                samples_table.c[SampleColumn.RESPONSE_JSON].is_not(None),
                self._unclaimed_predicate(query.run_id, consumer_id),
            )
            .order_by(
                samples_table.c.sample_idx.asc(),
                samples_table.c.created_at.asc(),
            )
            .limit(query.n)
            .with_for_update(skip_locked=True)
            .cte("locked")
        )

        claim_source = select(
            func.cast(func.gen_random_uuid(), Text).label(ClaimColumn.CLAIM_ID),
            literal(query.run_id, type_=Text).label(ClaimColumn.RUN_ID),
            literal(query.request_id, type_=Text).label(ClaimColumn.REQUEST_ID),
            literal(query.consumer_tag, type_=Text).label(ClaimColumn.CONSUMER_TAG),
            locked.c.sample_id.label(ClaimColumn.SAMPLE_ID),
            (
                func.row_number().over(
                    order_by=[
                        locked.c.sample_idx.asc(),
                        locked.c.created_at.asc(),
                    ]
                )
                - 1
            ).label(ClaimColumn.CLAIM_IDX),
        ).select_from(locked)

        inserted = (
            pg_insert(claims_table)
            .from_select(
                [
                    ClaimColumn.CLAIM_ID,
                    ClaimColumn.RUN_ID,
                    ClaimColumn.REQUEST_ID,
                    ClaimColumn.CONSUMER_TAG,
                    ClaimColumn.SAMPLE_ID,
                    ClaimColumn.CLAIM_IDX,
                ],
                claim_source,
            )
            .on_conflict_do_nothing(
                index_elements=[claims_table.c.run_id, claims_table.c.sample_id]
            )
            .returning(claims_table.c.sample_id)
            .cte("inserted")
        )

        stmt = (
            select(*self._pool_tables.select_columns(PoolTableType.SAMPLES))
            .join(inserted, samples_table.c.sample_id == inserted.c.sample_id)
            .order_by(
                samples_table.c.sample_idx.asc(),
                samples_table.c.created_at.asc(),
            )
        )

        with self._runtime.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        samples = [self._pool_tables.sample_from_row(dict(row)) for row in rows]
        return AcquireResult(samples=samples)

    def remaining(
        self, *, run_id: str, key_values: dict[str, Any], consumer_id: str
    ) -> int:
        """Count completed, unclaimed samples for given key dimensions and run."""
        validate_key_values(self._schema, key_values)
        samples_table = self._pool_tables[PoolTableType.SAMPLES]
        stmt = select(func.count()).where(
            key_filter_clause(self._schema, samples_table, key_values),
            samples_table.c[SampleColumn.RESPONSE_JSON].is_not(None),
            self._unclaimed_predicate(run_id, consumer_id),
        )
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def _unclaimed_predicate(
        self, run_id: str, consumer_id: str
    ) -> ColumnElement[bool]:
        claims = self._get_claims(consumer_id)
        claims_table = claims.claims_table
        samples_table = self._pool_tables[PoolTableType.SAMPLES]
        return ~exists(
            select(1).where(
                claims_table.c.run_id == run_id,
                claims_table.c.sample_id == samples_table.c.sample_id,
            )
        )

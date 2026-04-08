"""Pool sample storage: CRUD, no-replacement acquisition, coverage."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from sqlalchemy import exists, func, literal, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dr_llm.pool.db.metadata import MetadataStore
from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import (
    is_constraint_error,
    key_filter_clause,
    partial_key_filter_clause,
    validate_key_values,
)
from dr_llm.pool.db.tables import PoolTables
from dr_llm.pool.models import AcquireQuery, AcquireResult, CoverageRow, InsertResult
from dr_llm.pool.pending.store import PendingStore
from dr_llm.pool.pool_sample import PoolSample

logger = logging.getLogger(__name__)

_SAMPLE_IDX_RETRY_ATTEMPTS = 5


class PoolStore:
    """Pool storage operations parameterized by schema."""

    def __init__(self, schema: PoolSchema, runtime: DbRuntime) -> None:
        self._schema = schema
        self._runtime = runtime
        self._tables = PoolTables(schema)
        self._schema_lock = threading.Lock()
        self._schema_initialized = False
        self._pending = PendingStore(schema, runtime, self._tables)
        self._metadata = MetadataStore(schema, runtime, self._tables)

    @property
    def schema(self) -> PoolSchema:
        return self._schema

    @property
    def pending(self) -> PendingStore:
        return self._pending

    @property
    def metadata(self) -> MetadataStore:
        return self._metadata

    def init_schema(
        self,
    ) -> None:
        """Create dynamic pool tables if they don't exist.

        These tables remain runtime-owned because their physical names derive from
        PoolSchema. Alembic intentionally excludes them until the pool schema
        design moves away from per-pool table sets.
        """
        if self._schema_initialized:
            return
        with self._schema_lock:
            if self._schema_initialized:
                return
            with self._runtime.begin() as conn:
                self._tables.metadata.create_all(
                    bind=conn,
                    tables=self._tables.all_tables,
                    checkfirst=True,
                )
            self._schema_initialized = True

    # --- Sample CRUD ---

    def insert_sample(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert a single sample. Auto-assigns sample_idx if None."""
        self.init_schema()
        validate_key_values(self._schema, sample.key_values)

        if sample.sample_idx is not None:
            return self._insert_sample_row(sample, ignore_conflicts=ignore_conflicts)

        for attempt in range(_SAMPLE_IDX_RETRY_ATTEMPTS):
            try:
                return self._insert_sample_auto_idx(
                    sample, ignore_conflicts=ignore_conflicts
                )
            except Exception as exc:
                if not (is_constraint_error(exc) and ignore_conflicts):
                    raise
                if attempt < _SAMPLE_IDX_RETRY_ATTEMPTS - 1:
                    logger.debug(
                        "sample_idx allocation conflict for %s (attempt %d/%d)",
                        self._schema.samples_table,
                        attempt + 1,
                        _SAMPLE_IDX_RETRY_ATTEMPTS,
                    )
                    continue
                logger.warning(
                    "Failed to allocate sample_idx after %d attempts for %s",
                    _SAMPLE_IDX_RETRY_ATTEMPTS,
                    self._schema.samples_table,
                )
                return False
        return False

    def _insert_sample_row(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert with explicit sample_idx."""
        insert_row = sample.to_db_insert_row(self._schema)
        stmt = (
            pg_insert(self._tables.samples)
            .values(**insert_row)
            .returning(self._tables.samples.c.sample_id)
        )
        if ignore_conflicts:
            stmt = stmt.on_conflict_do_nothing()

        with self._runtime.begin() as conn:
            try:
                inserted_sample_id = conn.execute(stmt).scalar_one_or_none()
                return inserted_sample_id is not None
            except Exception as exc:
                if ignore_conflicts and is_constraint_error(exc):
                    return False
                raise

    def _insert_sample_auto_idx(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert with auto-incrementing sample_idx via CTE."""
        insert_row = sample.to_db_insert_row(self._schema)
        next_idx = (
            select(
                (
                    func.coalesce(func.max(self._tables.samples.c.sample_idx), -1) + 1
                ).label("idx")
            )
            .where(
                key_filter_clause(self._schema, self._tables.samples, sample.key_values)
            )
            .cte("next_idx")
        )
        column_names = list(insert_row.keys())
        select_expressions = []
        for column_name, value in insert_row.items():
            if column_name == "sample_idx":
                select_expressions.append(next_idx.c.idx)
            else:
                select_expressions.append(
                    literal(value, type_=self._tables.samples.c[column_name].type)
                )
        stmt = pg_insert(self._tables.samples).from_select(
            column_names,
            select(*select_expressions).select_from(next_idx),
        )
        if ignore_conflicts:
            stmt = stmt.on_conflict_do_nothing()
        stmt = stmt.returning(self._tables.samples.c.sample_id)

        with self._runtime.begin() as conn:
            try:
                inserted_sample_id = conn.execute(stmt).scalar_one_or_none()
                return inserted_sample_id is not None
            except Exception as exc:
                if ignore_conflicts and is_constraint_error(exc):
                    return False
                raise

    def insert_samples(
        self, samples: Iterable[PoolSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        """Bulk insert samples."""
        self.init_schema()
        explicit: list[PoolSample] = []
        auto_idx: list[PoolSample] = []
        for sample in samples:
            validate_key_values(self._schema, sample.key_values)
            if sample.sample_idx is None:
                auto_idx.append(sample)
            else:
                explicit.append(sample)

        inserted = 0
        skipped = 0
        failed = 0

        if explicit:
            result = self._batch_insert_explicit(
                explicit, ignore_conflicts=ignore_conflicts
            )
            inserted += result.inserted
            skipped += result.skipped
            failed += result.failed

        for sample in auto_idx:
            try:
                if self.insert_sample(sample, ignore_conflicts=ignore_conflicts):
                    inserted += 1
                else:
                    skipped += 1
            except Exception:
                if not ignore_conflicts:
                    raise
                failed += 1

        return InsertResult(inserted=inserted, skipped=skipped, failed=failed)

    def _batch_insert_explicit(
        self, samples: list[PoolSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        """Batch insert samples that have explicit sample_idx values."""
        insert_rows = [sample.to_db_insert_row(self._schema) for sample in samples]
        stmt = pg_insert(self._tables.samples).returning(
            self._tables.samples.c.sample_id
        )
        if ignore_conflicts:
            stmt = stmt.on_conflict_do_nothing()

        with self._runtime.begin() as conn:
            try:
                row_count = len(list(conn.execute(stmt, insert_rows).scalars()))
                return InsertResult(
                    inserted=row_count, skipped=len(samples) - row_count
                )
            except Exception as exc:
                if ignore_conflicts and is_constraint_error(exc):
                    return InsertResult(skipped=len(samples))
                raise

    # --- No-Replacement Acquisition ---

    def acquire(self, query: AcquireQuery) -> AcquireResult:
        """Acquire up to query.n unclaimed samples for given key dimensions."""
        self.init_schema()
        validate_key_values(self._schema, query.key_values)
        claim_exists = exists(
            select(1).where(
                self._tables.claims.c.run_id == query.run_id,
                self._tables.claims.c.sample_id == self._tables.samples.c.sample_id,
            )
        )
        stmt = (
            select(*self._tables.sample_select_columns())
            .where(
                key_filter_clause(self._schema, self._tables.samples, query.key_values),
                self._tables.samples.c.status == "active",
                ~claim_exists,
            )
            .order_by(
                self._tables.samples.c.sample_idx.asc(),
                self._tables.samples.c.created_at.asc(),
            )
            .limit(query.n)
        )

        with self._runtime.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
            samples: list[PoolSample] = []
            claimed = 0

            for idx, row in enumerate(rows):
                sample = PoolSample.from_db_row(self._schema, row)
                claim_stmt = pg_insert(self._tables.claims).values(
                    claim_id=uuid4().hex,
                    run_id=query.run_id,
                    request_id=query.request_id,
                    consumer_tag=query.consumer_tag,
                    sample_id=row["sample_id"],
                    claim_idx=idx,
                )
                claim_stmt = claim_stmt.on_conflict_do_nothing(
                    index_elements=[
                        self._tables.claims.c.run_id,
                        self._tables.claims.c.sample_id,
                    ]
                )
                claim_stmt = claim_stmt.returning(self._tables.claims.c.claim_id)
                try:
                    claim_id = conn.execute(claim_stmt).scalar_one_or_none()
                    if claim_id is None:
                        logger.debug(
                            "Claim conflict for sample %s in run %s",
                            row["sample_id"],
                            query.run_id,
                        )
                        continue
                    samples.append(sample)
                    claimed += 1
                except Exception as exc:
                    if is_constraint_error(exc):
                        logger.debug(
                            "Claim conflict for sample %s in run %s",
                            row["sample_id"],
                            query.run_id,
                        )
                        continue
                    raise

            return AcquireResult(samples=samples, claimed=claimed)

    def acquire_batch(self, queries: list[AcquireQuery]) -> dict[str, AcquireResult]:
        """Convenience wrapper: acquire() for each query, returning results by request_id."""
        return {
            query_item.request_id: self.acquire(query_item) for query_item in queries
        }

    def remaining(self, *, run_id: str, key_values: dict[str, Any]) -> int:
        """Count unclaimed samples for given key dimensions and run."""
        self.init_schema()
        validate_key_values(self._schema, key_values)
        claim_exists = exists(
            select(1).where(
                self._tables.claims.c.run_id == run_id,
                self._tables.claims.c.sample_id == self._tables.samples.c.sample_id,
            )
        )
        stmt = select(func.count()).where(
            key_filter_clause(self._schema, self._tables.samples, key_values),
            self._tables.samples.c.status == "active",
            ~claim_exists,
        )
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    # --- Coverage / Analytics ---

    def coverage(self) -> list[CoverageRow]:
        """Return sample counts grouped by all key dimensions."""
        self.init_schema()
        stmt = select(
            *self._tables.samples_key_columns,
            func.count().label("cnt"),
        ).group_by(*self._tables.samples_key_columns)
        with self._runtime.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            CoverageRow(
                key_values={name: row[name] for name in self._schema.key_column_names},
                count=int(row["cnt"]),
            )
            for row in rows
        ]

    def cell_depth(self, *, key_values: dict[str, Any]) -> int:
        """Count total samples for a specific cell."""
        self.init_schema()
        validate_key_values(self._schema, key_values)
        stmt = select(func.count()).where(
            key_filter_clause(self._schema, self._tables.samples, key_values)
        )
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    # --- Bulk Load ---

    def bulk_load(
        self, *, key_filter: dict[str, Any] | None = None
    ) -> list[PoolSample]:
        """Load all samples, optionally filtered by partial key match."""
        self.init_schema()
        stmt = select(*self._tables.sample_select_columns()).order_by(
            self._tables.samples.c.sample_idx.asc()
        )
        partial_filter = partial_key_filter_clause(
            self._schema, self._tables.samples, key_filter
        )
        if partial_filter is not None:
            stmt = stmt.where(partial_filter)

        with self._runtime.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [PoolSample.from_db_row(self._schema, dict(row)) for row in rows]

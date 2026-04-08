"""Pool sample storage: CRUD, no-replacement acquisition, coverage."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from sqlalchemy import Column, Integer, Text, exists, func, literal, select, values
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql.elements import ColumnElement

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import (
    execute_insert_count,
    insert_keyed_samples,
    key_filter_clause,
    stream_select_rows,
    validate_key_values,
)
from dr_llm.pool.db.tables import PoolTables
from dr_llm.pool.metadata_store import MetadataStore
from dr_llm.pool.models import AcquireQuery, AcquireResult, CoverageRow, InsertResult
from dr_llm.pool.pending.store import PendingStore
from dr_llm.pool.pool_sample import PoolSample, SampleStatus


class PoolStore:
    """Pool storage operations parameterized by schema.

    Construction is side-effect free. Call :meth:`ensure_schema` once at
    application startup to create the dynamic pool tables and indexes.
    """

    def __init__(self, schema: PoolSchema, runtime: DbRuntime) -> None:
        self.schema = schema
        self._runtime = runtime
        self._tables = PoolTables(schema)
        self.pending = PendingStore(schema, runtime, self._tables)
        self.metadata = MetadataStore(schema, runtime, self._tables)

    def ensure_schema(self) -> None:
        """Create dynamic pool tables and indexes if they don't exist.

        These tables remain runtime-owned because their physical names derive
        from PoolSchema. Alembic intentionally excludes them until the pool
        schema design moves away from per-pool table sets. Safe to call
        multiple times, but each call issues several pg_catalog round-trips,
        so prefer calling exactly once at startup.
        """
        with self._runtime.begin() as conn:
            self._tables.sa_metadata.create_all(
                bind=conn,
                tables=self._tables.all_tables,
                checkfirst=True,
            )
            self._tables.ensure_indexes(conn)

    def insert_sample(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert a single sample. Auto-assigns sample_idx if None."""
        result = self.insert_samples([sample], ignore_conflicts=ignore_conflicts)
        return result.inserted == 1

    def insert_samples(
        self, samples: Iterable[PoolSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        """Bulk insert samples."""
        explicit: list[PoolSample] = []
        auto_idx: list[PoolSample] = []
        for sample in samples:
            validate_key_values(self.schema, sample.key_values)
            if sample.sample_idx is None:
                auto_idx.append(sample)
            else:
                explicit.append(sample)

        result = InsertResult()
        if explicit:
            result += self._insert_explicit(explicit, ignore_conflicts=ignore_conflicts)
        if auto_idx:
            result += self._batch_insert_auto_idx(
                auto_idx, ignore_conflicts=ignore_conflicts
            )
        return result

    def _insert_explicit(
        self, samples: list[PoolSample], *, ignore_conflicts: bool
    ) -> InsertResult:
        """Insert samples that already carry an explicit ``sample_idx``."""
        inserted = insert_keyed_samples(
            self._runtime,
            self._tables.samples,
            self._tables.samples.c.sample_id,
            [sample.to_db_insert_row() for sample in samples],
            ignore_conflicts=ignore_conflicts,
        )
        return InsertResult(inserted=inserted, skipped=len(samples) - inserted)

    def _batch_insert_auto_idx(
        self, samples: list[PoolSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        """Insert auto-idx samples in one statement.

        Pre-computes per-cell row offsets in Python, then issues one
        ``INSERT...SELECT FROM (VALUES ...)`` where each row's
        ``sample_idx`` is derived as ``max(existing for cell) + row_offset``
        via a correlated subquery against the samples table. ON CONFLICT
        DO NOTHING tolerates the rare race against concurrent inserters
        for the same cell.
        """
        if not samples:
            return InsertResult()

        samples_table = self._tables.samples
        key_names = self.schema.key_column_names

        # Assign each row a 1-based row_idx within its key cell. The offset
        # is later added to max(sample_idx) per cell to derive unique
        # sample_idx values without an extra round-trip.
        cell_offsets: dict[tuple[Any, ...], int] = {}
        records: list[dict[str, Any]] = []
        for sample in samples:
            row = sample.to_db_insert_row()
            row.pop("sample_idx", None)
            cell_key = tuple(row[name] for name in key_names)
            cell_offsets[cell_key] = cell_offsets.get(cell_key, 0) + 1
            row["row_idx"] = cell_offsets[cell_key]
            records.append(row)

        # Wrap pre-offset records in a SQL VALUES table. Column types are
        # pulled from samples_table.c[name].type so VALUES rows bind with the
        # correct postgres types; row_idx is hard-coded Integer since it's a
        # synthetic per-cell offset (not a real samples column).
        record_keys = list(records[0].keys())
        value_columns: list[Column[Any]] = [
            Column(
                name,
                Integer() if name == "row_idx" else samples_table.c[name].type,
            )
            for name in record_keys
        ]
        input_data = values(*value_columns, name="input_data").data(
            [tuple(record[name] for name in record_keys) for record in records]
        )

        # Build INSERT ... SELECT FROM input_data. Each row's sample_idx
        # becomes max(existing for cell) + row_idx via a correlated scalar
        # subquery against the samples table.
        max_subquery = (
            select(func.coalesce(func.max(samples_table.c.sample_idx), -1))
            .where(*[samples_table.c[name] == input_data.c[name] for name in key_names])
            .scalar_subquery()
        )
        non_idx_keys = [name for name in record_keys if name != "row_idx"]
        target_columns = [*non_idx_keys, "sample_idx"]
        select_exprs: list[Any] = [
            input_data.c[name].label(name) for name in non_idx_keys
        ]
        select_exprs.append((max_subquery + input_data.c.row_idx).label("sample_idx"))
        stmt = (
            pg_insert(samples_table)
            .from_select(
                target_columns,
                select(*select_exprs).select_from(input_data),
            )
            .returning(samples_table.c.sample_id)
        )

        inserted = execute_insert_count(
            self._runtime, stmt, ignore_conflicts=ignore_conflicts
        )
        return InsertResult(inserted=inserted, skipped=len(samples) - inserted)

    def acquire(self, query: AcquireQuery) -> AcquireResult:
        """Acquire up to query.n unclaimed samples for given key dimensions.

        Single round-trip via data-modifying CTE: lock candidate sample rows
        with FOR UPDATE SKIP LOCKED, insert claim rows for them, then return
        the joined sample data. ON CONFLICT DO NOTHING tolerates the rare
        race where a row was claimed between the lock and the insert.
        """
        validate_key_values(self.schema, query.key_values)
        if query.n == 0:
            return AcquireResult()

        samples_table = self._tables.samples
        claims_table = self._tables.claims

        locked = (
            select(
                samples_table.c.sample_id,
                samples_table.c.sample_idx,
                samples_table.c.created_at,
            )
            .where(
                key_filter_clause(self.schema, samples_table, query.key_values),
                samples_table.c.status == SampleStatus.active,
                self._unclaimed_predicate(query.run_id),
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
            func.cast(func.gen_random_uuid(), Text).label("claim_id"),
            literal(query.run_id, type_=Text).label("run_id"),
            literal(query.request_id, type_=Text).label("request_id"),
            literal(query.consumer_tag, type_=Text).label("consumer_tag"),
            locked.c.sample_id.label("sample_id"),
            (
                func.row_number().over(
                    order_by=[
                        locked.c.sample_idx.asc(),
                        locked.c.created_at.asc(),
                    ]
                )
                - 1
            ).label("claim_idx"),
        ).select_from(locked)

        inserted = (
            pg_insert(claims_table)
            .from_select(
                [
                    "claim_id",
                    "run_id",
                    "request_id",
                    "consumer_tag",
                    "sample_id",
                    "claim_idx",
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
            select(*self._tables.sample_select_columns())
            .join(inserted, samples_table.c.sample_id == inserted.c.sample_id)
            .order_by(
                samples_table.c.sample_idx.asc(),
                samples_table.c.created_at.asc(),
            )
        )

        with self._runtime.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        samples = [PoolSample.from_db_row(self.schema, dict(row)) for row in rows]
        return AcquireResult(samples=samples)

    def remaining(self, *, run_id: str, key_values: dict[str, Any]) -> int:
        """Count unclaimed samples for given key dimensions and run."""
        validate_key_values(self.schema, key_values)
        stmt = select(func.count()).where(
            key_filter_clause(self.schema, self._tables.samples, key_values),
            self._tables.samples.c.status == SampleStatus.active,
            self._unclaimed_predicate(run_id),
        )
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def _unclaimed_predicate(self, run_id: str) -> ColumnElement[bool]:
        """Predicate matching samples with no claim row for the given run."""
        return ~exists(
            select(1).where(
                self._tables.claims.c.run_id == run_id,
                self._tables.claims.c.sample_id == self._tables.samples.c.sample_id,
            )
        )

    def coverage(self) -> list[CoverageRow]:
        """Return sample counts grouped by all key dimensions."""
        stmt = select(
            *self._tables.samples_key_columns,
            func.count().label("cnt"),
        ).group_by(*self._tables.samples_key_columns)
        with self._runtime.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            CoverageRow(
                key_values={name: row[name] for name in self.schema.key_column_names},
                count=int(row["cnt"]),
            )
            for row in rows
        ]

    def cell_depth(self, *, key_values: dict[str, Any]) -> int:
        """Count total samples for a specific cell."""
        validate_key_values(self.schema, key_values)
        stmt = select(func.count()).where(
            key_filter_clause(self.schema, self._tables.samples, key_values)
        )
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def bulk_load(
        self, *, key_filter: dict[str, Any] | None = None
    ) -> list[PoolSample]:
        """Load all samples, optionally filtered by partial key match.

        Materializes the full result set in memory; for pools with 100k+ rows
        prefer ``iter_samples`` to stream in chunks.
        """
        return list(self.iter_samples(key_filter=key_filter))

    def iter_samples(
        self,
        *,
        key_filter: dict[str, Any] | None = None,
        chunk_size: int = 1000,
    ) -> Iterator[PoolSample]:
        """Stream samples in chunks via server-side cursoring.

        Uses SQLAlchemy's ``yield_per`` so the driver fetches ``chunk_size``
        rows at a time instead of materializing the entire result set. Safe
        for pools far larger than memory. The underlying connection is held
        open for the lifetime of the iterator — fully consume or close it
        promptly.
        """
        rows = stream_select_rows(
            self._runtime,
            self.schema,
            self._tables.samples,
            self._tables.sample_select_columns(),
            order_by=[self._tables.samples.c.sample_idx.asc()],
            key_filter=key_filter,
            chunk_size=chunk_size,
        )
        for row in rows:
            yield PoolSample.from_db_row(self.schema, row)

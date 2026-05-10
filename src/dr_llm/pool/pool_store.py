"""Pool sample storage and leasing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import delete, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from dr_llm.pool.db import (
    DbRuntime,
    LeaseColumn,
    PoolSchema,
    PoolTables,
    PoolTableType,
    SampleColumn,
)
from dr_llm.pool.db.sql_helpers import (
    insert_keyed_samples,
    is_constraint_error,
    key_filter_clause,
    partial_key_filter_clause,
    stream_select_rows,
    validate_key_values,
)
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.results import InsertResult


class PoolStore:
    """Pool storage operations parameterized by schema.

    Construction is side-effect free. Call :meth:`ensure_schema` once at
    application startup to create the dynamic pool tables and indexes.
    """

    def __init__(self, schema: PoolSchema, runtime: DbRuntime) -> None:
        self.schema = schema
        self._runtime = runtime
        self._tables = PoolTables(schema)

    def close(self) -> None:
        """Dispose the underlying runtime owned by this store."""
        self._runtime.close()

    def ensure_schema(self) -> None:
        """Create dynamic pool tables and indexes if they don't exist.

        These tables remain runtime-owned because their physical names derive
        from PoolSchema. Alembic intentionally excludes them until the pool
        schema design moves away from per-pool table sets. Safe to call
        multiple times, but each call issues several pg_catalog round-trips
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
        samples_table = self._tables[PoolTableType.SAMPLES]
        inserted = insert_keyed_samples(
            self._runtime,
            samples_table,
            samples_table.c.sample_id,
            [sample.to_db_insert_row() for sample in samples],
            ignore_conflicts=ignore_conflicts,
        )
        return InsertResult(inserted=inserted, skipped=len(samples) - inserted)

    def _batch_insert_auto_idx(
        self, samples: list[PoolSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        """Insert auto-idx samples with transaction-scoped per-cell allocation."""
        if not samples:
            return InsertResult()

        samples_table = self._tables[PoolTableType.SAMPLES]
        key_names = self.schema.key_column_names
        base_rows: list[dict[str, Any]] = []
        for sample in samples:
            row = sample.to_db_insert_row()
            row.pop(SampleColumn.SAMPLE_IDX, None)
            base_rows.append(row)

        for attempt in range(1, self._AUTO_IDX_INSERT_RETRIES + 1):
            try:
                with self._runtime.begin() as conn:
                    rows = self._allocate_auto_idx_rows(
                        conn, base_rows=base_rows, key_names=key_names
                    )
                    stmt = pg_insert(samples_table)
                    if ignore_conflicts:
                        stmt = stmt.on_conflict_do_nothing()
                    stmt = stmt.returning(samples_table.c.sample_id)
                    result = (
                        conn.execute(stmt.values(rows[0]))
                        if len(rows) == 1
                        else conn.execute(stmt, rows)
                    )
                    inserted = 0
                    for _ in result.scalars():
                        inserted += 1
                return InsertResult(inserted=inserted, skipped=len(samples) - inserted)
            except Exception as exc:
                if is_constraint_error(exc):
                    if attempt < self._AUTO_IDX_INSERT_RETRIES:
                        continue
                    if ignore_conflicts:
                        return InsertResult(inserted=0, skipped=len(samples))
                raise
        raise AssertionError("auto-idx insert retry loop exhausted unexpectedly")

    def _allocate_auto_idx_rows(
        self,
        conn: Connection,
        *,
        base_rows: list[dict[str, Any]],
        key_names: list[str],
    ) -> list[dict[str, Any]]:
        samples_table = self._tables[PoolTableType.SAMPLES]
        cell_keys = sorted(
            {tuple(row[name] for name in key_names) for row in base_rows},
            key=repr,
        )
        for cell_key in cell_keys:
            conn.execute(
                select(func.pg_advisory_xact_lock(self._cell_lock_id(cell_key)))
            )

        max_sample_idx_by_cell: dict[tuple[Any, ...], int] = {}
        for cell_key in cell_keys:
            key_values = dict(zip(key_names, cell_key, strict=True))
            max_sample_idx_by_cell[cell_key] = int(
                conn.execute(
                    select(
                        func.coalesce(func.max(samples_table.c.sample_idx), -1)
                    ).where(key_filter_clause(self.schema, samples_table, key_values))
                ).scalar_one()
            )

        cell_offsets: dict[tuple[Any, ...], int] = {}
        rows: list[dict[str, Any]] = []
        for base_row in base_rows:
            row = dict(base_row)
            cell_key = tuple(row[name] for name in key_names)
            cell_offsets[cell_key] = cell_offsets.get(cell_key, 0) + 1
            row[SampleColumn.SAMPLE_IDX] = (
                max_sample_idx_by_cell[cell_key] + cell_offsets[cell_key]
            )
            rows.append(row)
        return rows

    def _cell_lock_id(self, cell_key: tuple[Any, ...]) -> int:
        lock_payload = json.dumps(
            {
                "pool": self.schema.table_name(PoolTableType.SAMPLES),
                "key_values": {
                    name: value
                    for name, value in zip(
                        self.schema.key_column_names, cell_key, strict=True
                    )
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.blake2b(lock_payload.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=True)

    def complete_sample(
        self,
        *,
        sample_id: str,
        response: dict[str, Any],
        finish_reason: str | None,
        attempt_count: int,
    ) -> bool:
        """Fill in the response fields for one incomplete sample."""
        samples_table = self._tables[PoolTableType.SAMPLES]
        stmt = (
            update(samples_table)
            .where(
                samples_table.c[SampleColumn.SAMPLE_ID] == sample_id,
                samples_table.c[SampleColumn.RESPONSE_JSON].is_(None),
            )
            .values(
                {
                    SampleColumn.RESPONSE_JSON: to_jsonable_python(response),
                    SampleColumn.FINISH_REASON: finish_reason,
                    SampleColumn.ATTEMPT_COUNT: attempt_count,
                }
            )
            .returning(samples_table.c[SampleColumn.SAMPLE_ID])
        )
        with self._runtime.begin() as conn:
            return conn.execute(stmt).scalar_one_or_none() is not None

    def claim_lease(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        key_filter: PoolKeyFilter | None = None,
    ) -> PoolSample | None:
        """Lease one incomplete sample via ``FOR UPDATE SKIP LOCKED``."""
        if lease_seconds <= 0:
            raise ValueError(
                f"lease_seconds must be a positive integer; got {lease_seconds}"
            )

        samples_table = self._tables[PoolTableType.SAMPLES]
        leases_table = self._tables[PoolTableType.LEASES]
        sample_columns = self._tables.select_columns(PoolTableType.SAMPLES)
        predicates = [
            samples_table.c[SampleColumn.RESPONSE_JSON].is_(None),
            or_(
                leases_table.c[LeaseColumn.SAMPLE_ID].is_(None),
                leases_table.c[LeaseColumn.LEASE_EXPIRES_AT] < func.now(),
            ),
        ]
        partial_filter = partial_key_filter_clause(
            self.schema, samples_table, key_filter
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
        stmt = select(
            *(locked.c[column.name] for column in sample_columns)
        ).select_from(
            locked.join(
                leased,
                locked.c[SampleColumn.SAMPLE_ID] == leased.c[LeaseColumn.SAMPLE_ID],
            )
        )

        with self._runtime.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return PoolSample.from_db_row(self.schema, dict(row))

    def release_lease(self, *, sample_id: str, worker_id: str) -> bool:
        """Release a lease owned by ``worker_id``."""
        leases_table = self._tables[PoolTableType.LEASES]
        stmt = (
            delete(leases_table)
            .where(
                leases_table.c[LeaseColumn.SAMPLE_ID] == sample_id,
                leases_table.c[LeaseColumn.WORKER_ID] == worker_id,
            )
            .returning(leases_table.c[LeaseColumn.SAMPLE_ID])
        )
        with self._runtime.begin() as conn:
            return conn.execute(stmt).scalar_one_or_none() is not None

    def expire_leases(self) -> int:
        """Delete expired lease rows and return the number removed."""
        leases_table = self._tables[PoolTableType.LEASES]
        stmt = (
            delete(leases_table)
            .where(leases_table.c[LeaseColumn.LEASE_EXPIRES_AT] < func.now())
            .returning(leases_table.c[LeaseColumn.SAMPLE_ID])
        )
        with self._runtime.begin() as conn:
            return sum(1 for _ in conn.execute(stmt).scalars())

    def sample_count(self) -> int:
        """Return the total number of rows in the pool's samples table."""
        samples_table = self._tables[PoolTableType.SAMPLES]
        stmt = select(func.count()).select_from(samples_table)
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def incomplete_count(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        """Return the count of samples without responses."""
        return self._completion_count(is_complete=False, key_filter=key_filter)

    def complete_count(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        """Return the count of samples with responses."""
        return self._completion_count(is_complete=True, key_filter=key_filter)

    def _completion_count(
        self, *, is_complete: bool, key_filter: PoolKeyFilter | None
    ) -> int:
        samples_table = self._tables[PoolTableType.SAMPLES]
        response_predicate = (
            samples_table.c[SampleColumn.RESPONSE_JSON].is_not(None)
            if is_complete
            else samples_table.c[SampleColumn.RESPONSE_JSON].is_(None)
        )
        stmt = select(func.count()).select_from(samples_table).where(response_predicate)
        partial_filter = partial_key_filter_clause(
            self.schema, samples_table, key_filter
        )
        if partial_filter is not None:
            stmt = stmt.where(partial_filter)
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def cell_depth(self, *, key_values: dict[str, Any]) -> int:
        """Count total samples for a specific cell."""
        validate_key_values(self.schema, key_values)
        samples_table = self._tables[PoolTableType.SAMPLES]
        stmt = select(func.count()).where(
            key_filter_clause(self.schema, samples_table, key_values)
        )
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def bulk_load(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
    ) -> list[PoolSample]:
        """Load all samples, optionally filtered by partial key match.

        Materializes the full result set in memory; for pools with 100k+ rows
        prefer ``iter_samples`` to stream in chunks.
        """
        return list(self.iter_samples(key_filter=key_filter))

    def iter_samples(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        chunk_size: int = 1000,
    ) -> Iterator[PoolSample]:
        """Stream samples in chunks via server-side cursoring.

        Uses SQLAlchemy's ``yield_per`` so the driver fetches ``chunk_size``
        rows at a time instead of materializing the entire result set. Safe
        for pools far larger than memory. The underlying connection is held
        open for the lifetime of the iterator — fully consume or close it
        promptly.
        """
        samples_table = self._tables[PoolTableType.SAMPLES]
        rows = stream_select_rows(
            self._runtime,
            self.schema,
            samples_table,
            self._tables.select_columns(PoolTableType.SAMPLES),
            order_by=[samples_table.c.sample_idx.asc()],
            key_filter=key_filter,
            chunk_size=chunk_size,
        )
        for row in rows:
            yield PoolSample.from_db_row(self.schema, row)

    _AUTO_IDX_INSERT_RETRIES = 3

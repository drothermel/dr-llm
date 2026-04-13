"""Pool sample storage: CRUD, no-replacement acquisition, coverage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from typing import Any

from sqlalchemy import Text, exists, func, literal, select
from sqlalchemy.engine import Connection
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql.elements import ColumnElement

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import (
    insert_keyed_samples,
    is_constraint_error,
    key_filter_clause,
    stream_select_rows,
    validate_key_values,
)
from dr_llm.pool.call_stats import CallStats
from dr_llm.pool.db.tables import PoolTables
from dr_llm.pool.metadata_store import MetadataStore
from dr_llm.pool.models import AcquireQuery, AcquireResult, CoverageRow, InsertResult
from dr_llm.pool.pending.store import PendingStore
from dr_llm.pool.pool_sample import PoolSample, SampleStatus

SCHEMA_METADATA_KEY = "_schema"
"""Reserved metadata key under which ``ensure_schema`` persists the pool's
``PoolSchema`` so consumers can reconstruct it via :class:`PoolReader`.
"""


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
        multiple times, but each call issues several pg_catalog round-trips
        plus one metadata upsert, so prefer calling exactly once at startup.

        After creating tables, the pool's ``PoolSchema`` is serialized into
        the metadata table under :data:`SCHEMA_METADATA_KEY` so that
        :class:`PoolReader` can later reconstruct it from the database alone.
        The upsert runs in a separate transaction (it must, because the
        metadata store opens its own connection) and is idempotent.
        """
        with self._runtime.begin() as conn:
            self._tables.sa_metadata.create_all(
                bind=conn,
                tables=self._tables.all_tables,
                checkfirst=True,
            )
            self._tables.ensure_indexes(conn)
        self.metadata.upsert(SCHEMA_METADATA_KEY, self.schema.model_dump(mode="json"))

    def insert_call_stats(self, stats: CallStats) -> None:
        """Insert a call-stats row for a promoted sample."""
        row = {
            "sample_id": stats.sample_id,
            "latency_ms": stats.latency_ms,
            "total_cost_usd": stats.total_cost_usd,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "reasoning_tokens": stats.reasoning_tokens,
            "total_tokens": stats.total_tokens,
            "attempt_count": stats.attempt_count,
            "finish_reason": stats.finish_reason,
        }
        with self._runtime.begin() as conn:
            conn.execute(pg_insert(self._tables.call_stats).values(**row))

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
        """Insert auto-idx samples with transaction-scoped per-cell allocation."""
        if not samples:
            return InsertResult()

        samples_table = self._tables.samples
        key_names = self.schema.key_column_names
        base_rows: list[dict[str, Any]] = []
        for sample in samples:
            row = sample.to_db_insert_row()
            row.pop("sample_idx", None)
            base_rows.append(row)

        for attempt in range(1, self._AUTO_IDX_INSERT_RETRIES + 1):
            try:
                with self._runtime.begin() as conn:
                    rows = self._allocate_auto_idx_rows(
                        conn, base_rows=base_rows, key_names=key_names
                    )
                    stmt = pg_insert(samples_table).returning(samples_table.c.sample_id)
                    if ignore_conflicts:
                        stmt = stmt.on_conflict_do_nothing()
                    result = (
                        conn.execute(stmt.values(**rows[0]))
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
        samples_table = self._tables.samples
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
            row["sample_idx"] = (
                max_sample_idx_by_cell[cell_key] + cell_offsets[cell_key]
            )
            rows.append(row)
        return rows

    def _cell_lock_id(self, cell_key: tuple[Any, ...]) -> int:
        lock_payload = json.dumps(
            {
                "pool": self.schema.samples_table,
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

    def sample_count(self) -> int:
        """Return the total number of rows in the pool's samples table."""
        stmt = select(func.count()).select_from(self._tables.samples)
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

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
        self,
        *,
        key_filter: dict[str, Any] | None = None,
        status: SampleStatus | Iterable[SampleStatus] | None = None,
    ) -> list[PoolSample]:
        """Load all samples, optionally filtered by partial key match.

        Materializes the full result set in memory; for pools with 100k+ rows
        prefer ``iter_samples`` to stream in chunks.
        """
        return list(self.iter_samples(key_filter=key_filter, status=status))

    def iter_samples(
        self,
        *,
        key_filter: dict[str, Any] | None = None,
        status: SampleStatus | Iterable[SampleStatus] | None = None,
        chunk_size: int = 1000,
    ) -> Iterator[PoolSample]:
        """Stream samples in chunks via server-side cursoring.

        Uses SQLAlchemy's ``yield_per`` so the driver fetches ``chunk_size``
        rows at a time instead of materializing the entire result set. Safe
        for pools far larger than memory. The underlying connection is held
        open for the lifetime of the iterator — fully consume or close it
        promptly.

        ``status`` accepts a single ``SampleStatus`` or any iterable of them;
        when ``None`` (the default) samples in any status are returned.
        """
        base_predicates: list[ColumnElement[bool]] = []
        if status is not None:
            statuses = _normalize_status_filter(status)
            base_predicates.append(self._tables.samples.c.status.in_(statuses))

        rows = stream_select_rows(
            self._runtime,
            self.schema,
            self._tables.samples,
            self._tables.sample_select_columns(),
            base_predicates=base_predicates,
            order_by=[self._tables.samples.c.sample_idx.asc()],
            key_filter=key_filter,
            chunk_size=chunk_size,
        )
        for row in rows:
            yield PoolSample.from_db_row(self.schema, row)

    _AUTO_IDX_INSERT_RETRIES = 3


def _normalize_status_filter(
    status: SampleStatus | Iterable[SampleStatus],
) -> frozenset[SampleStatus]:
    """Accept a single SampleStatus or any iterable; return a frozenset."""
    if isinstance(status, SampleStatus):
        return frozenset({status})
    return frozenset(status)

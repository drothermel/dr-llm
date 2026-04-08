"""Pool sample storage: CRUD, no-replacement acquisition, coverage."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from psycopg.rows import dict_row

from dr_llm.pool.db.ddl import generate_ddl
from dr_llm.pool.db.metadata import MetadataStore
from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import (
    is_constraint_error,
    key_where_clause,
    q,
    validate_key_filter,
    validate_key_values,
)
from dr_llm.pool.models import (
    AcquireQuery,
    AcquireResult,
    CoverageRow,
    InsertResult,
)
from dr_llm.pool.pending.store import PendingStore
from dr_llm.pool.pool_sample import PoolSample

logger = logging.getLogger(__name__)

_SAMPLE_IDX_RETRY_ATTEMPTS = 5

# Safety note: table and column names used in f-string SQL interpolation are
# validated by PoolSchema/KeyColumn to match ^[a-z][a-z0-9_]*$ — no user input
# reaches these identifiers without passing that regex gate.


class PoolStore:
    """Pool storage operations parameterized by schema."""

    def __init__(self, schema: PoolSchema, runtime: DbRuntime) -> None:
        self._schema = schema
        self._runtime = runtime
        self._schema_lock = threading.Lock()
        self._schema_initialized = False
        self._pending = PendingStore(schema, runtime)
        self._metadata = MetadataStore(schema, runtime)

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
        *,
        allow_destructive_cleanup: bool = False,
        dedicated_schema: str | None = None,
    ) -> None:
        """Create pool tables if they don't exist."""
        if self._schema_initialized:
            return
        with self._schema_lock:
            if self._schema_initialized:
                return
            self._runtime.cleanup_legacy_tables(
                allow_destructive_cleanup=allow_destructive_cleanup,
                dedicated_schema=dedicated_schema,
            )
            ddl = generate_ddl(self._schema)
            with self._runtime.conn() as conn:
                conn.execute(q(ddl))
                conn.commit()
            self._schema_initialized = True

    # --- Sample CRUD ---

    def insert_sample(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert a single sample. Auto-assigns sample_idx if None."""
        self.init_schema()
        validate_key_values(self._schema, sample.key_values)
        tbl = self._schema.samples_table

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
                        tbl,
                        attempt + 1,
                        _SAMPLE_IDX_RETRY_ATTEMPTS,
                    )
                    continue
                logger.warning(
                    "Failed to allocate sample_idx after %d attempts for %s",
                    _SAMPLE_IDX_RETRY_ATTEMPTS,
                    tbl,
                )
                return False
        return False

    def _insert_sample_row(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert with explicit sample_idx."""
        tbl = self._schema.samples_table
        insert_row = sample.to_db_insert_row(self._schema)
        cols = list(insert_row.keys())
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(cols)
        conflict = " ON CONFLICT DO NOTHING" if ignore_conflicts else ""
        values = list(insert_row.values())

        with self._runtime.conn() as conn:
            try:
                cur = conn.execute(
                    q(
                        f"INSERT INTO {tbl} ({col_list}) VALUES ({placeholders}){conflict}"
                    ),
                    values,
                )
                conn.commit()
                return (cur.rowcount or 0) > 0
            except Exception as exc:
                conn.rollback()
                if ignore_conflicts and is_constraint_error(exc):
                    return False
                raise

    def _insert_sample_auto_idx(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert with auto-incrementing sample_idx via CTE."""
        tbl = self._schema.samples_table

        kw, kp = key_where_clause(self._schema, sample.key_values)

        insert_row = sample.to_db_insert_row(self._schema)
        cols = list(insert_row.keys())
        col_list = ", ".join(cols)
        conflict = " ON CONFLICT DO NOTHING" if ignore_conflicts else ""

        select_parts: list[str] = []
        values: list[Any] = list(kp)
        for col_name, value in insert_row.items():
            if col_name == "sample_idx":
                select_parts.append("next_idx.idx")
                continue
            select_parts.append("%s")
            values.append(value)

        insert_sql = (
            f"WITH next_idx AS ("
            f"  SELECT COALESCE(MAX(sample_idx), -1) + 1 AS idx"
            f"  FROM {tbl} WHERE {kw}"
            f") "
            f"INSERT INTO {tbl} ({col_list}) "
            f"SELECT {', '.join(select_parts)} "
            f"FROM next_idx{conflict}"
        )

        with self._runtime.conn() as conn:
            try:
                cur = conn.execute(q(insert_sql), values)
                conn.commit()
                return (cur.rowcount or 0) > 0
            except Exception as exc:
                conn.rollback()
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
        for s in samples:
            validate_key_values(self._schema, s.key_values)
            if s.sample_idx is not None:
                explicit.append(s)
            else:
                auto_idx.append(s)

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
        tbl = self._schema.samples_table
        first_row = samples[0].to_db_insert_row(self._schema)
        cols = list(first_row.keys())
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(cols)
        conflict = " ON CONFLICT DO NOTHING" if ignore_conflicts else ""

        all_values = [
            list(sample.to_db_insert_row(self._schema).values()) for sample in samples
        ]

        values_clause = ", ".join(f"({placeholders})" for _ in all_values)
        flat_params = [v for row in all_values for v in row]

        with self._runtime.conn() as conn:
            try:
                cur = conn.execute(
                    q(
                        f"INSERT INTO {tbl} ({col_list}) VALUES {values_clause}{conflict}"
                    ),
                    flat_params,
                )
                conn.commit()
                row_count = cur.rowcount or 0
                return InsertResult(
                    inserted=row_count,
                    skipped=len(samples) - row_count,
                )
            except Exception as exc:
                conn.rollback()
                if ignore_conflicts and is_constraint_error(exc):
                    return InsertResult(skipped=len(samples))
                raise

    # --- No-Replacement Acquisition ---

    def acquire(self, query: AcquireQuery) -> AcquireResult:
        """Acquire up to query.n unclaimed samples for given key dimensions."""
        self.init_schema()
        validate_key_values(self._schema, query.key_values)
        samples_tbl = self._schema.samples_table
        claims_tbl = self._schema.claims_table
        kw, kp = key_where_clause(self._schema, query.key_values)

        select_sql = (
            "SELECT "
            + ", ".join(PoolSample.db_select_columns(self._schema))
            + f" FROM {samples_tbl} "
            f"WHERE {kw} "
            f"AND status = 'active' "
            f"AND NOT EXISTS ("
            f"  SELECT 1 FROM {claims_tbl} c "
            f"  WHERE c.run_id = %s AND c.sample_id = {samples_tbl}.sample_id"
            f") "
            f"ORDER BY sample_idx ASC, created_at ASC "
            f"LIMIT %s"
        )
        select_params = kp + [query.run_id, query.n]

        with self._runtime.conn() as conn:
            try:
                with conn.cursor(row_factory=dict_row) as cur:
                    rows = cur.execute(q(select_sql), select_params).fetchall()
                samples: list[PoolSample] = []
                claimed = 0

                for idx, row in enumerate(rows):
                    sample = PoolSample.from_db_row(self._schema, row)

                    claim_id = uuid4().hex
                    try:
                        conn.execute(
                            q(
                                f"INSERT INTO {claims_tbl} "
                                f"(claim_id, run_id, request_id, consumer_tag, sample_id, claim_idx) "
                                f"VALUES (%s, %s, %s, %s, %s, %s)"
                            ),
                            [
                                claim_id,
                                query.run_id,
                                query.request_id,
                                query.consumer_tag,
                                row["sample_id"],
                                idx,
                            ],
                        )
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

                conn.commit()
                return AcquireResult(samples=samples, claimed=claimed)
            except Exception:
                conn.rollback()
                raise

    def acquire_batch(self, queries: list[AcquireQuery]) -> dict[str, AcquireResult]:
        """Convenience wrapper: acquire() for each query, returning results by request_id."""
        results: dict[str, AcquireResult] = {}
        for query_item in queries:
            results[query_item.request_id] = self.acquire(query_item)
        return results

    def remaining(self, *, run_id: str, key_values: dict[str, Any]) -> int:
        """Count unclaimed samples for given key dimensions and run."""
        self.init_schema()
        validate_key_values(self._schema, key_values)
        samples_tbl = self._schema.samples_table
        claims_tbl = self._schema.claims_table
        kw, kp = key_where_clause(self._schema, key_values)

        count_sql = (
            f"SELECT COUNT(*) FROM {samples_tbl} "
            f"WHERE {kw} "
            f"AND status = 'active' "
            f"AND NOT EXISTS ("
            f"  SELECT 1 FROM {claims_tbl} c "
            f"  WHERE c.run_id = %s AND c.sample_id = {samples_tbl}.sample_id"
            f")"
        )
        with self._runtime.conn() as conn:
            row = conn.execute(q(count_sql), kp + [run_id]).fetchone()
            return row[0] if row else 0

    # --- Coverage / Analytics ---

    def coverage(self) -> list[CoverageRow]:
        """Return sample counts grouped by all key dimensions."""
        self.init_schema()
        tbl = self._schema.samples_table
        key_list = ", ".join(self._schema.key_column_names)
        with self._runtime.conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    q(
                        f"SELECT {key_list}, COUNT(*) as cnt FROM {tbl} GROUP BY {key_list}"
                    )
                ).fetchall()
            return [
                CoverageRow(
                    key_values={
                        name: row[name] for name in self._schema.key_column_names
                    },
                    count=row["cnt"],
                )
                for row in rows
            ]

    def cell_depth(self, *, key_values: dict[str, Any]) -> int:
        """Count total samples for a specific cell."""
        self.init_schema()
        validate_key_values(self._schema, key_values)
        tbl = self._schema.samples_table
        kw, kp = key_where_clause(self._schema, key_values)
        with self._runtime.conn() as conn:
            row = conn.execute(
                q(f"SELECT COUNT(*) FROM {tbl} WHERE {kw}"), kp
            ).fetchone()
            return row[0] if row else 0

    # --- Bulk Load ---

    def bulk_load(
        self, *, key_filter: dict[str, Any] | None = None
    ) -> list[PoolSample]:
        """Load all samples, optionally filtered by partial key match."""
        self.init_schema()
        tbl = self._schema.samples_table
        all_cols = PoolSample.db_select_columns(self._schema)
        col_list = ", ".join(all_cols)

        where_parts: list[str] = []
        params: list[Any] = []
        if key_filter:
            validate_key_filter(self._schema, key_filter)
            for k, v in key_filter.items():
                if k in self._schema.key_column_names:
                    where_parts.append(f"{k} = %s")
                    params.append(v)

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        with self._runtime.conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    q(
                        f"SELECT {col_list} FROM {tbl} {where_clause} ORDER BY sample_idx ASC"
                    ),
                    params,
                ).fetchall()

            return [PoolSample.from_db_row(self._schema, row) for row in rows]

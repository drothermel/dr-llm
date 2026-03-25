"""Generic pool storage operations parameterized by schema."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable
from typing import Any, LiteralString, cast
from uuid import uuid4

from psycopg import errors as pg_errors
from psycopg import sql
from psycopg.rows import dict_row

from dr_llm.pool.ddl import generate_ddl
from dr_llm.pool.errors import PoolSchemaError
from dr_llm.pool.models import (
    AcquireQuery,
    AcquireResult,
    CoverageRow,
    InsertResult,
    PendingSample,
    PendingStatus,
    PoolSample,
)
from dr_llm.pool.schema import PoolSchema
from dr_llm.storage._runtime import StorageRuntime

logger = logging.getLogger(__name__)

_SAMPLE_IDX_RETRY_ATTEMPTS = 5

# Safety note: table and column names used in f-string SQL interpolation are
# validated by PoolSchema/KeyColumn to match ^[a-z][a-z0-9_]*$ — no user input
# reaches these identifiers without passing that regex gate.


def _q(query: str) -> sql.SQL:
    """Wrap a dynamically built query string as sql.SQL for psycopg's type checker.

    Safe because all interpolated identifiers (table/column names) are validated
    against ^[a-z][a-z0-9_]*$ by PoolSchema/KeyColumn before reaching here.
    """
    return sql.SQL(cast(LiteralString, query))


def _is_constraint_error(exc: BaseException) -> bool:
    return isinstance(
        exc, (pg_errors.UniqueViolation, pg_errors.IntegrityConstraintViolation)
    )


def _parse_json_field(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return json.loads(raw or "{}")


class PoolStore:
    """Generic pool storage operations parameterized by schema."""

    def __init__(self, schema: PoolSchema, runtime: StorageRuntime) -> None:
        self._schema = schema
        self._runtime = runtime
        self._schema_lock = threading.Lock()
        self._schema_initialized = False

    @property
    def schema(self) -> PoolSchema:
        return self._schema

    def init_schema(self) -> None:
        """Create pool tables if they don't exist."""
        if self._schema_initialized:
            return
        with self._schema_lock:
            if self._schema_initialized:
                return
            self._runtime.open_pool()
            ddl = generate_ddl(self._schema)
            with self._runtime.conn() as conn:
                conn.execute(_q(ddl))
                conn.commit()
            self._schema_initialized = True

    def _validate_key_values(self, key_values: dict[str, Any]) -> None:
        expected = set(self._schema.key_column_names)
        provided = set(key_values.keys())
        missing = expected - provided
        if missing:
            raise PoolSchemaError(
                f"Missing key columns: {missing}. Expected: {expected}"
            )

    def _key_where_clause(self, key_values: dict[str, Any]) -> tuple[str, list[Any]]:
        """Build WHERE clause for key column matching."""
        conditions: list[str] = []
        params: list[Any] = []
        for kc in self._schema.key_columns:
            conditions.append(f"{kc.name} = %s")
            params.append(key_values[kc.name])
        return " AND ".join(conditions), params

    def _key_values_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {name: row[name] for name in self._schema.key_column_names}

    def _validate_key_filter(self, key_filter: dict[str, Any]) -> None:
        """Warn on unknown keys in a partial key filter."""
        unknown = set(key_filter.keys()) - set(self._schema.key_column_names)
        if unknown:
            logger.warning(
                "key_filter contains unknown columns %s (valid: %s)",
                unknown,
                self._schema.key_column_names,
            )

    # --- Sample CRUD ---

    def insert_sample(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert a single sample. Auto-assigns sample_idx if None."""
        self.init_schema()
        self._validate_key_values(sample.key_values)
        tbl = self._schema.samples_table

        if sample.sample_idx is not None:
            return self._insert_sample_row(sample, ignore_conflicts=ignore_conflicts)

        for attempt in range(_SAMPLE_IDX_RETRY_ATTEMPTS):
            try:
                return self._insert_sample_auto_idx(
                    sample, ignore_conflicts=ignore_conflicts
                )
            except Exception as exc:
                if not (_is_constraint_error(exc) and ignore_conflicts):
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
        key_names = self._schema.key_column_names
        cols = (
            ["sample_id"]
            + key_names
            + [
                "sample_idx",
                "payload_json",
                "source_run_id",
                "call_id",
                "metadata_json",
                "status",
            ]
        )
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(cols)
        conflict = " ON CONFLICT DO NOTHING" if ignore_conflicts else ""

        values = [sample.sample_id]
        for name in key_names:
            values.append(sample.key_values[name])
        values.extend(
            [
                sample.sample_idx,
                json.dumps(sample.payload, default=str),
                sample.source_run_id,
                sample.call_id,
                json.dumps(sample.metadata, default=str),
                sample.status.value,
            ]
        )

        with self._runtime.conn() as conn:
            try:
                cur = conn.execute(
                    _q(
                        f"INSERT INTO {tbl} ({col_list}) VALUES ({placeholders}){conflict}"
                    ),
                    values,
                )
                conn.commit()
                return (cur.rowcount or 0) > 0
            except Exception as exc:
                conn.rollback()
                if ignore_conflicts and _is_constraint_error(exc):
                    return False
                raise

    def _insert_sample_auto_idx(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert with auto-incrementing sample_idx via CTE."""
        tbl = self._schema.samples_table
        key_names = self._schema.key_column_names

        key_where, key_params = self._key_where_clause(sample.key_values)

        cols = (
            ["sample_id"]
            + key_names
            + [
                "sample_idx",
                "payload_json",
                "source_run_id",
                "call_id",
                "metadata_json",
                "status",
            ]
        )
        col_list = ", ".join(cols)
        conflict = " ON CONFLICT DO NOTHING" if ignore_conflicts else ""

        # Build SELECT values: literals for all columns except sample_idx which
        # comes from the CTE.
        select_parts: list[str] = ["%s"]  # sample_id
        for _ in key_names:
            select_parts.append("%s")
        select_parts.append("next_idx.idx")  # sample_idx from CTE
        select_parts.append("%s")  # payload_json
        select_parts.append("%s")  # source_run_id
        select_parts.append("%s")  # call_id
        select_parts.append("%s")  # metadata_json
        select_parts.append("%s")  # status

        insert_sql = (
            f"WITH next_idx AS ("
            f"  SELECT COALESCE(MAX(sample_idx), -1) + 1 AS idx"
            f"  FROM {tbl} WHERE {key_where}"
            f") "
            f"INSERT INTO {tbl} ({col_list}) "
            f"SELECT {', '.join(select_parts)} "
            f"FROM next_idx{conflict}"
        )

        # CTE params (key_where), then SELECT params (everything except sample_idx)
        values: list[Any] = list(key_params)
        values.append(sample.sample_id)
        for name in key_names:
            values.append(sample.key_values[name])
        values.extend(
            [
                json.dumps(sample.payload, default=str),
                sample.source_run_id,
                sample.call_id,
                json.dumps(sample.metadata, default=str),
                sample.status.value,
            ]
        )

        with self._runtime.conn() as conn:
            try:
                cur = conn.execute(_q(insert_sql), values)
                conn.commit()
                return (cur.rowcount or 0) > 0
            except Exception as exc:
                conn.rollback()
                if ignore_conflicts and _is_constraint_error(exc):
                    return False
                raise

    def insert_samples(
        self, samples: Iterable[PoolSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        """Bulk insert samples.

        Samples with an explicit sample_idx are batched into a single INSERT
        statement. Samples without sample_idx require serial auto-idx allocation
        and fall back to per-row inserts.
        """
        self.init_schema()
        explicit: list[PoolSample] = []
        auto_idx: list[PoolSample] = []
        for s in samples:
            self._validate_key_values(s.key_values)
            if s.sample_idx is not None:
                explicit.append(s)
            else:
                auto_idx.append(s)

        inserted = 0
        skipped = 0
        failed = 0

        # Batch insert for explicit-idx samples
        if explicit:
            result = self._batch_insert_explicit(
                explicit, ignore_conflicts=ignore_conflicts
            )
            inserted += result.inserted
            skipped += result.skipped
            failed += result.failed

        # Per-row insert for auto-idx samples (each needs MAX+1 query)
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
        key_names = self._schema.key_column_names
        cols = (
            ["sample_id"]
            + key_names
            + [
                "sample_idx",
                "payload_json",
                "source_run_id",
                "call_id",
                "metadata_json",
                "status",
            ]
        )
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(cols)
        conflict = " ON CONFLICT DO NOTHING" if ignore_conflicts else ""

        all_values: list[list[Any]] = []
        for sample in samples:
            values: list[Any] = [sample.sample_id]
            for name in key_names:
                values.append(sample.key_values[name])
            values.extend(
                [
                    sample.sample_idx,
                    json.dumps(sample.payload, default=str),
                    sample.source_run_id,
                    sample.call_id,
                    json.dumps(sample.metadata, default=str),
                    sample.status.value,
                ]
            )
            all_values.append(values)

        values_clause = ", ".join(f"({placeholders})" for _ in all_values)
        flat_params = [v for row in all_values for v in row]

        with self._runtime.conn() as conn:
            try:
                cur = conn.execute(
                    _q(
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
                if ignore_conflicts and _is_constraint_error(exc):
                    return InsertResult(skipped=len(samples))
                raise

    # --- No-Replacement Acquisition ---

    def acquire(self, query: AcquireQuery) -> AcquireResult:
        """
        Acquire up to query.n unclaimed samples for given key dimensions.
        Records claims so same sample is never drawn twice in one run_id.
        """
        self.init_schema()
        self._validate_key_values(query.key_values)
        samples_tbl = self._schema.samples_table
        claims_tbl = self._schema.claims_table
        key_names = self._schema.key_column_names
        key_where, key_params = self._key_where_clause(query.key_values)

        select_sql = (
            "SELECT sample_id, sample_idx, payload_json, source_run_id, "
            "call_id, metadata_json, status, created_at, "
            + ", ".join(key_names)
            + f" FROM {samples_tbl} "
            f"WHERE {key_where} "
            f"AND status = 'active' "
            f"AND NOT EXISTS ("
            f"  SELECT 1 FROM {claims_tbl} c "
            f"  WHERE c.run_id = %s AND c.sample_id = {samples_tbl}.sample_id"
            f") "
            f"ORDER BY sample_idx ASC, created_at ASC "
            f"LIMIT %s"
        )
        select_params = key_params + [query.run_id, query.n]

        with self._runtime.conn() as conn:
            try:
                with conn.cursor(row_factory=dict_row) as cur:
                    rows = cur.execute(_q(select_sql), select_params).fetchall()
                samples: list[PoolSample] = []
                claimed = 0

                for idx, row in enumerate(rows):
                    key_vals = {name: row[name] for name in key_names}
                    sample = PoolSample(
                        sample_id=row["sample_id"],
                        sample_idx=row["sample_idx"],
                        key_values=key_vals,
                        payload=_parse_json_field(row["payload_json"]),
                        source_run_id=row["source_run_id"],
                        call_id=row["call_id"],
                        metadata=_parse_json_field(row["metadata_json"]),
                        status=row["status"],
                        created_at=row["created_at"],
                    )

                    claim_id = uuid4().hex
                    try:
                        conn.execute(
                            _q(
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
                        if _is_constraint_error(exc):
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
        """Convenience wrapper: acquire() for each query, returning results by request_id.

        No SQL-level batching — each query runs a separate acquire() call.
        """
        results: dict[str, AcquireResult] = {}
        for query in queries:
            results[query.request_id] = self.acquire(query)
        return results

    def remaining(self, *, run_id: str, key_values: dict[str, Any]) -> int:
        """Count unclaimed samples for given key dimensions and run."""
        self.init_schema()
        self._validate_key_values(key_values)
        samples_tbl = self._schema.samples_table
        claims_tbl = self._schema.claims_table
        key_where, key_params = self._key_where_clause(key_values)

        count_sql = (
            f"SELECT COUNT(*) FROM {samples_tbl} "
            f"WHERE {key_where} "
            f"AND status = 'active' "
            f"AND NOT EXISTS ("
            f"  SELECT 1 FROM {claims_tbl} c "
            f"  WHERE c.run_id = %s AND c.sample_id = {samples_tbl}.sample_id"
            f")"
        )
        with self._runtime.conn() as conn:
            row = conn.execute(_q(count_sql), key_params + [run_id]).fetchone()
            return row[0] if row else 0

    # --- Pending Samples ---

    def insert_pending(
        self, sample: PendingSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert a sample in pending state."""
        self.init_schema()
        self._validate_key_values(sample.key_values)
        tbl = self._schema.pending_table
        key_names = self._schema.key_column_names

        cols = (
            ["pending_id"]
            + key_names
            + [
                "sample_idx",
                "payload_json",
                "source_run_id",
                "call_id",
                "metadata_json",
                "priority",
                "status",
            ]
        )
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(cols)
        conflict = " ON CONFLICT DO NOTHING" if ignore_conflicts else ""

        values: list[Any] = [sample.pending_id]
        for name in key_names:
            values.append(sample.key_values[name])
        values.extend(
            [
                sample.sample_idx,
                json.dumps(sample.payload, default=str),
                sample.source_run_id,
                sample.call_id,
                json.dumps(sample.metadata, default=str),
                sample.priority,
                sample.status.value,
            ]
        )

        with self._runtime.conn() as conn:
            try:
                cur = conn.execute(
                    _q(
                        f"INSERT INTO {tbl} ({col_list}) VALUES ({placeholders}){conflict}"
                    ),
                    values,
                )
                conn.commit()
                return (cur.rowcount or 0) > 0
            except Exception as exc:
                conn.rollback()
                if ignore_conflicts and _is_constraint_error(exc):
                    return False
                raise

    def claim_pending(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        key_filter: dict[str, Any] | None = None,
    ) -> list[PendingSample]:
        """Lease pending samples for processing via FOR UPDATE SKIP LOCKED."""
        self.init_schema()
        tbl = self._schema.pending_table
        key_names = self._schema.key_column_names

        where_parts = ["(status = %s OR (status = %s AND lease_expires_at < now()))"]
        params: list[Any] = [PendingStatus.pending.value, PendingStatus.leased.value]

        if key_filter:
            self._validate_key_filter(key_filter)
            for k, v in key_filter.items():
                if k in self._schema.key_column_names:
                    where_parts.append(f"{k} = %s")
                    params.append(v)

        where_clause = " AND ".join(where_parts)
        params.extend([limit])

        all_cols = (
            ["pending_id"]
            + key_names
            + [
                "sample_idx",
                "payload_json",
                "source_run_id",
                "call_id",
                "metadata_json",
                "priority",
                "status",
                "worker_id",
                "lease_expires_at",
                "attempt_count",
                "created_at",
            ]
        )
        claim_sql = (
            f"WITH candidates AS ("
            f"  SELECT pending_id FROM {tbl} "
            f"  WHERE {where_clause} "
            f"  ORDER BY priority DESC, created_at ASC "
            f"  LIMIT %s "
            f"  FOR UPDATE SKIP LOCKED"
            f") "
            f"UPDATE {tbl} t SET "
            f"  status = %s, "
            f"  worker_id = %s, "
            f"  lease_expires_at = now() + make_interval(secs => %s), "
            f"  attempt_count = attempt_count + 1 "
            f"FROM candidates c "
            f"WHERE t.pending_id = c.pending_id "
            f"RETURNING {', '.join(f't.{c}' for c in all_cols)}"
        )
        update_params = params + [PendingStatus.leased.value, worker_id, lease_seconds]

        with self._runtime.conn() as conn:
            try:
                with conn.cursor(row_factory=dict_row) as cur:
                    rows = cur.execute(_q(claim_sql), update_params).fetchall()
                conn.commit()
                results: list[PendingSample] = []
                for row in rows:
                    key_vals = {name: row[name] for name in key_names}
                    results.append(
                        PendingSample(
                            pending_id=row["pending_id"],
                            key_values=key_vals,
                            sample_idx=row["sample_idx"],
                            payload=_parse_json_field(row["payload_json"]),
                            source_run_id=row["source_run_id"],
                            call_id=row["call_id"],
                            metadata=_parse_json_field(row["metadata_json"]),
                            priority=row["priority"],
                            status=PendingStatus.leased,
                            worker_id=worker_id,
                            lease_expires_at=row["lease_expires_at"],
                            attempt_count=row["attempt_count"],
                            created_at=row["created_at"],
                        )
                    )
                return results
            except Exception:
                conn.rollback()
                raise

    def promote_pending(
        self, *, pending_id: str, payload: dict[str, Any] | None = None
    ) -> PoolSample | None:
        """Promote a leased pending sample to finalized.

        Returns None if the pending_id doesn't exist or is not in 'leased' status.
        Inserts into samples table and marks the pending row as promoted.
        """
        self.init_schema()
        pending_tbl = self._schema.pending_table
        samples_tbl = self._schema.samples_table
        key_names = self._schema.key_column_names

        all_cols = (
            ["pending_id", "status"]
            + key_names
            + [
                "sample_idx",
                "payload_json",
                "source_run_id",
                "call_id",
                "metadata_json",
            ]
        )

        with self._runtime.conn() as conn:
            try:
                with conn.cursor(row_factory=dict_row) as cur:
                    row = cur.execute(
                        _q(
                            f"SELECT {', '.join(all_cols)} FROM {pending_tbl} "
                            f"WHERE pending_id = %s"
                        ),
                        [pending_id],
                    ).fetchone()

                if row is None or row["status"] != PendingStatus.leased.value:
                    conn.rollback()
                    return None

                key_vals = {name: row[name] for name in key_names}
                pending_payload = _parse_json_field(row["payload_json"])
                meta = _parse_json_field(row["metadata_json"])
                final_payload = payload if payload is not None else pending_payload
                sample_id = uuid4().hex

                sample_cols = (
                    ["sample_id"]
                    + key_names
                    + [
                        "sample_idx",
                        "payload_json",
                        "source_run_id",
                        "call_id",
                        "metadata_json",
                    ]
                )
                placeholders = ", ".join(["%s"] * len(sample_cols))
                values: list[Any] = [sample_id]
                for name in key_names:
                    values.append(key_vals[name])
                values.extend(
                    [
                        row["sample_idx"],
                        json.dumps(final_payload, default=str),
                        row["source_run_id"],
                        row["call_id"],
                        json.dumps(meta, default=str),
                    ]
                )

                conn.execute(
                    _q(
                        f"INSERT INTO {samples_tbl} ({', '.join(sample_cols)}) "
                        f"VALUES ({placeholders}) ON CONFLICT DO NOTHING"
                    ),
                    values,
                )

                conn.execute(
                    _q(f"UPDATE {pending_tbl} SET status = %s WHERE pending_id = %s"),
                    [PendingStatus.promoted.value, pending_id],
                )

                conn.commit()
                return PoolSample(
                    sample_id=sample_id,
                    sample_idx=row["sample_idx"],
                    key_values=key_vals,
                    payload=final_payload,
                    source_run_id=row["source_run_id"],
                    call_id=row["call_id"],
                    metadata=meta,
                )
            except Exception:
                conn.rollback()
                raise

    def fail_pending(self, *, pending_id: str, reason: str) -> None:
        """Mark pending sample as failed."""
        self.init_schema()
        tbl = self._schema.pending_table
        with self._runtime.conn() as conn:
            conn.execute(
                _q(
                    f"UPDATE {tbl} SET status = %s, "
                    f"metadata_json = jsonb_set(COALESCE(metadata_json, '{{}}'), '{{fail_reason}}', %s::jsonb) "
                    f"WHERE pending_id = %s"
                ),
                [PendingStatus.failed.value, json.dumps(reason), pending_id],
            )
            conn.commit()

    def release_pending_lease(self, *, pending_id: str, worker_id: str) -> None:
        """Release a lease, returning sample to pending status."""
        self.init_schema()
        tbl = self._schema.pending_table
        with self._runtime.conn() as conn:
            conn.execute(
                _q(
                    f"UPDATE {tbl} SET status = %s, worker_id = NULL, lease_expires_at = NULL "
                    f"WHERE pending_id = %s AND worker_id = %s"
                ),
                [PendingStatus.pending.value, pending_id, worker_id],
            )
            conn.commit()

    def pending_counts(self, *, key_values: dict[str, Any]) -> int:
        """Count pending samples for given key dimensions."""
        self.init_schema()
        self._validate_key_values(key_values)
        tbl = self._schema.pending_table
        key_where, key_params = self._key_where_clause(key_values)
        with self._runtime.conn() as conn:
            row = conn.execute(
                _q(f"SELECT COUNT(*) FROM {tbl} WHERE {key_where} AND status = %s"),
                key_params + [PendingStatus.pending.value],
            ).fetchone()
            return row[0] if row else 0

    def pending_counts_grouped(
        self,
        *,
        base_key_values: dict[str, Any],
        group_column: str,
        group_values: list[Any],
    ) -> dict[str, int]:
        """Count pending samples grouped by one varying key dimension.

        Returns {group_value: count} for values with count > 0.
        Single query instead of N queries.
        """
        self.init_schema()
        if group_column not in self._schema.key_column_names:
            raise PoolSchemaError(f"group_column {group_column!r} not in schema")
        tbl = self._schema.pending_table

        where_parts: list[str] = []
        params: list[Any] = []
        for kc in self._schema.key_columns:
            if kc.name == group_column:
                continue
            if kc.name in base_key_values:
                where_parts.append(f"{kc.name} = %s")
                params.append(base_key_values[kc.name])

        if group_values:
            placeholders = ",".join(["%s"] * len(group_values))
            where_parts.append(f"{group_column} IN ({placeholders})")
            params.extend(group_values)

        where_parts.append("status = %s")
        params.append(PendingStatus.pending.value)

        where_clause = " AND ".join(where_parts)
        sql_str = (
            f"SELECT {group_column}, COUNT(*) FROM {tbl} "
            f"WHERE {where_clause} GROUP BY {group_column}"
        )
        with self._runtime.conn() as conn:
            rows = conn.execute(_q(sql_str), params).fetchall()
            return {str(r[0]): int(r[1]) for r in rows if int(r[1]) > 0}

    def bump_pending_priority(
        self, *, key_values: dict[str, Any], priority: int
    ) -> int:
        """Increase priority for pending samples matching key dims."""
        self.init_schema()
        self._validate_key_values(key_values)
        tbl = self._schema.pending_table
        key_where, key_params = self._key_where_clause(key_values)
        with self._runtime.conn() as conn:
            cur = conn.execute(
                _q(
                    f"UPDATE {tbl} SET priority = GREATEST(priority, %s) "
                    f"WHERE {key_where} AND status = %s"
                ),
                [priority] + key_params + [PendingStatus.pending.value],
            )
            conn.commit()
            return cur.rowcount or 0

    # --- Coverage / Analytics ---

    def coverage(self) -> list[CoverageRow]:
        """Return sample counts grouped by all key dimensions."""
        self.init_schema()
        tbl = self._schema.samples_table
        key_names = self._schema.key_column_names
        key_list = ", ".join(key_names)
        with self._runtime.conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _q(
                        f"SELECT {key_list}, COUNT(*) as cnt FROM {tbl} GROUP BY {key_list}"
                    )
                ).fetchall()
            return [
                CoverageRow(
                    key_values={name: row[name] for name in key_names},
                    count=row["cnt"],
                )
                for row in rows
            ]

    def cell_depth(self, *, key_values: dict[str, Any]) -> int:
        """Count total samples for a specific cell."""
        self.init_schema()
        self._validate_key_values(key_values)
        tbl = self._schema.samples_table
        key_where, key_params = self._key_where_clause(key_values)
        with self._runtime.conn() as conn:
            row = conn.execute(
                _q(f"SELECT COUNT(*) FROM {tbl} WHERE {key_where}"), key_params
            ).fetchone()
            return row[0] if row else 0

    # --- Metadata ---

    def upsert_metadata(self, key: str, value_json: dict[str, Any]) -> None:
        """Upsert a metadata entry for this pool."""
        self.init_schema()
        tbl = self._schema.metadata_table
        with self._runtime.conn() as conn:
            conn.execute(
                _q(
                    f"INSERT INTO {tbl} (pool_name, key, value_json) "
                    f"VALUES (%s, %s, %s) "
                    f"ON CONFLICT (pool_name, key) DO UPDATE SET "
                    f"value_json = EXCLUDED.value_json, updated_at = now()"
                ),
                [self._schema.name, key, json.dumps(value_json, default=str)],
            )
            conn.commit()

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get a metadata entry for this pool."""
        self.init_schema()
        tbl = self._schema.metadata_table
        with self._runtime.conn() as conn:
            row = conn.execute(
                _q(f"SELECT value_json FROM {tbl} WHERE pool_name = %s AND key = %s"),
                [self._schema.name, key],
            ).fetchone()
            if row is None:
                return None
            raw = row[0]
            return raw if isinstance(raw, dict) else json.loads(raw)

    # --- Bulk Load ---

    def bulk_load(
        self, *, key_filter: dict[str, Any] | None = None
    ) -> list[PoolSample]:
        """Load all samples, optionally filtered by partial key match."""
        self.init_schema()
        tbl = self._schema.samples_table
        key_names = self._schema.key_column_names

        all_cols = (
            ["sample_id"]
            + key_names
            + [
                "sample_idx",
                "payload_json",
                "source_run_id",
                "call_id",
                "metadata_json",
                "status",
                "created_at",
            ]
        )
        col_list = ", ".join(all_cols)

        where_parts: list[str] = []
        params: list[Any] = []
        if key_filter:
            self._validate_key_filter(key_filter)
            for k, v in key_filter.items():
                if k in self._schema.key_column_names:
                    where_parts.append(f"{k} = %s")
                    params.append(v)

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        with self._runtime.conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    _q(
                        f"SELECT {col_list} FROM {tbl} {where_clause} ORDER BY sample_idx ASC"
                    ),
                    params,
                ).fetchall()

            results: list[PoolSample] = []
            for row in rows:
                key_vals = {name: row[name] for name in key_names}
                results.append(
                    PoolSample(
                        sample_id=row["sample_id"],
                        sample_idx=row["sample_idx"],
                        key_values=key_vals,
                        payload=_parse_json_field(row["payload_json"]),
                        source_run_id=row["source_run_id"],
                        call_id=row["call_id"],
                        metadata=_parse_json_field(row["metadata_json"]),
                        status=row["status"],
                        created_at=row["created_at"],
                    )
                )
            return results

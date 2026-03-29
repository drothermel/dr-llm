"""Pending sample lifecycle: insert, lease, promote, fail, release."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from psycopg.rows import dict_row

from dr_llm.pool.errors import PoolSchemaError
from dr_llm.pool.pool_schema import PoolSchema
from dr_llm.pool.runtime import DbRuntime
from dr_llm.pool.sample_models import PendingSample, PendingStatus, PoolSample
from dr_llm.pool.sql_helpers import (
    is_constraint_error,
    key_where_clause,
    parse_json_field,
    q,
    validate_key_filter,
    validate_key_values,
)

logger = logging.getLogger(__name__)


class PendingStore:
    """Pending sample lifecycle operations."""

    def __init__(self, schema: PoolSchema, runtime: DbRuntime) -> None:
        self._schema = schema
        self._runtime = runtime

    def insert_pending(
        self, sample: PendingSample, *, ignore_conflicts: bool = True
    ) -> bool:
        validate_key_values(self._schema, sample.key_values)
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

    def claim_pending(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        key_filter: dict[str, Any] | None = None,
    ) -> list[PendingSample]:
        """Lease pending samples for processing via FOR UPDATE SKIP LOCKED."""
        tbl = self._schema.pending_table
        key_names = self._schema.key_column_names

        where_parts = ["(status = %s OR (status = %s AND lease_expires_at < now()))"]
        params: list[Any] = [PendingStatus.pending.value, PendingStatus.leased.value]

        if key_filter:
            validate_key_filter(self._schema, key_filter)
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
                    rows = cur.execute(q(claim_sql), update_params).fetchall()
                conn.commit()
                results: list[PendingSample] = []
                for row in rows:
                    key_vals = {name: row[name] for name in key_names}
                    results.append(
                        PendingSample(
                            pending_id=row["pending_id"],
                            key_values=key_vals,
                            sample_idx=row["sample_idx"],
                            payload=parse_json_field(row["payload_json"]),
                            source_run_id=row["source_run_id"],
                            call_id=row["call_id"],
                            metadata=parse_json_field(row["metadata_json"]),
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
                        q(
                            f"SELECT {', '.join(all_cols)} FROM {pending_tbl} "
                            f"WHERE pending_id = %s FOR UPDATE"
                        ),
                        [pending_id],
                    ).fetchone()

                if row is None or row["status"] != PendingStatus.leased.value:
                    conn.rollback()
                    return None

                key_vals = {name: row[name] for name in key_names}
                pending_payload = parse_json_field(row["payload_json"])
                meta = parse_json_field(row["metadata_json"])
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

                insert_cur = conn.execute(
                    q(
                        f"INSERT INTO {samples_tbl} ({', '.join(sample_cols)}) "
                        f"VALUES ({placeholders}) ON CONFLICT DO NOTHING"
                    ),
                    values,
                )

                if (insert_cur.rowcount or 0) == 0:
                    conn.rollback()
                    logger.warning(
                        "promote_pending: sample insert conflict for pending_id=%s",
                        pending_id,
                    )
                    return None

                conn.execute(
                    q(f"UPDATE {pending_tbl} SET status = %s WHERE pending_id = %s"),
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

    def fail_pending(self, *, pending_id: str, worker_id: str, reason: str) -> None:
        """Mark a leased pending sample as failed."""
        tbl = self._schema.pending_table
        with self._runtime.conn() as conn:
            conn.execute(
                q(
                    f"UPDATE {tbl} SET status = %s, "
                    f"metadata_json = jsonb_set(COALESCE(metadata_json, '{{}}'), '{{fail_reason}}', %s::jsonb) "
                    f"WHERE pending_id = %s AND worker_id = %s AND status = %s"
                ),
                [
                    PendingStatus.failed.value,
                    json.dumps(reason),
                    pending_id,
                    worker_id,
                    PendingStatus.leased.value,
                ],
            )
            conn.commit()

    def release_pending_lease(self, *, pending_id: str, worker_id: str) -> None:
        """Release a lease, returning sample to pending status."""
        tbl = self._schema.pending_table
        with self._runtime.conn() as conn:
            conn.execute(
                q(
                    f"UPDATE {tbl} SET status = %s, worker_id = NULL, lease_expires_at = NULL "
                    f"WHERE pending_id = %s AND worker_id = %s AND status = %s"
                ),
                [
                    PendingStatus.pending.value,
                    pending_id,
                    worker_id,
                    PendingStatus.leased.value,
                ],
            )
            conn.commit()

    def pending_counts(self, *, key_values: dict[str, Any]) -> int:
        """Count in-flight pending samples (pending + leased) for given key dimensions."""
        validate_key_values(self._schema, key_values)
        tbl = self._schema.pending_table
        kw, kp = key_where_clause(self._schema, key_values)
        with self._runtime.conn() as conn:
            row = conn.execute(
                q(f"SELECT COUNT(*) FROM {tbl} WHERE {kw} AND status IN (%s, %s)"),
                kp + [PendingStatus.pending.value, PendingStatus.leased.value],
            ).fetchone()
            return row[0] if row else 0

    def pending_counts_grouped(
        self,
        *,
        base_key_values: dict[str, Any],
        group_column: str,
        group_values: list[Any],
    ) -> dict[str, int]:
        """Count pending samples grouped by one varying key dimension."""
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
            rows = conn.execute(q(sql_str), params).fetchall()
            return {str(r[0]): int(r[1]) for r in rows if int(r[1]) > 0}

    def bump_pending_priority(
        self, *, key_values: dict[str, Any], priority: int
    ) -> int:
        """Increase priority for pending samples matching key dims."""
        validate_key_values(self._schema, key_values)
        tbl = self._schema.pending_table
        kw, kp = key_where_clause(self._schema, key_values)
        with self._runtime.conn() as conn:
            cur = conn.execute(
                q(
                    f"UPDATE {tbl} SET priority = GREATEST(priority, %s) "
                    f"WHERE {kw} AND status = %s"
                ),
                [priority] + kp + [PendingStatus.pending.value],
            )
            conn.commit()
            return cur.rowcount or 0

    def bulk_load_pending(
        self, *, key_filter: dict[str, Any] | None = None
    ) -> list[PendingSample]:
        """Load in-flight pending samples, optionally filtered by partial key match."""
        tbl = self._schema.pending_table
        key_names = self._schema.key_column_names

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
        col_list = ", ".join(all_cols)

        where_parts: list[str] = ["status IN (%s, %s)"]
        params: list[Any] = [PendingStatus.pending.value, PendingStatus.leased.value]
        if key_filter:
            validate_key_filter(self._schema, key_filter)
            for k, v in key_filter.items():
                if k in self._schema.key_column_names:
                    where_parts.append(f"{k} = %s")
                    params.append(v)

        where_clause = f"WHERE {' AND '.join(where_parts)}"

        with self._runtime.conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    q(
                        f"SELECT {col_list} FROM {tbl} {where_clause} "
                        f"ORDER BY priority DESC, created_at ASC"
                    ),
                    params,
                ).fetchall()

            results: list[PendingSample] = []
            for row in rows:
                key_vals = {name: row[name] for name in key_names}
                results.append(
                    PendingSample(
                        pending_id=row["pending_id"],
                        key_values=key_vals,
                        sample_idx=row["sample_idx"],
                        payload=parse_json_field(row["payload_json"]),
                        source_run_id=row["source_run_id"],
                        call_id=row["call_id"],
                        metadata=parse_json_field(row["metadata_json"]),
                        priority=row["priority"],
                        status=PendingStatus(row["status"]),
                        worker_id=row["worker_id"],
                        lease_expires_at=row["lease_expires_at"],
                        attempt_count=row["attempt_count"],
                        created_at=row["created_at"],
                    )
                )
            return results

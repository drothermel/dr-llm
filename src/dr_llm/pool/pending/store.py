"""Pending sample lifecycle: insert, lease, promote, fail, release."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, cast, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import INTERVAL, JSONB, insert as pg_insert

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import (
    is_constraint_error,
    key_filter_clause,
    partial_key_filter_clause,
    resolve_group_column,
    validate_key_values,
)
from dr_llm.pool.db.tables import PoolTables
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatus, PendingStatusCounts
from dr_llm.pool.pool_sample import PoolSample

logger = logging.getLogger(__name__)


class PendingStore:
    """Pending sample lifecycle operations."""

    def __init__(
        self, schema: PoolSchema, runtime: DbRuntime, tables: PoolTables
    ) -> None:
        self._schema = schema
        self._runtime = runtime
        self._tables = tables

    def insert_pending(
        self, sample: PendingSample, *, ignore_conflicts: bool = True
    ) -> bool:
        validate_key_values(self._schema, sample.key_values)
        stmt = pg_insert(self._tables.pending).values(
            **sample.to_db_insert_row(self._schema)
        )
        if ignore_conflicts:
            stmt = stmt.on_conflict_do_nothing()
        stmt = stmt.returning(self._tables.pending.c.pending_id)

        with self._runtime.begin() as conn:
            try:
                inserted_pending_id = conn.execute(stmt).scalar_one_or_none()
                return inserted_pending_id is not None
            except Exception as exc:
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
        reusable = and_(
            self._tables.pending.c.status == PendingStatus.leased.value,
            self._tables.pending.c.lease_expires_at < func.now(),
        )
        predicates = [
            or_(
                self._tables.pending.c.status == PendingStatus.pending.value,
                reusable,
            )
        ]
        partial_filter = partial_key_filter_clause(
            self._schema, self._tables.pending, key_filter
        )
        if partial_filter is not None:
            predicates.append(partial_filter)

        candidates = (
            select(self._tables.pending.c.pending_id)
            .where(*predicates)
            .order_by(
                self._tables.pending.c.priority.desc(),
                self._tables.pending.c.created_at.asc(),
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
            .cte("candidates")
        )
        stmt = (
            update(self._tables.pending)
            .where(self._tables.pending.c.pending_id == candidates.c.pending_id)
            .values(
                status=PendingStatus.leased.value,
                worker_id=worker_id,
                lease_expires_at=func.now()
                + cast(literal(f"{lease_seconds} seconds"), INTERVAL),
                attempt_count=self._tables.pending.c.attempt_count + 1,
            )
            .returning(*self._tables.pending_select_columns())
        )

        with self._runtime.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [PendingSample.from_db_row(self._schema, dict(row)) for row in rows]

    def promote_pending(
        self, *, pending_id: str, payload: dict[str, Any] | None = None
    ) -> PoolSample | None:
        """Promote a leased pending sample to finalized.

        Returns None if the pending_id doesn't exist or is not in 'leased' status.
        Inserts into samples table and marks the pending row as promoted.
        """
        stmt = (
            select(*self._tables.pending_select_columns())
            .where(self._tables.pending.c.pending_id == pending_id)
            .with_for_update()
        )

        with self._runtime.begin() as conn:
            row = conn.execute(stmt).mappings().first()
            if row is None or row["status"] != PendingStatus.leased.value:
                return None

            pending_sample = PendingSample.from_db_row(self._schema, dict(row))
            final_payload = payload if payload is not None else pending_sample.payload
            sample = PoolSample(
                sample_id=uuid4().hex,
                sample_idx=pending_sample.sample_idx,
                key_values=pending_sample.key_values,
                payload=final_payload,
                source_run_id=pending_sample.source_run_id,
                metadata=pending_sample.metadata,
            )
            insert_stmt = pg_insert(self._tables.samples).values(
                **sample.to_db_insert_row(self._schema)
            )
            insert_stmt = insert_stmt.on_conflict_do_nothing()
            insert_stmt = insert_stmt.returning(self._tables.samples.c.sample_id)
            inserted_sample_id = conn.execute(insert_stmt).scalar_one_or_none()
            if inserted_sample_id is None:
                logger.warning(
                    "promote_pending: sample insert conflict for pending_id=%s",
                    pending_id,
                )
                return None

            conn.execute(
                update(self._tables.pending)
                .where(self._tables.pending.c.pending_id == pending_id)
                .values(status=PendingStatus.promoted.value)
            )
            return sample

    def fail_pending(self, *, pending_id: str, worker_id: str, reason: str) -> None:
        """Mark a leased pending sample as failed."""
        metadata_patch = func.jsonb_build_object("fail_reason", reason)
        stmt = (
            update(self._tables.pending)
            .where(
                self._tables.pending.c.pending_id == pending_id,
                self._tables.pending.c.worker_id == worker_id,
                self._tables.pending.c.status == PendingStatus.leased.value,
            )
            .values(
                status=PendingStatus.failed.value,
                metadata_json=func.coalesce(
                    self._tables.pending.c.metadata_json,
                    literal({}, type_=JSONB),
                ).op("||")(metadata_patch),
            )
        )
        with self._runtime.begin() as conn:
            conn.execute(stmt)

    def release_pending_lease(self, *, pending_id: str, worker_id: str) -> None:
        """Release a lease, returning sample to pending status."""
        stmt = (
            update(self._tables.pending)
            .where(
                self._tables.pending.c.pending_id == pending_id,
                self._tables.pending.c.worker_id == worker_id,
                self._tables.pending.c.status == PendingStatus.leased.value,
            )
            .values(
                status=PendingStatus.pending.value,
                worker_id=None,
                lease_expires_at=None,
            )
        )
        with self._runtime.begin() as conn:
            conn.execute(stmt)

    def pending_counts(self, *, key_values: dict[str, Any]) -> int:
        """Count in-flight pending samples (pending + leased) for given key dimensions."""
        validate_key_values(self._schema, key_values)
        stmt = select(func.count()).where(
            key_filter_clause(self._schema, self._tables.pending, key_values),
            self._tables.pending.c.status.in_(
                [PendingStatus.pending.value, PendingStatus.leased.value]
            ),
        )
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def pending_counts_grouped(
        self,
        *,
        base_key_values: dict[str, Any],
        group_column: str,
        group_values: list[Any],
    ) -> dict[str, int]:
        """Count pending samples grouped by one varying key dimension."""
        group_col = resolve_group_column(
            self._schema, self._tables.pending, group_column
        )
        predicates = []
        for key_column in self._schema.key_columns:
            if key_column.name == group_column:
                continue
            if key_column.name in base_key_values:
                predicates.append(
                    self._tables.pending.c[key_column.name]
                    == base_key_values[key_column.name]
                )
        if group_values:
            predicates.append(group_col.in_(group_values))
        predicates.append(
            self._tables.pending.c.status.in_(
                [PendingStatus.pending.value, PendingStatus.leased.value]
            )
        )
        stmt = (
            select(group_col.label("group_value"), func.count().label("cnt"))
            .where(*predicates)
            .group_by(group_col)
        )
        with self._runtime.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return {
            str(row["group_value"]): int(row["cnt"])
            for row in rows
            if int(row["cnt"]) > 0
        }

    def bump_pending_priority(
        self, *, key_values: dict[str, Any], priority: int
    ) -> int:
        """Increase priority for pending samples matching key dims."""
        validate_key_values(self._schema, key_values)
        stmt = (
            update(self._tables.pending)
            .where(
                key_filter_clause(self._schema, self._tables.pending, key_values),
                self._tables.pending.c.status == PendingStatus.pending.value,
            )
            .values(priority=func.greatest(self._tables.pending.c.priority, priority))
        )
        with self._runtime.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount or 0

    def bulk_load_pending(
        self, *, key_filter: dict[str, Any] | None = None
    ) -> list[PendingSample]:
        """Load in-flight pending samples, optionally filtered by partial key match."""
        stmt = select(*self._tables.pending_select_columns()).where(
            self._tables.pending.c.status.in_(
                [PendingStatus.pending.value, PendingStatus.leased.value]
            )
        )
        partial_filter = partial_key_filter_clause(
            self._schema, self._tables.pending, key_filter
        )
        if partial_filter is not None:
            stmt = stmt.where(partial_filter)
        stmt = stmt.order_by(
            self._tables.pending.c.priority.desc(),
            self._tables.pending.c.created_at.asc(),
        )

        with self._runtime.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [PendingSample.from_db_row(self._schema, dict(row)) for row in rows]

    def status_counts(
        self, *, key_filter: dict[str, Any] | None = None
    ) -> PendingStatusCounts:
        """Count pending rows by lifecycle status."""
        stmt = select(
            self._tables.pending.c.status,
            func.count().label("cnt"),
        ).group_by(self._tables.pending.c.status)
        partial_filter = partial_key_filter_clause(
            self._schema, self._tables.pending, key_filter
        )
        if partial_filter is not None:
            stmt = stmt.where(partial_filter)

        with self._runtime.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return PendingStatusCounts.from_rows(dict(row) for row in rows)

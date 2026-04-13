"""Pending sample lifecycle: insert, lease, promote, fail, release."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import (
    CTE,
    Select,
    Table,
    Text,
    and_,
    bindparam,
    delete,
    exists,
    func,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.db.schema import PoolSchema
from dr_llm.pool.db.sql_helpers import (
    insert_keyed_samples,
    key_filter_clause,
    partial_key_filter_clause,
    stream_select_rows,
    validate_key_values,
)
from dr_llm.pool.db.tables import PoolTables
from dr_llm.pool.models import InsertResult
from dr_llm.pool.key_filter import PoolKeyFilter
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import (
    IN_FLIGHT_STATUSES,
    PendingStatus,
    PendingStatusCounts,
)
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

    @property
    def _pending(self) -> Table:
        return self._tables.pending

    def insert(self, sample: PendingSample, *, ignore_conflicts: bool = True) -> bool:
        result = self.insert_many([sample], ignore_conflicts=ignore_conflicts)
        return result.inserted > 0

    def insert_many(
        self, samples: list[PendingSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        """Bulk insert pending samples in a single transaction."""
        if not samples:
            return InsertResult()
        for sample in samples:
            validate_key_values(self._schema, sample.key_values)
        inserted = insert_keyed_samples(
            self._runtime,
            self._pending,
            self._pending.c.pending_id,
            [sample.to_db_insert_row() for sample in samples],
            ignore_conflicts=ignore_conflicts,
        )
        return InsertResult(inserted=inserted, skipped=len(samples) - inserted)

    def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        key_filter: PoolKeyFilter | None = None,
    ) -> PendingSample | None:
        """Lease one pending sample for processing via FOR UPDATE SKIP LOCKED."""
        if lease_seconds <= 0:
            raise ValueError(
                f"lease_seconds must be a positive integer; got {lease_seconds}"
            )
        p = self._pending
        reusable = and_(
            p.c.status == PendingStatus.leased,
            p.c.lease_expires_at < func.now(),
        )
        predicates = [
            or_(
                p.c.status == PendingStatus.pending,
                reusable,
            )
        ]
        partial_filter = partial_key_filter_clause(self._schema, p, key_filter)
        if partial_filter is not None:
            predicates.append(partial_filter)

        locked = (
            select(p.c.pending_id)
            .where(*predicates)
            .order_by(
                p.c.priority.desc(),
                p.c.created_at.asc(),
            )
            .limit(1)
            .with_for_update(skip_locked=True)
            .cte("locked")
        )
        stmt = (
            update(p)
            .where(p.c.pending_id == locked.c.pending_id)
            .values(
                status=PendingStatus.leased,
                worker_id=worker_id,
                lease_expires_at=func.now()
                + func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds),
                attempt_count=p.c.attempt_count + 1,
            )
            .returning(*self._tables.pending_select_columns())
        )

        with self._runtime.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return PendingSample.from_db_row(self._schema, dict(row))

    def promote(
        self,
        *,
        pending_id: str,
        worker_id: str,
        payload: dict[str, Any] | None = None,
    ) -> PoolSample | None:
        """Promote a leased pending sample to finalized.

        Returns None if the pending_id doesn't exist, is not in 'leased' status,
        or is currently leased by a different worker (lease was lost and re-leased).

        Single round-trip via data-modifying CTE: locks the pending row with
        ``FOR UPDATE``, inserts the new sample (substituting the worker's
        payload when supplied), and marks the pending row promoted — all in
        one statement so workers don't pay 3x latency on the success path.
        """
        inserted, promoted = self._build_promote_ctes(
            pending_id=pending_id, worker_id=worker_id, payload=payload
        )
        stmt = self._build_promote_select(inserted=inserted, promoted=promoted)

        with self._runtime.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            logger.warning(
                "promote: pending_id=%s not promoted (missing, "
                "wrong worker, or sample insert conflict)",
                pending_id,
            )
            return None
        return PoolSample.from_db_row(self._schema, dict(row))

    def _build_promote_ctes(
        self,
        *,
        pending_id: str,
        worker_id: str,
        payload: dict[str, Any] | None,
    ) -> tuple[CTE, CTE]:
        """Build the CTE chain for promote: lock → insert sample → mark promoted."""
        p = self._pending
        key_names = self._schema.key_column_names

        locked = (
            select(
                p.c.pending_id,
                p.c.sample_idx,
                p.c.source_run_id,
                p.c.payload_json,
                p.c.metadata_json,
                *(p.c[name] for name in key_names),
            )
            .where(
                p.c.pending_id == pending_id,
                p.c.status == PendingStatus.leased,
                p.c.worker_id == worker_id,
            )
            .with_for_update()
            .cte("locked")
        )

        payload_expr: Any = (
            literal(to_jsonable_python(payload), type_=JSONB)
            if payload is not None
            else locked.c.payload_json
        )
        inserted = (
            pg_insert(self._tables.samples)
            .from_select(
                [
                    "sample_id",
                    "sample_idx",
                    "payload_json",
                    "source_run_id",
                    "metadata_json",
                    *key_names,
                ],
                select(
                    func.cast(func.gen_random_uuid(), Text),
                    locked.c.sample_idx,
                    payload_expr,
                    locked.c.source_run_id,
                    locked.c.metadata_json,
                    *(locked.c[name] for name in key_names),
                ),
            )
            .on_conflict_do_nothing()
            .returning(*self._tables.sample_select_columns())
            .cte("inserted")
        )

        promoted = (
            update(p)
            .where(
                p.c.pending_id == pending_id,
                exists(select(1).select_from(inserted)),
            )
            .values(status=PendingStatus.promoted)
            .returning(p.c.pending_id)
            .cte("promoted")
        )

        return inserted, promoted

    def _build_promote_select(self, *, inserted: CTE, promoted: CTE) -> Select[Any]:
        """Return the finalized sample row only when the promoted CTE fired."""
        return (
            select(
                *[inserted.c[col.name] for col in self._tables.sample_select_columns()]
            )
            .select_from(inserted)
            .where(exists(select(1).select_from(promoted)))
        )

    def fail(self, *, pending_id: str, worker_id: str, reason: str) -> bool:
        """Mark a leased pending sample as failed.

        Returns True if the row was marked failed, False if the lease was
        stale (sample missing, status changed, or re-leased by another worker).
        """
        metadata_patch = func.jsonb_build_object("fail_reason", reason)
        stmt = (
            update(self._pending)
            .where(
                self._pending.c.pending_id == pending_id,
                self._pending.c.worker_id == worker_id,
                self._pending.c.status == PendingStatus.leased,
            )
            .values(
                status=PendingStatus.failed,
                metadata_json=self._pending.c.metadata_json.op("||")(metadata_patch),
            )
        )
        with self._runtime.begin() as conn:
            result = conn.execute(stmt)
        return (result.rowcount or 0) > 0

    def release_lease(self, *, pending_id: str, worker_id: str) -> bool:
        """Release a lease, returning sample to pending status.

        Returns True if the lease was released, False if it was stale
        (sample missing, status changed, or re-leased by another worker).
        """
        stmt = (
            update(self._pending)
            .where(
                self._pending.c.pending_id == pending_id,
                self._pending.c.worker_id == worker_id,
                self._pending.c.status == PendingStatus.leased,
            )
            .values(
                status=PendingStatus.pending,
                worker_id=None,
                lease_expires_at=None,
            )
        )
        with self._runtime.begin() as conn:
            result = conn.execute(stmt)
        return (result.rowcount or 0) > 0

    def count_in_flight(self, *, key_values: dict[str, Any]) -> int:
        """Count in-flight pending samples (pending + leased) for given key dimensions."""
        validate_key_values(self._schema, key_values)
        stmt = select(func.count()).where(
            key_filter_clause(self._schema, self._pending, key_values),
            self._pending.c.status.in_(IN_FLIGHT_STATUSES),
        )
        with self._runtime.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def bump_priority(self, *, key_values: dict[str, Any], priority: int) -> int:
        """Increase priority for pending samples matching key dims."""
        validate_key_values(self._schema, key_values)
        stmt = (
            update(self._pending)
            .where(
                key_filter_clause(self._schema, self._pending, key_values),
                self._pending.c.status == PendingStatus.pending,
            )
            .values(priority=func.greatest(self._pending.c.priority, priority))
        )
        with self._runtime.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount or 0

    def shuffle_priorities(
        self,
        *,
        seed: int | None = None,
        key_filter: PoolKeyFilter | None = None,
        upper_bound: int = 1_000_000_000,
    ) -> int:
        """Assign each pending row a uniformly random priority.

        Use to break up insertion-order claim patterns so workers
        interleave across providers/cells instead of draining the front
        of the queue in cross-product order. Only touches rows whose
        ``status`` is ``pending`` — leased rows are currently being
        worked on and aren't reorderable from this side.

        Workers claim rows by ``ORDER BY priority DESC, created_at ASC``,
        so overwriting the priority column with random values reorders
        the queue immediately on the next ``claim`` call.

        Reproducibility: when ``seed`` is provided, pending_ids are
        listed in sorted order and each one is assigned a value drawn
        from ``random.Random(seed)``. The same set of pending_ids plus
        the same seed always produces the same priority mapping. We
        deliberately do **not** use Postgres's ``setseed`` + ``random()``
        because the planner is free to evaluate ``random()`` in any row
        order across runs, which produces a *different* random-value
        permutation across rows even when the seed is identical.

        Args:
            seed: When provided, the per-row assignment is reproducible.
            key_filter: Optional partial-key filter; only pending rows
                matching the filter are shuffled. None shuffles every
                pending row in the pool.
            upper_bound: Upper bound on the random priority range
                (exclusive). Default of 1e9 gives essentially no ties
                while staying well within int4 range.

        Returns:
            Number of pending rows that were updated.
        """
        if upper_bound < 1:
            raise ValueError(f"upper_bound must be >= 1, got {upper_bound}")

        predicates: list[Any] = [
            self._pending.c.status == PendingStatus.pending,
        ]
        partial_filter = partial_key_filter_clause(
            self._schema, self._pending, key_filter
        )
        if partial_filter is not None:
            predicates.append(partial_filter)

        select_pending_ids = (
            select(self._pending.c.pending_id)
            .where(*predicates)
            .order_by(self._pending.c.pending_id.asc())
        )
        update_one = (
            update(self._pending)
            .where(self._pending.c.pending_id == bindparam("b_pending_id"))
            .values(priority=bindparam("b_priority"))
        )

        rng = random.Random(seed)
        with self._runtime.begin() as conn:
            pending_ids = [row[0] for row in conn.execute(select_pending_ids)]
            if not pending_ids:
                return 0
            assignments = [
                {"b_pending_id": pid, "b_priority": rng.randrange(upper_bound)}
                for pid in pending_ids
            ]
            conn.execute(update_one, assignments)
        return len(pending_ids)

    def bulk_load(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        status: PendingStatus | Iterable[PendingStatus] | None = None,
    ) -> list[PendingSample]:
        """Load pending samples, optionally filtered by partial key match.

        Materializes the full result set; for very large queues prefer
        ``iter_pending`` to stream in chunks.
        """
        return list(self.iter_pending(key_filter=key_filter, status=status))

    def iter_pending(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        status: PendingStatus | Iterable[PendingStatus] | None = None,
        chunk_size: int = 1000,
    ) -> Iterator[PendingSample]:
        """Stream pending samples in chunks via server-side cursoring.

        ``status`` accepts a single ``PendingStatus`` or any iterable of them.
        When ``None`` (the default) only in-flight statuses (``pending`` and
        ``leased``) are returned, matching the historical worker-facing
        behavior; pass an explicit set to inspect terminal states like
        ``promoted`` or ``failed``.
        """
        if status is None:
            statuses: frozenset[PendingStatus] = IN_FLIGHT_STATUSES
        elif isinstance(status, PendingStatus):
            statuses = frozenset({status})
        else:
            statuses = frozenset(status)

        rows = stream_select_rows(
            self._runtime,
            self._schema,
            self._pending,
            self._tables.pending_select_columns(),
            base_predicates=[self._pending.c.status.in_(statuses)],
            order_by=[
                self._pending.c.priority.desc(),
                self._pending.c.created_at.asc(),
            ],
            key_filter=key_filter,
            chunk_size=chunk_size,
        )
        for row in rows:
            yield PendingSample.from_db_row(self._schema, row)

    def status_counts(
        self, *, key_filter: PoolKeyFilter | None = None
    ) -> PendingStatusCounts:
        """Count pending rows by lifecycle status."""
        stmt = select(
            self._pending.c.status,
            func.count().label("cnt"),
        ).group_by(self._pending.c.status)
        partial_filter = partial_key_filter_clause(
            self._schema, self._pending, key_filter
        )
        if partial_filter is not None:
            stmt = stmt.where(partial_filter)

        with self._runtime.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return PendingStatusCounts.from_rows(dict(row) for row in rows)

    def requeue_failed(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        """Reset failed rows matching ``key_filter`` back to pending."""
        metadata_json = self._pending.c.metadata_json - "fail_reason"
        predicates: list[Any] = [self._pending.c.status == PendingStatus.failed]
        partial_filter = partial_key_filter_clause(
            self._schema, self._pending, key_filter
        )
        if partial_filter is not None:
            predicates.append(partial_filter)

        stmt = (
            update(self._pending)
            .where(*predicates)
            .values(
                status=PendingStatus.pending,
                worker_id=None,
                lease_expires_at=None,
                attempt_count=0,
                metadata_json=metadata_json,
            )
        )
        with self._runtime.begin() as conn:
            result = conn.execute(stmt)
        return result.rowcount or 0

    def clear_pending(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        """Delete pending rows matching ``key_filter``."""
        predicates: list[Any] = [self._pending.c.status == PendingStatus.pending]
        partial_filter = partial_key_filter_clause(
            self._schema, self._pending, key_filter
        )
        if partial_filter is not None:
            predicates.append(partial_filter)

        stmt = delete(self._pending).where(*predicates)
        with self._runtime.begin() as conn:
            result = conn.execute(stmt)
        return result.rowcount or 0

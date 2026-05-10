"""Pool sample storage and leasing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from dr_llm.pool.db import (
    DbRuntime,
    PoolSchema,
    PoolTables,
    PoolTableType,
)
from dr_llm.pool.completion_filter import CompletionFilter
from dr_llm.pool.db.key_filter import PoolKeyFilter
from dr_llm.pool.insert_result import InsertResult
from dr_llm.pool.pool_progress import PoolProgress
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.db import catalog
from dr_llm.pool.store_ops import completion as completion_ops
from dr_llm.pool.store_ops import insert as insert_ops
from dr_llm.pool.store_ops import leasing as leasing_ops
from dr_llm.pool.store_ops import queries as query_ops
from dr_llm.pool.store_ops import requeue as requeue_ops
from dr_llm.pool.store_ops import request_update as request_update_ops
from dr_llm.pool.store_ops.request_update import RequestUpdate


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
        catalog.ensure_catalog_table(self._runtime)
        catalog.upsert_schema(self._runtime, self.schema)

    def insert_sample(
        self, sample: PoolSample, *, ignore_conflicts: bool = True
    ) -> bool:
        """Insert a single sample. Auto-assigns sample_idx if None."""
        return insert_ops.insert_sample(
            self._runtime,
            self.schema,
            self._tables,
            sample,
            ignore_conflicts=ignore_conflicts,
        )

    def insert_samples(
        self, samples: Iterable[PoolSample], *, ignore_conflicts: bool = True
    ) -> InsertResult:
        """Bulk insert samples."""
        return insert_ops.insert_samples(
            self._runtime,
            self.schema,
            self._tables,
            samples,
            ignore_conflicts=ignore_conflicts,
        )

    def complete_sample(
        self,
        *,
        sample_id: str,
        response: dict[str, Any],
        finish_reason: str | None,
        attempt_count: int,
    ) -> bool:
        """Fill in the response fields for one incomplete sample."""
        return completion_ops.complete_sample(
            self._runtime,
            self._tables[PoolTableType.SAMPLES],
            sample_id=sample_id,
            response=response,
            finish_reason=finish_reason,
            attempt_count=attempt_count,
        )

    def update_incomplete_request(
        self,
        *,
        sample_id: str,
        request: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update request_json for an incomplete sample. No-op if already complete."""
        return request_update_ops.update_incomplete_request(
            self._runtime,
            self._tables[PoolTableType.SAMPLES],
            sample_id=sample_id,
            request=request,
            metadata=metadata,
        )

    def update_incomplete_requests(
        self,
        rows: Iterable[RequestUpdate],
    ) -> int:
        """Batch-update request_json for incomplete samples. Returns updated count."""
        return request_update_ops.update_incomplete_requests(
            self._runtime,
            self._tables[PoolTableType.SAMPLES],
            rows=rows,
        )

    def requeue_errors(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        reset_attempt_count: bool = True,
    ) -> int:
        """Reset error rows to incomplete, clearing response and deleting leases."""
        return requeue_ops.requeue_errors(
            self._runtime,
            self.schema,
            self._tables,
            key_filter=key_filter,
            reset_attempt_count=reset_attempt_count,
        )

    def reset_samples(
        self,
        *,
        sample_ids: Sequence[str],
        reset_request: dict[str, Any] | None = None,
        reset_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Administratively reset specific samples to incomplete."""
        return requeue_ops.reset_samples(
            self._runtime,
            self.schema,
            self._tables,
            sample_ids=sample_ids,
            reset_request=reset_request,
            reset_metadata=reset_metadata,
        )

    def claim_lease(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        key_filter: PoolKeyFilter | None = None,
    ) -> PoolSample | None:
        """Lease one incomplete sample via ``FOR UPDATE SKIP LOCKED``."""
        return leasing_ops.claim_lease(
            self._runtime,
            self.schema,
            self._tables,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            key_filter=key_filter,
        )

    def release_lease(self, *, sample_id: str, worker_id: str) -> bool:
        """Release a lease owned by ``worker_id``."""
        return leasing_ops.release_lease(
            self._runtime,
            self._tables[PoolTableType.LEASES],
            sample_id=sample_id,
            worker_id=worker_id,
        )

    def expire_leases(self) -> int:
        """Delete expired lease rows and return the number removed."""
        return leasing_ops.expire_leases(
            self._runtime, self._tables[PoolTableType.LEASES]
        )

    def sample_count(self) -> int:
        """Return the total number of rows in the pool's samples table."""
        return query_ops.sample_count(
            self._runtime, self._tables[PoolTableType.SAMPLES]
        )

    def incomplete_count(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        """Return the count of samples without responses."""
        return query_ops.incomplete_count(
            self._runtime,
            self.schema,
            self._tables[PoolTableType.SAMPLES],
            key_filter=key_filter,
        )

    def complete_count(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        """Return the count of samples with responses."""
        return query_ops.complete_count(
            self._runtime,
            self.schema,
            self._tables[PoolTableType.SAMPLES],
            key_filter=key_filter,
        )

    def error_count(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        """Return the count of samples completed with finish_reason='error'."""
        return query_ops.error_count(
            self._runtime,
            self.schema,
            self._tables[PoolTableType.SAMPLES],
            key_filter=key_filter,
        )

    def leased_count(self, *, key_filter: PoolKeyFilter | None = None) -> int:
        """Return the count of samples with active (non-expired) leases."""
        return query_ops.leased_count(
            self._runtime,
            self.schema,
            self._tables[PoolTableType.SAMPLES],
            self._tables[PoolTableType.LEASES],
            key_filter=key_filter,
        )

    def progress(self, *, key_filter: PoolKeyFilter | None = None) -> PoolProgress:
        """Return a snapshot of pool completion state."""
        return query_ops.progress(
            self._runtime, self.schema, self._tables, key_filter=key_filter
        )

    def cell_depth(self, *, key_values: dict[str, Any]) -> int:
        """Count total samples for a specific cell."""
        return query_ops.cell_depth(
            self._runtime,
            self.schema,
            self._tables[PoolTableType.SAMPLES],
            key_values=key_values,
        )

    def bulk_load(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        completion: CompletionFilter = "all",
    ) -> list[PoolSample]:
        """Load all samples, optionally filtered by partial key match and completion state."""
        return query_ops.bulk_load(
            self._runtime,
            self.schema,
            self._tables,
            key_filter=key_filter,
            completion=completion,
        )

    def iter_samples(
        self,
        *,
        key_filter: PoolKeyFilter | None = None,
        completion: CompletionFilter = "all",
        chunk_size: int = 1000,
    ) -> Iterator[PoolSample]:
        """Stream samples in chunks via server-side cursoring."""
        return query_ops.iter_samples(
            self._runtime,
            self.schema,
            self._tables,
            key_filter=key_filter,
            completion=completion,
            chunk_size=chunk_size,
        )

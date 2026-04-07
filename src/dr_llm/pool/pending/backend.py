from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.logging import generation_log_context
from dr_llm.pool.db.sql_helpers import validate_key_filter
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.sample_store import PoolStore
from dr_llm.workers import ErrorDecision, WorkerBackend


class PoolPendingBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(default=0, ge=0)
    key_filter: dict[str, Any] | None = None


class PoolPendingBackendState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status_counts: PendingStatusCounts = Field(default_factory=PendingStatusCounts)
    key_filter: dict[str, Any] | None = None
    max_retries: int = 0


class PoolPendingBackend(
    WorkerBackend[PendingSample, dict[str, Any], PoolPendingBackendState]
):
    """Pool-specific backend for the generic worker runtime."""

    CLAIM_LIMIT: ClassVar[int] = 1

    def __init__(self, store: PoolStore, *, config: PoolPendingBackendConfig) -> None:
        self._store = store
        self._config = config
        self._store.init_schema()
        if config.key_filter is not None:
            validate_key_filter(self._store.schema, config.key_filter)

    def claim(self, *, worker_id: str, lease_seconds: int) -> list[PendingSample]:
        # Pool workers intentionally process exactly one item per loop to match
        # the generic worker contract and keep execution easy to reason about.
        claimed = self._store.pending.claim_pending(
            worker_id=worker_id,
            limit=self.CLAIM_LIMIT,
            lease_seconds=lease_seconds,
            key_filter=self._config.key_filter,
        )
        if len(claimed) > self.CLAIM_LIMIT:
            raise RuntimeError("PoolPendingBackend.claim returned more than one item")
        return claimed

    def complete(
        self,
        *,
        item: PendingSample,
        result: dict[str, Any],
        worker_id: str,
    ) -> None:
        del worker_id
        if not isinstance(result, dict):
            raise TypeError("process_fn must return a payload dict")
        promoted = self._store.pending.promote_pending(
            pending_id=item.pending_id,
            payload=result,
        )
        if promoted is None:
            raise RuntimeError(
                f"Failed to promote leased pending sample {item.pending_id}"
            )

    def handle_process_error(
        self,
        *,
        item: PendingSample,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision:
        if item.attempt_count <= self._config.max_retries:
            self._store.pending.release_pending_lease(
                pending_id=item.pending_id,
                worker_id=worker_id,
            )
            return ErrorDecision.retry

        self._store.pending.fail_pending(
            pending_id=item.pending_id,
            worker_id=worker_id,
            reason=_format_error_reason(exc),
        )
        return ErrorDecision.fail

    def snapshot(self) -> PoolPendingBackendState:
        return PoolPendingBackendState(
            status_counts=self._store.pending.status_counts(
                key_filter=self._config.key_filter
            ),
            key_filter=(
                None
                if self._config.key_filter is None
                else dict(self._config.key_filter)
            ),
            max_retries=self._config.max_retries,
        )

    def process_context(
        self,
        *,
        item: PendingSample,
        worker_id: str,
    ) -> AbstractContextManager[Any]:
        del item
        return generation_log_context(
            {
                "pool_name": self._store.schema.name,
                "worker_id": worker_id,
            }
        )


def _format_error_reason(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__

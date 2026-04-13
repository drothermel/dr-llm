from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.logging.events import generation_log_context
from dr_llm.pool.call_stats import CallStats
from dr_llm.pool.db.sql_helpers import validate_key_filter
from dr_llm.pool.pending.pending_sample import PendingSample
from dr_llm.pool.pending.pending_status import PendingStatusCounts
from dr_llm.pool.pool_store import PoolStore
from dr_llm.workers import ErrorDecision, WorkerBackend

logger = logging.getLogger(__name__)


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

    def __init__(self, store: PoolStore, *, config: PoolPendingBackendConfig) -> None:
        self._store = store
        self._config = config
        if config.key_filter is not None:
            validate_key_filter(self._store.schema, config.key_filter)

    def claim(self, *, worker_id: str, lease_seconds: int) -> PendingSample | None:
        return self._store.pending.claim(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            key_filter=self._config.key_filter,
        )

    def complete(
        self,
        *,
        item: PendingSample,
        result: dict[str, Any],
        worker_id: str,
    ) -> None:
        promoted = self._store.pending.promote(
            pending_id=item.pending_id,
            worker_id=worker_id,
            payload=result,
        )
        if promoted is None:
            raise RuntimeError(
                f"Failed to promote leased pending sample {item.pending_id}"
            )
        stats = CallStats.from_response(
            sample_id=promoted.sample_id,
            response=result,
            attempt_count=item.attempt_count,
        )
        self._store.insert_call_stats(stats)

    def handle_process_error(
        self,
        *,
        item: PendingSample,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision:
        if item.attempt_count <= self._config.max_retries:
            released = self._store.pending.release_lease(
                pending_id=item.pending_id,
                worker_id=worker_id,
            )
            if not released:
                logger.warning(
                    "release_lease no-op: pending_id=%s worker_id=%s "
                    "(lease was stale — sample missing or re-leased)",
                    item.pending_id,
                    worker_id,
                )
            return ErrorDecision.retry

        message = str(exc).strip()
        reason = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
        failed = self._store.pending.fail(
            pending_id=item.pending_id,
            worker_id=worker_id,
            reason=reason,
        )
        if not failed:
            logger.warning(
                "fail no-op: pending_id=%s worker_id=%s "
                "(lease was stale — sample missing or re-leased)",
                item.pending_id,
                worker_id,
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
        return generation_log_context(
            {
                "pool_name": self._store.schema.name,
                "worker_id": worker_id,
            }
        )

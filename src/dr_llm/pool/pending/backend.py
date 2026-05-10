from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.logging.events import generation_log_context
from dr_llm.pool.db.sql_helpers import validate_key_filter
from dr_llm.pool.key_filter import PoolKeyFilter
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.workers import ErrorDecision, WorkerBackend

logger = logging.getLogger(__name__)


class PoolPendingBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(default=0, ge=0)
    key_filter: PoolKeyFilter | None = None


class PoolPendingBackendState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incomplete: int = 0
    complete: int = 0
    key_filter: PoolKeyFilter | None = None
    max_retries: int = 0


class PoolPendingBackend(
    WorkerBackend[PoolSample, dict[str, Any], PoolPendingBackendState]
):
    """Pool-specific backend for the generic worker runtime."""

    def __init__(self, store: PoolStore, *, config: PoolPendingBackendConfig) -> None:
        self._store = store
        self._config = config
        self._attempt_counts: dict[str, int] = {}
        if config.key_filter is not None:
            validate_key_filter(self._store.schema, config.key_filter)

    def claim(self, *, worker_id: str, lease_seconds: int) -> PoolSample | None:
        sample = self._store.claim_lease(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            key_filter=self._config.key_filter,
        )
        if sample is not None:
            self._attempt_counts[sample.sample_id] = (
                self._attempt_counts.get(sample.sample_id, 0) + 1
            )
        return sample

    def complete(
        self,
        *,
        item: PoolSample,
        result: dict[str, Any],
        worker_id: str,
    ) -> None:
        attempt_count = self._attempt_count(item)
        completed = self._store.complete_sample(
            sample_id=item.sample_id,
            response=result,
            finish_reason=_finish_reason(result),
            attempt_count=attempt_count,
        )
        released = self._store.release_lease(
            sample_id=item.sample_id,
            worker_id=worker_id,
        )
        self._attempt_counts.pop(item.sample_id, None)
        if not completed:
            raise RuntimeError(
                f"Failed to complete leased pool sample {item.sample_id}"
            )
        if not released:
            logger.warning(
                "release_lease no-op: sample_id=%s worker_id=%s "
                "(lease was stale or re-leased)",
                item.sample_id,
                worker_id,
            )

    def handle_process_error(
        self,
        *,
        item: PoolSample,
        worker_id: str,
        exc: Exception,
    ) -> ErrorDecision:
        attempt_count = self._attempt_count(item)
        if attempt_count <= self._config.max_retries:
            released = self._store.release_lease(
                sample_id=item.sample_id,
                worker_id=worker_id,
            )
            if not released:
                logger.warning(
                    "release_lease no-op: sample_id=%s worker_id=%s "
                    "(lease was stale or re-leased)",
                    item.sample_id,
                    worker_id,
                )
            return ErrorDecision.retry

        message = str(exc).strip()
        reason = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
        completed = self._store.complete_sample(
            sample_id=item.sample_id,
            response={"error": reason},
            finish_reason="error",
            attempt_count=attempt_count,
        )
        released = self._store.release_lease(
            sample_id=item.sample_id,
            worker_id=worker_id,
        )
        self._attempt_counts.pop(item.sample_id, None)
        if not completed:
            logger.warning(
                "complete_sample no-op: sample_id=%s worker_id=%s "
                "(sample was already complete or missing)",
                item.sample_id,
                worker_id,
            )
        if not released:
            logger.warning(
                "release_lease no-op: sample_id=%s worker_id=%s "
                "(lease was stale or re-leased)",
                item.sample_id,
                worker_id,
            )
        return ErrorDecision.fail

    def snapshot(self) -> PoolPendingBackendState:
        return PoolPendingBackendState(
            incomplete=self._store.incomplete_count(key_filter=self._config.key_filter),
            complete=self._store.complete_count(key_filter=self._config.key_filter),
            key_filter=(
                None
                if self._config.key_filter is None
                else self._config.key_filter.model_copy(deep=True)
            ),
            max_retries=self._config.max_retries,
        )

    def process_context(
        self,
        *,
        item: PoolSample,
        worker_id: str,
    ) -> AbstractContextManager[Any]:
        return generation_log_context(
            {
                "pool_name": self._store.schema.name,
                "worker_id": worker_id,
            }
        )

    def _attempt_count(self, item: PoolSample) -> int:
        return self._attempt_counts.get(item.sample_id, max(item.attempt_count, 1))


def _finish_reason(result: dict[str, Any]) -> str | None:
    value = result.get("finish_reason")
    return value if isinstance(value, str) else None

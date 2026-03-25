"""Pool service with top-up orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from dr_llm.pool.errors import PoolTopupError
from dr_llm.pool.models import AcquireQuery, AcquireResult, PoolSample
from dr_llm.pool.store import PoolStore

logger = logging.getLogger(__name__)


class PoolService:
    """Top-up orchestration: acquire from pool, generate missing, re-acquire."""

    def __init__(
        self,
        store: PoolStore,
        *,
        pending_poll_interval_s: float = 2.0,
        pending_poll_timeout_s: float = 120.0,
        pending_priority_bump: int = 100,
    ) -> None:
        self._store = store
        self._pending_poll_interval_s = pending_poll_interval_s
        self._pending_poll_timeout_s = pending_poll_timeout_s
        self._pending_priority_bump = pending_priority_bump

    @property
    def store(self) -> PoolStore:
        return self._store

    def acquire_or_generate(
        self,
        query: AcquireQuery,
        *,
        generator_fn: Callable[[dict[str, Any], int], list[PoolSample]],
    ) -> AcquireResult:
        """
        Acquire samples with automatic top-up on deficit.

        1. Try to acquire from pool
        2. If deficit: wait for relevant pending samples to be promoted
        3. If still deficit: invoke generator_fn(key_values, deficit)
        4. Insert generated samples
        5. Re-acquire and return combined result
        """
        result = self._store.acquire(query)
        deficit = result.deficit(query.n)
        if deficit <= 0:
            return result

        # Wait for pending samples that might be in-flight
        pending_count = self._store.pending_counts(key_values=query.key_values)
        if pending_count > 0:
            self._store.bump_pending_priority(
                key_values=query.key_values,
                priority=self._pending_priority_bump,
            )
            result = self._wait_and_reacquire(query, result)
            deficit = result.deficit(query.n)
            if deficit <= 0:
                return result

        # Generate missing samples via caller-provided function
        try:
            generated = self._generate_topup(query.key_values, deficit, generator_fn)
        except Exception as exc:
            raise PoolTopupError(
                f"Top-up generation failed for {query.key_values}: {exc}"
            ) from exc

        if generated:
            insert_result = self._store.insert_samples(generated, ignore_conflicts=True)
            logger.info(
                "Top-up inserted %d samples (skipped %d, failed %d)",
                insert_result.inserted,
                insert_result.skipped,
                insert_result.failed,
            )

            # Re-acquire to pick up the new samples
            reacquire_query = AcquireQuery(
                run_id=query.run_id,
                request_id=query.request_id,
                key_values=query.key_values,
                n=deficit,
                consumer_tag=query.consumer_tag,
            )
            extra = self._store.acquire(reacquire_query)
            return AcquireResult(
                samples=result.samples + extra.samples,
                claimed=result.claimed + extra.claimed,
            )

        return result

    def _wait_and_reacquire(
        self, query: AcquireQuery, partial: AcquireResult
    ) -> AcquireResult:
        """Poll for pending samples to be promoted, then re-acquire.

        Note: blocks the calling thread with time.sleep. Use from sync contexts
        only; an async variant would be needed for asyncio callers.
        """
        deadline = time.monotonic() + self._pending_poll_timeout_s
        deficit = partial.deficit(query.n)

        while time.monotonic() < deadline and deficit > 0:
            time.sleep(self._pending_poll_interval_s)
            pending = self._store.pending_counts(key_values=query.key_values)
            if pending == 0:
                break

            reacquire_query = AcquireQuery(
                run_id=query.run_id,
                request_id=query.request_id,
                key_values=query.key_values,
                n=deficit,
                consumer_tag=query.consumer_tag,
            )
            extra = self._store.acquire(reacquire_query)
            if extra.claimed > 0:
                partial = AcquireResult(
                    samples=partial.samples + extra.samples,
                    claimed=partial.claimed + extra.claimed,
                )
                deficit = partial.deficit(query.n)

        return partial

    def _generate_topup(
        self,
        key_values: dict[str, Any],
        deficit: int,
        generator_fn: Callable[[dict[str, Any], int], list[PoolSample]],
    ) -> list[PoolSample]:
        """Generate top-up samples via the caller-provided function."""
        logger.info("Generating %d top-up samples for %s", deficit, key_values)
        return generator_fn(key_values, deficit)

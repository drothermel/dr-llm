"""Pool service with top-up orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from dr_llm.pool.errors import PoolTopupError
from dr_llm.pool.models import AcquireQuery, AcquireResult
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore

logger = logging.getLogger(__name__)


class PoolService:
    """Top-up orchestration: acquire from pool, generate missing, re-acquire."""

    _INITIAL_POLL_INTERVAL_S = 0.05

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
        if self._is_satisfied(result, query):
            return result

        result = self._wait_and_reacquire(query, result)
        if self._is_satisfied(result, query):
            return result

        deficit = result.deficit(query.n)
        try:
            logger.info(
                "Generating %d top-up samples for %s", deficit, query.key_values
            )
            generated = generator_fn(query.key_values, deficit)
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
            reacquired = self._store.acquire(query.model_copy(update={"n": deficit}))
            return AcquireResult(samples=result.samples + reacquired.samples)

        return result

    def _wait_and_reacquire(
        self, query: AcquireQuery, acquired_so_far: AcquireResult
    ) -> AcquireResult:
        """Bump pending priority then poll for promoted samples to be acquired.

        ``bump_priority`` returns 0 when no pending rows match — that
        signal lets us short-circuit the wait without a separate count query.
        Polls back off exponentially from 50ms toward ``pending_poll_interval_s``
        so the first promoted sample is observed within ~50ms instead of being
        pinned to the long ceiling.
        """
        bumped = self._store.pending.bump_priority(
            key_values=query.key_values,
            priority=self._pending_priority_bump,
        )
        if bumped == 0:
            return acquired_so_far

        deadline = time.monotonic() + self._pending_poll_timeout_s
        sleep_s = self._INITIAL_POLL_INTERVAL_S

        while not self._is_satisfied(acquired_so_far, query):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(sleep_s, remaining))
            sleep_s = min(sleep_s * 2.0, self._pending_poll_interval_s)

            deficit = acquired_so_far.deficit(query.n)
            reacquired = self._store.acquire(query.model_copy(update={"n": deficit}))
            if reacquired.claimed > 0:
                acquired_so_far = AcquireResult(
                    samples=acquired_so_far.samples + reacquired.samples,
                )
                continue

            if self._store.pending.count_in_flight(key_values=query.key_values) == 0:
                break

        return acquired_so_far

    @staticmethod
    def _is_satisfied(result: AcquireResult, query: AcquireQuery) -> bool:
        """True when ``result`` already holds enough samples for ``query``."""
        return result.deficit(query.n) <= 0

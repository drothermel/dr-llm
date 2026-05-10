"""Pool service with top-up orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore
from dr_llm.sampling.acquisition import AcquireQuery, AcquireResult
from dr_llm.sampling.errors import PoolTopupError
from dr_llm.sampling.sampling_store import SamplingStore

logger = logging.getLogger(__name__)


class PoolService:
    """Top-up orchestration: acquire from pool, generate missing, re-acquire."""

    def __init__(
        self,
        store: PoolStore,
    ) -> None:
        self._store = store
        self._sampling = SamplingStore.from_pool_store(store)

    @property
    def store(self) -> PoolStore:
        return self._store

    @property
    def sampling(self) -> SamplingStore:
        return self._sampling

    def acquire_or_generate(
        self,
        query: AcquireQuery,
        *,
        consumer_id: str,
        generator_fn: Callable[[dict[str, Any], int], list[PoolSample]],
    ) -> AcquireResult:
        """
        Acquire completed samples with automatic top-up on deficit.

        1. Try to acquire completed samples from the pool
        2. If still deficit: invoke generator_fn(key_values, deficit)
        3. Insert generated samples
        4. Re-acquire and return combined result

        ``generator_fn`` should return completed ``PoolSample`` rows, because
        sampling deliberately does not claim unfilled samples.
        """
        result = self._sampling.acquire(query, consumer_id)
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

            reacquired = self._sampling.acquire(
                query.model_copy(update={"n": deficit}), consumer_id
            )
            return AcquireResult(samples=result.samples + reacquired.samples)

        return result

    @staticmethod
    def _is_satisfied(result: AcquireResult, query: AcquireQuery) -> bool:
        """True when ``result`` already holds enough samples for ``query``."""
        return result.deficit(query.n) <= 0

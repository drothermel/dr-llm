"""Claim strategies for distributing work across key values."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dr_llm.pool.db.key_filter import PoolKeyEqClause, PoolKeyFilter
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.pool.pool_store import PoolStore


class ClaimOrder(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["created_at", "random"] = "created_at"
    seed: int | None = None


def _merge_filter(
    base: PoolKeyFilter | None, key: str, value: str
) -> PoolKeyFilter:
    clauses = dict(base.root) if base is not None else {}
    clauses[key] = PoolKeyEqClause(value=value)
    return PoolKeyFilter(clauses)


class RoundRobinClaimer:
    def __init__(
        self,
        store: PoolStore,
        *,
        round_robin_key: str,
        round_robin_values: Sequence[str],
        base_key_filter: PoolKeyFilter | None = None,
        order: ClaimOrder | None = None,
    ) -> None:
        self._store = store
        self._round_robin_key = round_robin_key
        self._base_key_filter = base_key_filter
        self._cursor = 0

        values = list(round_robin_values)
        effective_order = order or ClaimOrder()
        if effective_order.kind == "random":
            rng = random.Random(effective_order.seed)
            rng.shuffle(values)
        self._values = values

    def claim(
        self, *, worker_id: str, lease_seconds: int
    ) -> PoolSample | None:
        if not self._values:
            return None

        n = len(self._values)
        for _ in range(n):
            value = self._values[self._cursor % n]
            self._cursor += 1
            key_filter = _merge_filter(
                self._base_key_filter, self._round_robin_key, value
            )
            sample = self._store.claim_lease(
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                key_filter=key_filter,
            )
            if sample is not None:
                return sample
        return None

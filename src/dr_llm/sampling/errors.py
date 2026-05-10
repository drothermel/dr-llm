from __future__ import annotations

from dr_llm.pool.errors import PoolError


class PoolAcquireError(PoolError):
    """Failed to acquire samples from pool."""


class PoolTopupError(PoolError):
    """Top-up generation or persistence failed."""

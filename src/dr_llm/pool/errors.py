from __future__ import annotations

from dr_llm.errors import LlmPoolError


class PoolError(LlmPoolError):
    """Base error for pool operations."""


class PoolSchemaError(PoolError):
    """Invalid pool schema declaration."""


class PoolAcquireError(PoolError):
    """Failed to acquire samples from pool."""


class PoolTopupError(PoolError):
    """Top-up generation or persistence failed."""

from __future__ import annotations

from dr_llm.errors import LlmPoolError


class PoolError(LlmPoolError):
    """Base error for pool operations."""


class PoolSchemaError(PoolError):
    """Invalid pool schema declaration."""


class PoolNotFoundError(PoolError):
    """Pool tables do not exist for the requested pool name."""

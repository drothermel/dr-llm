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


class PoolNotFoundError(PoolError):
    """Pool tables do not exist for the requested pool name."""


class PoolSchemaNotPersistedError(PoolError):
    """Pool tables exist but the schema metadata row is missing.

    Raised by :class:`dr_llm.PoolReader` when opening a pool created before
    ``ensure_schema`` started persisting the pool's :class:`PoolSchema` into
    the metadata table. The fix is to construct the reader explicitly via
    :meth:`PoolReader.from_runtime` (which accepts a schema directly), or to
    re-run :meth:`PoolStore.ensure_schema` to backfill the metadata row.
    """

"""Pool schema helpers for PoolBackend."""

from __future__ import annotations

from dr_llm.pool.db.schema import KeyColumn, PoolSchema

BACKENDS_KEY_COLUMN = "request_fingerprint"


def backends_pool_schema(pool_name: str) -> PoolSchema:
    """Build the canonical single-key schema used by PoolBackend."""
    return PoolSchema(
        name=pool_name,
        key_columns=[KeyColumn(name=BACKENDS_KEY_COLUMN)],
    )

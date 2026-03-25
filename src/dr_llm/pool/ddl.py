"""DDL generation from PoolSchema declarations.

Produces idempotent CREATE TABLE IF NOT EXISTS + CREATE INDEX IF NOT EXISTS
statements for the samples, claims, pending, and metadata tables.
"""

from __future__ import annotations

from dr_llm.pool.schema import ColumnType, PoolSchema


_PG_TYPE_MAP: dict[ColumnType, str] = {
    ColumnType.text: "TEXT",
    ColumnType.integer: "INTEGER",
    ColumnType.boolean: "BOOLEAN",
    ColumnType.float_: "DOUBLE PRECISION",
}


def generate_ddl(schema: PoolSchema) -> str:
    """Generate full DDL for all pool tables and indexes."""
    parts = [
        _samples_ddl(schema),
        _claims_ddl(schema),
        _pending_ddl(schema),
        _metadata_ddl(schema),
    ]
    return "\n\n".join(parts)


def _key_columns_ddl(schema: PoolSchema) -> str:
    """Generate column definitions for key columns."""
    lines: list[str] = []
    for kc in schema.key_columns:
        pg_type = _PG_TYPE_MAP[kc.type]
        lines.append(f"    {kc.name} {pg_type} NOT NULL")
    return ",\n".join(lines)


def _key_column_list(schema: PoolSchema) -> str:
    """Comma-separated key column names."""
    return ", ".join(kc.name for kc in schema.key_columns)


def _samples_ddl(schema: PoolSchema) -> str:
    tbl = schema.samples_table
    key_cols = _key_columns_ddl(schema)
    key_list = _key_column_list(schema)

    return f"""CREATE TABLE IF NOT EXISTS {tbl} (
    sample_id TEXT PRIMARY KEY,
{key_cols},
    sample_idx INTEGER NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    source_run_id TEXT,
    call_id TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_{tbl}_cell
    ON {tbl} ({key_list}, sample_idx);

CREATE INDEX IF NOT EXISTS idx_{tbl}_key
    ON {tbl} ({key_list});"""


def _claims_ddl(schema: PoolSchema) -> str:
    tbl = schema.claims_table

    return f"""CREATE TABLE IF NOT EXISTS {tbl} (
    claim_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    consumer_tag TEXT NOT NULL DEFAULT '',
    sample_id TEXT NOT NULL,
    claim_idx INTEGER NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_{tbl}_run_sample
    ON {tbl} (run_id, sample_id);

CREATE INDEX IF NOT EXISTS idx_{tbl}_run
    ON {tbl} (run_id);"""


def _pending_ddl(schema: PoolSchema) -> str:
    tbl = schema.pending_table
    key_cols = _key_columns_ddl(schema)
    key_list = _key_column_list(schema)

    return f"""CREATE TABLE IF NOT EXISTS {tbl} (
    pending_id TEXT PRIMARY KEY,
{key_cols},
    sample_idx INTEGER NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    source_run_id TEXT,
    call_id TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    worker_id TEXT,
    lease_expires_at TIMESTAMPTZ,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_{tbl}_cell
    ON {tbl} ({key_list}, sample_idx);

CREATE INDEX IF NOT EXISTS idx_{tbl}_status_priority
    ON {tbl} (status, priority DESC, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_{tbl}_key
    ON {tbl} ({key_list});"""


def _metadata_ddl(schema: PoolSchema) -> str:
    tbl = schema.metadata_table

    return f"""CREATE TABLE IF NOT EXISTS {tbl} (
    pool_name TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_{tbl}_pool_key
    ON {tbl} (pool_name, key);"""

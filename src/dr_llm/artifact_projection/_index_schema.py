from __future__ import annotations

import sqlite3


def initialize_artifact_index_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.commit()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifact_references (
    artifact_id TEXT PRIMARY KEY,
    projection_version TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    source_idempotency_key TEXT NOT NULL,
    payload_role TEXT NOT NULL,
    source_object_key TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    reference_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifact_references_source_event
ON artifact_references(source_event_id);

CREATE INDEX IF NOT EXISTS idx_artifact_references_shard
ON artifact_references(shard_id);

CREATE TABLE IF NOT EXISTS open_shards (
    shard_id TEXT PRIMARY KEY,
    projection_version TEXT NOT NULL,
    shard_uri TEXT NOT NULL,
    writer_session TEXT NOT NULL,
    opened_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_open_shards_writer_session
ON open_shards(writer_session);

CREATE TABLE IF NOT EXISTS open_artifact_references (
    artifact_id TEXT PRIMARY KEY,
    projection_version TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    source_idempotency_key TEXT NOT NULL,
    payload_role TEXT NOT NULL,
    source_object_key TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    writer_session TEXT NOT NULL,
    reference_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(shard_id) REFERENCES open_shards(shard_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_open_artifact_references_source_event
ON open_artifact_references(source_event_id);

CREATE INDEX IF NOT EXISTS idx_open_artifact_references_shard
ON open_artifact_references(shard_id);

CREATE TABLE IF NOT EXISTS shards (
    shard_id TEXT PRIMARY KEY,
    projection_version TEXT NOT NULL,
    shard_uri TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    finalized_at TEXT
);

CREATE TABLE IF NOT EXISTS projection_checkpoints (
    projection_version TEXT NOT NULL,
    durable_consumer TEXT NOT NULL,
    stream_sequence INTEGER NOT NULL,
    event_id TEXT,
    checkpoint_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (projection_version, durable_consumer)
);

CREATE TABLE IF NOT EXISTS projection_errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    projection_version TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    source_idempotency_key TEXT NOT NULL,
    payload_role TEXT,
    source_object_key TEXT,
    error_kind TEXT NOT NULL,
    error_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


__all__ = ["SCHEMA_SQL", "initialize_artifact_index_schema"]

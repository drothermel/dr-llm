from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

from dr_llm.artifact_projection.models import (
    ArtifactReference,
    ShardManifest,
)


class ArtifactShardRows:
    def __init__(self, connection: Callable[[], sqlite3.Connection]) -> None:
        self._connection = connection

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection()

    def insert_shard(self, manifest: ShardManifest) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO shards (
                shard_id, projection_version, shard_uri, manifest_json,
                finalized_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                manifest.shard_id,
                manifest.projection_version,
                manifest.shard_uri,
                manifest.model_dump_json(),
                (
                    manifest.finalized_at.isoformat()
                    if manifest.finalized_at is not None
                    else None
                ),
            ),
        )

    def insert_open_shard(
        self, reference: ArtifactReference, *, writer_session: str
    ) -> None:
        opened_at = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            INSERT INTO open_shards (
                shard_id, projection_version, shard_uri, writer_session,
                opened_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(shard_id) DO NOTHING
            """,
            (
                reference.shard_id,
                reference.projection_version,
                reference.shard_uri,
                writer_session,
                opened_at,
            ),
        )

    def clear_open_shard(self, shard_id: str) -> None:
        self.connection.execute(
            """
            DELETE FROM open_artifact_references
            WHERE shard_id = ?
            """,
            (shard_id,),
        )
        self.connection.execute(
            "DELETE FROM open_shards WHERE shard_id = ?",
            (shard_id,),
        )

    def clear_rebuildable_rows(self) -> None:
        self.connection.execute("DELETE FROM open_artifact_references")
        self.connection.execute("DELETE FROM open_shards")
        self.connection.execute("DELETE FROM artifact_references")
        self.connection.execute("DELETE FROM shards")


__all__ = ["ArtifactShardRows"]

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from dr_llm.artifact_projection._index_progress import ArtifactProgressRows
from dr_llm.artifact_projection.models import ArtifactIndexSummary


COUNT_TABLE_NAMES = frozenset(
    {
        "artifact_references",
        "open_artifact_references",
        "shards",
        "open_shards",
        "projection_errors",
    }
)


class ArtifactSummaryQueries:
    def __init__(
        self,
        connection: Callable[[], sqlite3.Connection],
        *,
        progress: ArtifactProgressRows,
    ) -> None:
        self._connection = connection
        self.progress = progress

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection()

    def summary(
        self, *, projection_version: str, durable_consumer: str
    ) -> ArtifactIndexSummary:
        artifact_count = self.count("artifact_references")
        open_artifact_count = self.count("open_artifact_references")
        shard_count = self.count("shards")
        open_shard_count = self.count("open_shards")
        error_count = self.count("projection_errors")
        checkpoint = self.progress.latest_checkpoint(
            projection_version=projection_version,
            durable_consumer=durable_consumer,
        )
        return ArtifactIndexSummary(
            artifact_count=artifact_count,
            open_artifact_count=open_artifact_count,
            shard_count=shard_count,
            open_shard_count=open_shard_count,
            error_count=error_count,
            checkpoint=checkpoint,
        )

    def count(self, table_name: str) -> int:
        if table_name not in COUNT_TABLE_NAMES:
            raise ValueError(f"unknown artifact index table {table_name!r}")
        row = self.connection.execute(
            f"SELECT COUNT(*) AS row_count FROM {table_name}"
        ).fetchone()
        return int(row["row_count"])


__all__ = ["ArtifactSummaryQueries"]

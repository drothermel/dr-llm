from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType

from dr_llm.artifact_projection._index_progress import ArtifactProgressRows
from dr_llm.artifact_projection._index_references import (
    ArtifactIndexConflictError,
    ArtifactReferenceRows,
)
from dr_llm.artifact_projection._index_schema import (
    initialize_artifact_index_schema,
)
from dr_llm.artifact_projection._index_shards import ArtifactShardRows
from dr_llm.artifact_projection._index_summary import ArtifactSummaryQueries
from dr_llm.artifact_projection._manifest_io import (
    dump_json_line,
    load_manifest_references,
    load_shard_manifest,
)
from dr_llm.artifact_projection.models import (
    ArtifactIndexSummary,
    ArtifactReference,
    ProjectionCheckpoint,
    ProjectionError,
    ShardManifest,
)


class ArtifactIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None
        self._references = ArtifactReferenceRows(lambda: self.connection)
        self._shards = ArtifactShardRows(lambda: self.connection)
        self._progress = ArtifactProgressRows(lambda: self.connection)
        self._summaries = ArtifactSummaryQueries(
            lambda: self.connection,
            progress=self._progress,
        )

    def __enter__(self) -> ArtifactIndex:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def connect(self) -> None:
        if self._connection is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        self._connection = connection

    def close(self) -> None:
        if self._connection is None:
            return
        self._connection.close()
        self._connection = None

    def initialize(self) -> None:
        initialize_artifact_index_schema(self.connection)

    def insert_reference(self, reference: ArtifactReference) -> bool:
        existing = self.get_reference(reference.artifact_id)
        if existing is not None:
            self._references.raise_if_reference_conflicts(existing, reference)
            return False
        with self.transaction():
            self._references.insert_finalized_reference(reference)
        return True

    def insert_open_reference(
        self, reference: ArtifactReference, *, writer_session: str
    ) -> bool:
        existing = self.get_reference(reference.artifact_id)
        if existing is not None:
            self._references.raise_if_reference_conflicts(existing, reference)
            return False
        with self.transaction():
            self._shards.insert_open_shard(
                reference, writer_session=writer_session
            )
            return self._references.insert_open_reference(
                reference, writer_session=writer_session
            )

    def finalize_shard_references(
        self, manifest: ShardManifest, references: list[ArtifactReference]
    ) -> None:
        with self.transaction():
            self._shards.insert_shard(manifest)
            for reference in references:
                existing = self.get_finalized_reference(reference.artifact_id)
                if existing is not None:
                    self._references.raise_if_reference_conflicts(
                        existing, reference
                    )
                    continue
                self._references.insert_finalized_reference(reference)
            self._shards.clear_open_shard(manifest.shard_id)

    def get_reference(self, artifact_id: str) -> ArtifactReference | None:
        return self._references.get_reference(artifact_id)

    def get_finalized_reference(
        self, artifact_id: str
    ) -> ArtifactReference | None:
        return self._references.get_finalized_reference(artifact_id)

    def get_open_reference(self, artifact_id: str) -> ArtifactReference | None:
        return self._references.get_open_reference(artifact_id)

    def list_references(self) -> list[ArtifactReference]:
        return self._references.list_references()

    def insert_shard(self, manifest: ShardManifest) -> None:
        with self.transaction():
            self._shards.insert_shard(manifest)

    def record_checkpoint(self, checkpoint: ProjectionCheckpoint) -> None:
        self._progress.record_checkpoint(checkpoint)

    def latest_checkpoint(
        self, *, projection_version: str, durable_consumer: str
    ) -> ProjectionCheckpoint | None:
        return self._progress.latest_checkpoint(
            projection_version=projection_version,
            durable_consumer=durable_consumer,
        )

    def record_error(self, error: ProjectionError) -> None:
        self._progress.record_error(error)

    def record_reference_conflict(
        self, reference: ArtifactReference
    ) -> ProjectionError:
        return self._progress.record_reference_conflict(reference)

    def summary(
        self, *, projection_version: str, durable_consumer: str
    ) -> ArtifactIndexSummary:
        return self._summaries.summary(
            projection_version=projection_version,
            durable_consumer=durable_consumer,
        )

    def clear_rebuildable_rows(self) -> None:
        with self.transaction():
            self._shards.clear_rebuildable_rows()

    def transaction(self) -> _Transaction:
        return _Transaction(self.connection)

    @property
    def connection(self) -> sqlite3.Connection:
        self.connect()
        if self._connection is None:
            raise RuntimeError("artifact index is not connected")
        return self._connection


class _Transaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __enter__(self) -> None:
        self.connection.execute("BEGIN")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        if exc_type is None:
            self.connection.commit()
            return
        self.connection.rollback()


__all__ = [
    "ArtifactIndex",
    "ArtifactIndexConflictError",
    "dump_json_line",
    "load_manifest_references",
    "load_shard_manifest",
]

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from dr_llm.artifact_projection.models import (
    ArtifactIndexSummary,
    ArtifactReference,
    ArtifactSourceRef,
    ProjectionCheckpoint,
    ProjectionError,
    ProjectionErrorKind,
    ShardManifest,
)


COUNT_TABLE_NAMES = frozenset(
    {
        "artifact_references",
        "open_artifact_references",
        "shards",
        "open_shards",
        "projection_errors",
    }
)


class ArtifactIndexConflictError(RuntimeError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(f"Artifact {artifact_id!r} conflicts with index row")
        self.artifact_id = artifact_id


class ArtifactIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None

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
        connection = self.connection
        connection.executescript(SCHEMA_SQL)
        connection.commit()

    def insert_reference(self, reference: ArtifactReference) -> bool:
        existing = self.get_reference(reference.artifact_id)
        if existing is not None:
            self._raise_if_reference_conflicts(existing, reference)
            return False
        with self.transaction():
            self._insert_finalized_reference(reference)
        return True

    def insert_open_reference(
        self, reference: ArtifactReference, *, writer_session: str
    ) -> bool:
        existing = self.get_reference(reference.artifact_id)
        if existing is not None:
            self._raise_if_reference_conflicts(existing, reference)
            return False
        with self.transaction():
            self._insert_open_shard(reference, writer_session=writer_session)
            cursor = self.connection.execute(
                """
                INSERT INTO open_artifact_references (
                    artifact_id, projection_version, source_event_id,
                    source_idempotency_key, payload_role, source_object_key,
                    source_sha256, shard_id, writer_session, reference_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO NOTHING
                """,
                (
                    reference.artifact_id,
                    reference.projection_version,
                    *_source_ref_index_values(reference.source_ref),
                    reference.shard_id,
                    writer_session,
                    reference.model_dump_json(),
                    reference.created_at.isoformat(),
                ),
            )
            if cursor.rowcount == 1:
                return True
            existing = self.get_reference(reference.artifact_id)
            if existing is None:
                raise RuntimeError(
                    f"Artifact {reference.artifact_id!r} was not inserted"
                )
            self._raise_if_reference_conflicts(existing, reference)
            return False

    def finalize_shard_references(
        self, manifest: ShardManifest, references: list[ArtifactReference]
    ) -> None:
        with self.transaction():
            self._insert_shard(manifest)
            for reference in references:
                existing = self.get_finalized_reference(reference.artifact_id)
                if existing is not None:
                    self._raise_if_reference_conflicts(existing, reference)
                    continue
                self._insert_finalized_reference(reference)
            self.connection.execute(
                """
                DELETE FROM open_artifact_references
                WHERE shard_id = ?
                """,
                (manifest.shard_id,),
            )
            self.connection.execute(
                "DELETE FROM open_shards WHERE shard_id = ?",
                (manifest.shard_id,),
            )

    def _insert_finalized_reference(
        self, reference: ArtifactReference
    ) -> None:
        try:
            self.connection.execute(
                """
                INSERT INTO artifact_references (
                    artifact_id, projection_version, source_event_id,
                    source_idempotency_key, payload_role, source_object_key,
                    source_sha256, shard_id, reference_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reference.artifact_id,
                    reference.projection_version,
                    *_source_ref_index_values(reference.source_ref),
                    reference.shard_id,
                    reference.model_dump_json(),
                    reference.created_at.isoformat(),
                ),
            )
        except sqlite3.IntegrityError:
            existing = self.get_finalized_reference(reference.artifact_id)
            if existing is None:
                raise
            self._raise_if_reference_conflicts(existing, reference)

    def get_reference(self, artifact_id: str) -> ArtifactReference | None:
        finalized = self.get_finalized_reference(artifact_id)
        if finalized is not None:
            return finalized
        return self.get_open_reference(artifact_id)

    def get_finalized_reference(
        self, artifact_id: str
    ) -> ArtifactReference | None:
        row = self.connection.execute(
            """
            SELECT reference_json
            FROM artifact_references
            WHERE artifact_id = ?
            """,
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return ArtifactReference.model_validate_json(row["reference_json"])

    def get_open_reference(self, artifact_id: str) -> ArtifactReference | None:
        row = self.connection.execute(
            """
            SELECT reference_json
            FROM open_artifact_references
            WHERE artifact_id = ?
            """,
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return ArtifactReference.model_validate_json(row["reference_json"])

    def list_references(self) -> list[ArtifactReference]:
        rows = self.connection.execute(
            """
            SELECT reference_json
            FROM artifact_references
            ORDER BY artifact_id
            """
        ).fetchall()
        return [
            ArtifactReference.model_validate_json(row["reference_json"])
            for row in rows
        ]

    def insert_shard(self, manifest: ShardManifest) -> None:
        with self.transaction():
            self._insert_shard(manifest)

    def _insert_shard(self, manifest: ShardManifest) -> None:
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

    def _insert_open_shard(
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

    def record_checkpoint(self, checkpoint: ProjectionCheckpoint) -> None:
        self.connection.execute(
            """
            INSERT INTO projection_checkpoints (
                projection_version, durable_consumer, stream_sequence,
                event_id, checkpoint_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(projection_version, durable_consumer)
            DO UPDATE SET
                stream_sequence = excluded.stream_sequence,
                event_id = excluded.event_id,
                checkpoint_json = excluded.checkpoint_json,
                updated_at = excluded.updated_at
            """,
            (
                checkpoint.projection_version,
                checkpoint.durable_consumer,
                checkpoint.stream_sequence,
                checkpoint.event_id,
                checkpoint.model_dump_json(),
                checkpoint.updated_at.isoformat(),
            ),
        )
        self.connection.commit()

    def latest_checkpoint(
        self, *, projection_version: str, durable_consumer: str
    ) -> ProjectionCheckpoint | None:
        row = self.connection.execute(
            """
            SELECT checkpoint_json
            FROM projection_checkpoints
            WHERE projection_version = ? AND durable_consumer = ?
            """,
            (projection_version, durable_consumer),
        ).fetchone()
        if row is None:
            return None
        return ProjectionCheckpoint.model_validate_json(row["checkpoint_json"])

    def record_error(self, error: ProjectionError) -> None:
        self.connection.execute(
            """
            INSERT INTO projection_errors (
                projection_version, source_event_id, source_idempotency_key,
                payload_role, source_object_key, error_kind, error_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                error.projection_version,
                *_source_ref_error_values(error.source_ref),
                error.error_kind,
                error.model_dump_json(),
                error.created_at.isoformat(),
            ),
        )
        self.connection.commit()

    def record_reference_conflict(
        self, reference: ArtifactReference
    ) -> ProjectionError:
        error = ProjectionError(
            projection_version=reference.projection_version,
            source_ref=reference.source_ref,
            event_context=reference.event_context,
            error_kind=ProjectionErrorKind.duplicate_artifact_conflict,
            message=(
                f"Artifact {reference.artifact_id!r} conflicts with "
                "an existing artifact reference"
            ),
        )
        self.record_error(error)
        return error

    def summary(
        self, *, projection_version: str, durable_consumer: str
    ) -> ArtifactIndexSummary:
        artifact_count = self._count("artifact_references")
        open_artifact_count = self._count("open_artifact_references")
        shard_count = self._count("shards")
        open_shard_count = self._count("open_shards")
        error_count = self._count("projection_errors")
        checkpoint = self.latest_checkpoint(
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

    def clear_rebuildable_rows(self) -> None:
        with self.transaction():
            self.connection.execute("DELETE FROM open_artifact_references")
            self.connection.execute("DELETE FROM open_shards")
            self.connection.execute("DELETE FROM artifact_references")
            self.connection.execute("DELETE FROM shards")

    def transaction(self) -> _Transaction:
        return _Transaction(self.connection)

    @property
    def connection(self) -> sqlite3.Connection:
        self.connect()
        if self._connection is None:
            raise RuntimeError("artifact index is not connected")
        return self._connection

    def _count(self, table_name: str) -> int:
        if table_name not in COUNT_TABLE_NAMES:
            raise ValueError(f"unknown artifact index table {table_name!r}")
        row = self.connection.execute(
            f"SELECT COUNT(*) AS row_count FROM {table_name}"
        ).fetchone()
        return int(row["row_count"])

    def _raise_if_reference_conflicts(
        self,
        existing: ArtifactReference,
        candidate: ArtifactReference,
    ) -> None:
        if _reference_identity(existing) == _reference_identity(candidate):
            return
        raise ArtifactIndexConflictError(candidate.artifact_id)


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


def _reference_identity(reference: ArtifactReference) -> dict[str, object]:
    payload = reference.model_dump(mode="json")
    ignored_keys = {"created_at"}
    return {
        key: value for key, value in payload.items() if key not in ignored_keys
    }


def _source_ref_index_values(
    source_ref: ArtifactSourceRef,
) -> tuple[str, str, str, str, str]:
    return (
        source_ref.event_id,
        source_ref.idempotency_key,
        source_ref.payload_role,
        source_ref.object_key,
        source_ref.sha256,
    )


def _source_ref_error_values(
    source_ref: ArtifactSourceRef,
) -> tuple[str, str, str, str]:
    return (
        source_ref.event_id,
        source_ref.idempotency_key,
        source_ref.payload_role,
        source_ref.object_key,
    )


def load_manifest_references(path: Path) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            references.append(ArtifactReference.model_validate_json(line))
    return references


def load_shard_manifest(path: Path) -> ShardManifest:
    return ShardManifest.model_validate_json(path.read_text())


def dump_json_line(value: ArtifactReference) -> str:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True) + "\n"


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


__all__ = [
    "ArtifactIndex",
    "ArtifactIndexConflictError",
    "dump_json_line",
    "load_manifest_references",
    "load_shard_manifest",
]

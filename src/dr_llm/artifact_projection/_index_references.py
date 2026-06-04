from __future__ import annotations

import sqlite3
from collections.abc import Callable

from dr_llm.artifact_projection.models import (
    ArtifactReference,
    ArtifactSourceRef,
)


class ArtifactIndexConflictError(RuntimeError):
    def __init__(self, artifact_id: str) -> None:
        super().__init__(f"Artifact {artifact_id!r} conflicts with index row")
        self.artifact_id = artifact_id


class ArtifactReferenceRows:
    def __init__(self, connection: Callable[[], sqlite3.Connection]) -> None:
        self._connection = connection

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection()

    def insert_finalized_reference(self, reference: ArtifactReference) -> None:
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
            self.raise_if_reference_conflicts(existing, reference)

    def insert_open_reference(
        self, reference: ArtifactReference, *, writer_session: str
    ) -> bool:
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
        self.raise_if_reference_conflicts(existing, reference)
        return False

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

    def raise_if_reference_conflicts(
        self,
        existing: ArtifactReference,
        candidate: ArtifactReference,
    ) -> None:
        if _reference_identity(existing) == _reference_identity(candidate):
            return
        raise ArtifactIndexConflictError(candidate.artifact_id)


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


__all__ = ["ArtifactIndexConflictError", "ArtifactReferenceRows"]

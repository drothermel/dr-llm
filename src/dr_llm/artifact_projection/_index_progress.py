from __future__ import annotations

import sqlite3
from collections.abc import Callable

from dr_llm.artifact_projection.models import (
    ArtifactReference,
    ArtifactSourceRef,
    ProjectionCheckpoint,
    ProjectionError,
    ProjectionErrorKind,
)


class ArtifactProgressRows:
    def __init__(self, connection: Callable[[], sqlite3.Connection]) -> None:
        self._connection = connection

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection()

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


def _source_ref_error_values(
    source_ref: ArtifactSourceRef,
) -> tuple[str, str, str, str]:
    return (
        source_ref.event_id,
        source_ref.idempotency_key,
        source_ref.payload_role,
        source_ref.object_key,
    )


__all__ = ["ArtifactProgressRows"]

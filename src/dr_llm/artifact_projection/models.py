from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.streaming_log.events import EventEnvelope
from dr_llm.streaming_log.payloads import PayloadRef


class ArtifactLane(StrEnum):
    json = "json"
    text = "text"
    binary = "binary"


class ProjectionErrorKind(StrEnum):
    missing_payload = "missing_payload"
    source_hash_mismatch = "source_hash_mismatch"
    source_size_mismatch = "source_size_mismatch"
    unsupported_source_compression = "unsupported_source_compression"
    duplicate_artifact_conflict = "duplicate_artifact_conflict"
    invalid_payload = "invalid_payload"
    storage_error = "storage_error"


class PayloadArtifactSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_event_id: str
    source_event_type: str
    source_schema_version: int
    source_idempotency_key: str
    payload_role: str
    source_object_key: str
    source_sha256: str
    source_size_bytes: int = Field(ge=0)
    content_type: str
    encoding: str
    source_compression: str
    run_id: str | None = None
    work_id: str | None = None
    attempt_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    source: str | None = None
    producer: dict[str, Any] = Field(default_factory=dict)
    event_metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_event_ref(
        cls, *, event: EventEnvelope, payload_ref: PayloadRef
    ) -> PayloadArtifactSource:
        return cls(
            source_event_id=event.event_id,
            source_event_type=str(event.event_type),
            source_schema_version=event.schema_version,
            source_idempotency_key=event.idempotency_key,
            payload_role=payload_ref.role,
            source_object_key=payload_ref.object_key,
            source_sha256=payload_ref.sha256,
            source_size_bytes=payload_ref.size_bytes,
            content_type=payload_ref.content_type,
            encoding=payload_ref.encoding,
            source_compression=payload_ref.compression,
            run_id=event.run_id,
            work_id=event.work_id,
            attempt_id=event.attempt_id,
            causation_id=event.causation_id,
            correlation_id=event.correlation_id,
            source=event.source,
            producer=event.producer.model_dump(mode="json"),
            event_metadata=event.metadata,
        )


class ArtifactReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str
    projection_version: str
    source_event_id: str
    source_event_type: str
    source_schema_version: int
    source_idempotency_key: str
    payload_role: str
    source_object_key: str
    source_sha256: str
    logical_sha256: str
    size_bytes: int = Field(ge=0)
    content_type: str
    encoding: str
    source_compression: str
    lane: ArtifactLane
    shard_id: str
    shard_uri: str
    offset: int = Field(ge=0)
    length: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: int = Field(default=1, ge=1)
    run_id: str | None = None
    work_id: str | None = None
    attempt_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    source: str | None = None
    producer: dict[str, Any] = Field(default_factory=dict)
    event_metadata: dict[str, Any] = Field(default_factory=dict)


class ShardManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shard_id: str
    projection_version: str
    shard_uri: str
    artifact_count: int = Field(ge=0)
    lane_sizes: dict[ArtifactLane, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finalized_at: datetime | None = None
    schema_version: int = Field(default=1, ge=1)


class ProjectionCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_version: str
    durable_consumer: str
    stream_sequence: int = Field(ge=0)
    event_id: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProjectionError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_version: str
    source_event_id: str
    source_idempotency_key: str
    payload_role: str | None = None
    source_object_key: str | None = None
    error_kind: ProjectionErrorKind
    message: str
    stream_sequence: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ArtifactIndexSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_count: int = Field(ge=0)
    shard_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    checkpoint: ProjectionCheckpoint | None = None


__all__ = [
    "ArtifactIndexSummary",
    "ArtifactLane",
    "ArtifactReference",
    "PayloadArtifactSource",
    "ProjectionCheckpoint",
    "ProjectionError",
    "ProjectionErrorKind",
    "ShardManifest",
]

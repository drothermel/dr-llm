from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetadataEntityType(StrEnum):
    run = "run"
    work = "work"
    attempt = "attempt"
    producer = "producer"
    provider = "provider"
    model = "model"
    model_config = "model_config"
    prompt_instance = "prompt_instance"
    output_result = "output_result"
    artifact = "artifact"
    pool = "pool"
    pool_sample = "pool_sample"
    source_event = "source_event"


class MetadataAssertionType(StrEnum):
    pool_import_started = "pool_import_started"
    pool_sample_imported = "pool_sample_imported"
    pool_import_completed = "pool_import_completed"
    pool_import_failed = "pool_import_failed"
    work_submitted = "work_submitted"
    attempt_started = "attempt_started"
    provider_request_prepared = "provider_request_prepared"
    provider_response_received = "provider_response_received"
    attempt_succeeded = "attempt_succeeded"
    attempt_failed = "attempt_failed"
    work_retry_scheduled = "work_retry_scheduled"
    work_completed = "work_completed"
    work_cancelled = "work_cancelled"
    artifact_attached = "artifact_attached"
    producer_started = "producer_started"
    producer_stopped = "producer_stopped"
    streaming_log_error = "streaming_log_error"


class MetadataProjectionErrorKind(StrEnum):
    duplicate_entity_conflict = "duplicate_entity_conflict"
    duplicate_assertion_conflict = "duplicate_assertion_conflict"
    unsupported_event = "unsupported_event"
    invalid_event = "invalid_event"
    store_error = "store_error"


class MetadataEntity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str
    entity_type: str
    identity_key: str
    content_hash: str | None = None
    display_name: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MetadataAssertion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    assertion_id: str
    assertion_type: str
    projection_version: str
    source_event_id: str
    source_event_type: str
    source_schema_version: int
    source_idempotency_key: str
    occurred_at: datetime
    status: str | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MetadataAssertionRole(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    assertion_id: str
    role_name: str
    entity_id: str


class MetadataProjectionCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_version: str
    durable_consumer: str
    stream_sequence: int = Field(ge=0)
    event_id: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MetadataProjectionError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_version: str
    source_event_id: str
    source_idempotency_key: str
    source_event_type: str | None = None
    error_kind: MetadataProjectionErrorKind
    message: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    stream_sequence: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MetadataWritePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entities: list[MetadataEntity] = Field(default_factory=list)
    assertions: list[MetadataAssertion] = Field(default_factory=list)
    roles: list[MetadataAssertionRole] = Field(default_factory=list)
    errors: list[MetadataProjectionError] = Field(default_factory=list)


class MetadataProjectionSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_version: str
    entity_count: int = Field(ge=0)
    assertion_count: int = Field(ge=0)
    role_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    checkpoint: MetadataProjectionCheckpoint | None = None
    artifact_attach_checkpoint: MetadataProjectionCheckpoint | None = None


class MetadataVerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    problems: list[str] = Field(default_factory=list)


__all__ = [
    "MetadataAssertion",
    "MetadataAssertionRole",
    "MetadataAssertionType",
    "MetadataEntity",
    "MetadataEntityType",
    "MetadataProjectionCheckpoint",
    "MetadataProjectionError",
    "MetadataProjectionErrorKind",
    "MetadataProjectionSummary",
    "MetadataVerificationResult",
    "MetadataWritePlan",
]

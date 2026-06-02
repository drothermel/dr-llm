from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dr_llm.streaming_log.payloads import PayloadRef
from dr_llm.streaming_log.serialization import canonical_json_bytes

if TYPE_CHECKING:
    from dr_llm.streaming_log.work import QueuedWorkMessage


class StreamingLogEventType(StrEnum):
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
    producer_started = "producer_started"
    producer_stopped = "producer_stopped"
    streaming_log_error = "streaming_log_error"


class ProducerInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "dr-llm"
    version: str | None = None
    instance_id: str = Field(default_factory=lambda: uuid4().hex)


class EventContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str | None = None
    work_id: str | None = None
    attempt_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    source: str | None = None

    @classmethod
    def from_work(cls, work: QueuedWorkMessage) -> EventContext:
        return cls(
            run_id=work.run_id,
            work_id=work.work_id,
            correlation_id=work.correlation_id,
            source=work.source,
        )

    @classmethod
    def from_work_attempt(
        cls, work: QueuedWorkMessage, *, attempt_id: str
    ) -> EventContext:
        return cls(
            run_id=work.run_id,
            work_id=work.work_id,
            attempt_id=attempt_id,
            correlation_id=work.correlation_id,
            source=work.source,
        )


class EventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: StreamingLogEventType
    schema_version: int = Field(default=1, ge=1)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    producer: ProducerInfo = Field(default_factory=ProducerInfo)
    idempotency_key: str
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_refs: list[PayloadRef] = Field(default_factory=list)
    run_id: str | None = None
    work_id: str | None = None
    attempt_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value.astimezone(UTC)

    def json_bytes(self) -> bytes:
        payload = self.model_dump(
            mode="json", exclude_none=True, exclude_computed_fields=True
        )
        return canonical_json_bytes(payload)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def idempotency_key(*parts: object) -> str:
    return stable_hash([str(part) for part in parts])


def build_event(
    event_type: StreamingLogEventType,
    *,
    producer: ProducerInfo,
    idempotency_key: str,
    payload: dict[str, Any] | None = None,
    payload_refs: list[PayloadRef] | None = None,
    context: EventContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> EventEnvelope:
    context = context or EventContext()
    return EventEnvelope(
        event_type=event_type,
        producer=producer,
        idempotency_key=idempotency_key,
        payload=payload or {},
        payload_refs=payload_refs or [],
        run_id=context.run_id,
        work_id=context.work_id,
        attempt_id=context.attempt_id,
        causation_id=context.causation_id,
        correlation_id=context.correlation_id,
        source=context.source,
        metadata=metadata or {},
    )


__all__ = [
    "EventContext",
    "EventEnvelope",
    "ProducerInfo",
    "StreamingLogEventType",
    "build_event",
    "idempotency_key",
    "stable_hash",
]

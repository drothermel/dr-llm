from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

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


class EmptyEventPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PoolImportStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_name: str
    source_id: str


class PoolSampleImportedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_name: str
    source_id: str
    sample_id: str
    sample_idx: int | None = None
    run_id: str | None = None
    key_values: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None
    attempt_count: int = Field(ge=0)
    created_at: str | None = None
    completion_state: str
    reconstructed: bool
    row_state_hash: str


class PoolImportCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_name: str
    source_id: str
    imported_count: int = Field(ge=0)
    reconstructed: bool


class PoolImportFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pool_name: str
    source_id: str
    error_type: str
    message: str


class RequestSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    mode: str
    message_count: int = Field(ge=0)
    messages_sha256: str
    prompt_preview: str | None = None
    max_tokens: int | None = Field(default=None, ge=0)
    effort: str | None = None
    sampling: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    mode: str
    text_sha256: str
    text_preview: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any] | None = None
    latency_ms: int = Field(default=0, ge=0)


class WorkSubmittedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    work_id: str
    run_id: str | None = None
    max_retries: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_summary: RequestSummary | None = None


class AttemptStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: str
    attempt: int = Field(ge=1)


class ProviderRequestPreparedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    mode: str
    request_summary: RequestSummary | None = None


class ProviderResponseReceivedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    mode: str
    finish_reason: str | None = None
    response_summary: ResponseSummary | None = None


class AttemptSucceededPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int = Field(ge=1)


class AttemptFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: str
    message: str
    attempt: int = Field(ge=1)


class WorkRetryScheduledPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int = Field(ge=1)
    next_attempt: int = Field(ge=2)


class WorkCompletedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    attempt: int = Field(ge=1)
    error_type: str | None = None
    message: str | None = None


class WorkCancelledPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    work_id: str
    reason: str | None = None


class ProducerLifecyclePayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: str


class StreamingLogErrorPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: str
    message: str


type EventPayload = (
    EmptyEventPayload
    | PoolImportStartedPayload
    | PoolSampleImportedPayload
    | PoolImportCompletedPayload
    | PoolImportFailedPayload
    | WorkSubmittedPayload
    | AttemptStartedPayload
    | ProviderRequestPreparedPayload
    | ProviderResponseReceivedPayload
    | AttemptSucceededPayload
    | AttemptFailedPayload
    | WorkRetryScheduledPayload
    | WorkCompletedPayload
    | WorkCancelledPayload
    | ProducerLifecyclePayload
    | StreamingLogErrorPayload
)
type EventPayloadModel = type[EventPayload]


EVENT_PAYLOAD_MODELS: dict[StreamingLogEventType, EventPayloadModel] = {
    StreamingLogEventType.pool_import_started: PoolImportStartedPayload,
    StreamingLogEventType.pool_sample_imported: PoolSampleImportedPayload,
    StreamingLogEventType.pool_import_completed: PoolImportCompletedPayload,
    StreamingLogEventType.pool_import_failed: PoolImportFailedPayload,
    StreamingLogEventType.work_submitted: WorkSubmittedPayload,
    StreamingLogEventType.attempt_started: AttemptStartedPayload,
    StreamingLogEventType.provider_request_prepared: (
        ProviderRequestPreparedPayload
    ),
    StreamingLogEventType.provider_response_received: (
        ProviderResponseReceivedPayload
    ),
    StreamingLogEventType.attempt_succeeded: AttemptSucceededPayload,
    StreamingLogEventType.attempt_failed: AttemptFailedPayload,
    StreamingLogEventType.work_retry_scheduled: WorkRetryScheduledPayload,
    StreamingLogEventType.work_completed: WorkCompletedPayload,
    StreamingLogEventType.work_cancelled: WorkCancelledPayload,
    StreamingLogEventType.producer_started: ProducerLifecyclePayload,
    StreamingLogEventType.producer_stopped: ProducerLifecyclePayload,
    StreamingLogEventType.streaming_log_error: StreamingLogErrorPayload,
}


def payload_model_for_event_type(
    event_type: StreamingLogEventType,
) -> EventPayloadModel:
    return EVENT_PAYLOAD_MODELS[event_type]


class EventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: StreamingLogEventType
    schema_version: int = Field(default=1, ge=1)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    producer: ProducerInfo = Field(default_factory=ProducerInfo)
    idempotency_key: str
    payload: EventPayload = Field(default_factory=EmptyEventPayload)
    payload_refs: list[PayloadRef] = Field(default_factory=list)
    run_id: str | None = None
    work_id: str | None = None
    attempt_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _build_typed_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        raw_value = cast("dict[str, Any]", value)
        raw_event_type = raw_value.get("event_type")
        if raw_event_type is None:
            return value
        event_type = StreamingLogEventType(raw_event_type)
        payload_model = payload_model_for_event_type(event_type)
        payload = raw_value.get("payload", {})
        if isinstance(payload, dict):
            raw_value = dict(raw_value)
            raw_value["payload"] = payload_model(**payload)
        return raw_value

    @model_validator(mode="after")
    def _require_matching_payload_model(self) -> EventEnvelope:
        expected_model = payload_model_for_event_type(self.event_type)
        if not isinstance(self.payload, expected_model):
            raise ValueError(
                f"{self.event_type} payload must be {expected_model.__name__}"
            )
        return self

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
    payload: EventPayload,
    payload_refs: list[PayloadRef] | None = None,
    context: EventContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> EventEnvelope:
    context = context or EventContext()
    return EventEnvelope(
        event_type=event_type,
        producer=producer,
        idempotency_key=idempotency_key,
        payload=payload,
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
    "AttemptFailedPayload",
    "AttemptStartedPayload",
    "AttemptSucceededPayload",
    "EmptyEventPayload",
    "EventContext",
    "EventEnvelope",
    "EventPayload",
    "PoolImportCompletedPayload",
    "PoolImportFailedPayload",
    "PoolImportStartedPayload",
    "PoolSampleImportedPayload",
    "ProducerInfo",
    "ProducerLifecyclePayload",
    "ProviderRequestPreparedPayload",
    "ProviderResponseReceivedPayload",
    "StreamingLogErrorPayload",
    "StreamingLogEventType",
    "WorkCancelledPayload",
    "WorkCompletedPayload",
    "WorkRetryScheduledPayload",
    "WorkSubmittedPayload",
    "build_event",
    "idempotency_key",
    "payload_model_for_event_type",
    "stable_hash",
]

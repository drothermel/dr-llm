from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dr_llm.llm import LlmRequest, LlmResponse
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.streaming_log.events import (
    AttemptSucceededPayload,
    EventPayload,
    EventContext,
    PoolSampleImportedPayload,
    ProviderResponseReceivedPayload,
    RequestSummary,
    ResponseSummary,
    StreamingLogEventType,
    WorkCompletedPayload,
    idempotency_key,
    payload_model_for_event_type,
    stable_hash,
)
from dr_llm.streaming_log.payloads import (
    PreparedPayload,
    prepare_json_payload,
)


class StreamingEventPublishSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: StreamingLogEventType
    idempotency_key: str
    payload: EventPayload
    payloads: list[PreparedPayload] = Field(default_factory=list)
    context: EventContext | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_matching_payload_model(self) -> StreamingEventPublishSpec:
        expected_model = payload_model_for_event_type(self.event_type)
        if not isinstance(self.payload, expected_model):
            raise ValueError(
                f"{self.event_type} payload must be {expected_model.__name__}"
            )
        return self


def pool_sample_imported_event(
    *,
    pool_name: str,
    source_id: str,
    schema_payload: dict[str, Any],
    sample: PoolSample,
) -> StreamingEventPublishSpec:
    sample_payload = _sample_snapshot_payload(sample)
    state_hash = stable_hash(sample_payload)
    return StreamingEventPublishSpec(
        event_type=StreamingLogEventType.pool_sample_imported,
        idempotency_key=idempotency_key(
            source_id,
            pool_name,
            sample.sample_id,
            sample.sample_idx,
            state_hash,
        ),
        payload=_pool_sample_imported_payload(
            pool_name=pool_name,
            source_id=source_id,
            sample=sample,
            state_hash=state_hash,
        ),
        payloads=_pool_sample_payload_refs(
            schema_payload=schema_payload, sample=sample
        ),
        context=_pool_sample_event_context(sample=sample, source_id=source_id),
        metadata={"reconstructed": True},
    )


def provider_response_received_event(
    *,
    work_id: str,
    attempt: int,
    response: LlmResponse,
) -> StreamingEventPublishSpec:
    response_payload = response.model_dump(
        mode="json",
        exclude_none=True,
        exclude_computed_fields=True,
    )
    return StreamingEventPublishSpec(
        event_type=StreamingLogEventType.provider_response_received,
        idempotency_key=idempotency_key(
            "provider_response_received", work_id, attempt
        ),
        payload=ProviderResponseReceivedPayload(
            provider=response.provider,
            model=response.model,
            mode=str(response.mode),
            finish_reason=response.finish_reason,
            response_summary=response_summary_from_response(response),
        ),
        payloads=[prepare_json_payload("response_json", response_payload)],
    )


def request_summary_from_request(request: LlmRequest) -> RequestSummary:
    messages = [
        message.model_dump(mode="json")
        for message in getattr(request, "messages", [])
    ]
    return RequestSummary(
        provider=str(request.provider),
        model=request.model,
        mode=str(request.mode),
        message_count=len(messages),
        messages_sha256=stable_hash(messages),
        prompt_preview=_prompt_preview(messages),
        max_tokens=request.max_tokens,
        effort=str(request.effort),
        sampling=_optional_model_dump(request.sampling),
        metadata=request.metadata,
    )


def response_summary_from_response(response: LlmResponse) -> ResponseSummary:
    return ResponseSummary(
        provider=response.provider,
        model=response.model,
        mode=str(response.mode),
        text_sha256=stable_hash(response.text),
        text_preview=_text_preview(response.text),
        finish_reason=response.finish_reason,
        usage=response.usage.model_dump(mode="json"),
        cost=_optional_model_dump(response.cost),
        latency_ms=response.latency_ms,
    )


def attempt_succeeded_event(
    *, work_id: str, attempt: int
) -> StreamingEventPublishSpec:
    return StreamingEventPublishSpec(
        event_type=StreamingLogEventType.attempt_succeeded,
        idempotency_key=idempotency_key("attempt_succeeded", work_id, attempt),
        payload=AttemptSucceededPayload(attempt=attempt),
    )


def work_completed_succeeded_event(
    *, work_id: str, attempt: int
) -> StreamingEventPublishSpec:
    return StreamingEventPublishSpec(
        event_type=StreamingLogEventType.work_completed,
        idempotency_key=idempotency_key("work_completed", work_id),
        payload=WorkCompletedPayload(status="succeeded", attempt=attempt),
    )


def _pool_sample_imported_payload(
    *,
    pool_name: str,
    source_id: str,
    sample: PoolSample,
    state_hash: str,
) -> PoolSampleImportedPayload:
    return PoolSampleImportedPayload(
        pool_name=pool_name,
        source_id=source_id,
        sample_id=sample.sample_id,
        sample_idx=sample.sample_idx,
        run_id=sample.run_id,
        key_values=sample.key_values,
        finish_reason=sample.finish_reason,
        attempt_count=sample.attempt_count,
        created_at=(
            sample.created_at.isoformat()
            if sample.created_at is not None
            else None
        ),
        completion_state="complete" if sample.is_complete else "incomplete",
        reconstructed=True,
        row_state_hash=state_hash,
    )


def _pool_sample_payload_refs(
    *, schema_payload: dict[str, Any], sample: PoolSample
) -> list[PreparedPayload]:
    return [
        prepare_json_payload("pool_schema", schema_payload),
        prepare_json_payload("request_json", sample.request),
        prepare_json_payload("metadata_json", sample.metadata),
        *_response_payloads(sample),
    ]


def _pool_sample_event_context(
    *, sample: PoolSample, source_id: str
) -> EventContext:
    return EventContext(run_id=sample.run_id, source=source_id)


def _response_payloads(sample: PoolSample) -> list[PreparedPayload]:
    if sample.response is None:
        return []
    return [prepare_json_payload("response_json", sample.response)]


def _optional_model_dump(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return value.model_dump(mode="json", exclude_none=True)


def _prompt_preview(messages: list[dict[str, Any]]) -> str | None:
    text = "\n".join(str(message.get("content", "")) for message in messages)
    return _text_preview(text)


def _text_preview(text: str, *, max_chars: int = 500) -> str | None:
    if not text:
        return None
    return text[:max_chars]


def _sample_snapshot_payload(sample: PoolSample) -> dict[str, Any]:
    return sample.model_dump(
        mode="json",
        exclude_none=True,
        exclude_computed_fields=True,
    )


__all__ = [
    "StreamingEventPublishSpec",
    "attempt_succeeded_event",
    "pool_sample_imported_event",
    "provider_response_received_event",
    "request_summary_from_request",
    "response_summary_from_response",
    "work_completed_succeeded_event",
]

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm import LlmResponse
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.streaming_log.events import (
    EventContext,
    StreamingLogEventType,
    idempotency_key,
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
    payload: dict[str, Any] = Field(default_factory=dict)
    payloads: list[PreparedPayload] = Field(default_factory=list)
    context: EventContext | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
        payload={
            "pool_name": pool_name,
            "source_id": source_id,
            "sample_id": sample.sample_id,
            "sample_idx": sample.sample_idx,
            "run_id": sample.run_id,
            "key_values": sample.key_values,
            "finish_reason": sample.finish_reason,
            "attempt_count": sample.attempt_count,
            "created_at": (
                sample.created_at.isoformat()
                if sample.created_at is not None
                else None
            ),
            "completion_state": (
                "complete" if sample.is_complete else "incomplete"
            ),
            "reconstructed": True,
            "row_state_hash": state_hash,
        },
        payloads=[
            prepare_json_payload("pool_schema", schema_payload),
            prepare_json_payload("request_json", sample.request),
            prepare_json_payload("metadata_json", sample.metadata),
            *_response_payloads(sample),
        ],
        context=EventContext(run_id=sample.run_id, source=source_id),
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
        payload={
            "provider": response.provider,
            "model": response.model,
            "mode": response.mode,
            "finish_reason": response.finish_reason,
        },
        payloads=[prepare_json_payload("response_json", response_payload)],
    )


def attempt_succeeded_event(
    *, work_id: str, attempt: int
) -> StreamingEventPublishSpec:
    return StreamingEventPublishSpec(
        event_type=StreamingLogEventType.attempt_succeeded,
        idempotency_key=idempotency_key("attempt_succeeded", work_id, attempt),
        payload={"attempt": attempt},
    )


def work_completed_succeeded_event(
    *, work_id: str, attempt: int
) -> StreamingEventPublishSpec:
    return StreamingEventPublishSpec(
        event_type=StreamingLogEventType.work_completed,
        idempotency_key=idempotency_key("work_completed", work_id),
        payload={"status": "succeeded", "attempt": attempt},
    )


def _response_payloads(sample: PoolSample) -> list[PreparedPayload]:
    if sample.response is None:
        return []
    return [prepare_json_payload("response_json", sample.response)]


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
    "work_completed_succeeded_event",
]

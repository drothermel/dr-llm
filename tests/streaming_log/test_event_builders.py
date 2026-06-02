from __future__ import annotations

from dr_llm.llm import CallMode, LlmResponse
from dr_llm.pool.pool_sample import PoolSample
from dr_llm.streaming_log.event_builders import (
    attempt_succeeded_event,
    pool_sample_imported_event,
    provider_response_received_event,
    work_completed_succeeded_event,
)
from dr_llm.streaming_log.events import (
    AttemptSucceededPayload,
    EventContext,
    PoolSampleImportedPayload,
    StreamingLogEventType,
    WorkCompletedPayload,
)


def _sample() -> PoolSample:
    return PoolSample(
        sample_id="sample-1",
        key_values={"dim": "a"},
        sample_idx=3,
        run_id="run-1",
        request={"prompt": "hello"},
        response={"text": "world"},
        finish_reason="stop",
        attempt_count=2,
        metadata={"m": 1},
    )


def _response() -> LlmResponse:
    return LlmResponse(
        text="done",
        provider="openai",
        model="gpt-test",
        mode=CallMode.api,
        finish_reason="stop",
        raw_json={"id": "resp-1"},
    )


def test_pool_sample_imported_event_builds_reconstructed_sample_fact() -> None:
    spec = pool_sample_imported_event(
        pool_name="pool-1",
        source_id="source-1",
        schema_payload={"name": "pool-1"},
        sample=_sample(),
    )

    assert spec.event_type is StreamingLogEventType.pool_sample_imported
    payload = spec.payload
    assert isinstance(payload, PoolSampleImportedPayload)
    assert payload.pool_name == "pool-1"
    assert payload.source_id == "source-1"
    assert payload.sample_id == "sample-1"
    assert payload.sample_idx == 3
    assert payload.run_id == "run-1"
    assert payload.key_values == {"dim": "a"}
    assert payload.finish_reason == "stop"
    assert payload.attempt_count == 2
    assert payload.completion_state == "complete"
    assert payload.reconstructed is True
    assert isinstance(payload.row_state_hash, str)
    assert [payload.role for payload in spec.payloads] == [
        "pool_schema",
        "request_json",
        "metadata_json",
        "response_json",
    ]
    assert spec.context == EventContext(run_id="run-1", source="source-1")
    assert spec.metadata == {"reconstructed": True}


def test_pool_sample_imported_event_key_changes_with_row_state() -> None:
    first = pool_sample_imported_event(
        pool_name="pool-1",
        source_id="source-1",
        schema_payload={"name": "pool-1"},
        sample=_sample(),
    )
    second = pool_sample_imported_event(
        pool_name="pool-1",
        source_id="source-1",
        schema_payload={"name": "pool-1"},
        sample=_sample().model_copy(update={"response": {"text": "changed"}}),
    )

    assert first.idempotency_key != second.idempotency_key


def test_provider_response_received_event_builds_response_fact() -> None:
    spec = provider_response_received_event(
        work_id="work-1",
        attempt=2,
        response=_response(),
    )

    assert spec.event_type is StreamingLogEventType.provider_response_received
    assert spec.payload.model_dump(mode="json") == {
        "provider": "openai",
        "model": "gpt-test",
        "mode": "api",
        "finish_reason": "stop",
    }
    assert [payload.role for payload in spec.payloads] == ["response_json"]
    assert spec.context is None
    assert spec.metadata == {}


def test_success_completion_event_builders_record_attempt_identity() -> None:
    succeeded = attempt_succeeded_event(work_id="work-1", attempt=2)
    completed = work_completed_succeeded_event(work_id="work-1", attempt=2)

    assert succeeded.event_type is StreamingLogEventType.attempt_succeeded
    assert isinstance(succeeded.payload, AttemptSucceededPayload)
    assert succeeded.payload.attempt == 2
    assert completed.event_type is StreamingLogEventType.work_completed
    assert isinstance(completed.payload, WorkCompletedPayload)
    assert completed.payload.status == "succeeded"
    assert completed.payload.attempt == 2
    assert succeeded.idempotency_key != completed.idempotency_key

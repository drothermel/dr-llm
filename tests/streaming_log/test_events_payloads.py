from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from dr_llm.llm import CallMode, LlmRequest, Message, ProviderName
from dr_llm.streaming_log.events import (
    AttemptStartedPayload,
    EventContext,
    EventEnvelope,
    ProducerInfo,
    ProducerLifecyclePayload,
    StreamingLogEventType,
    WorkSubmittedPayload,
    build_event,
    idempotency_key,
    stable_hash,
)
from dr_llm.streaming_log.payloads import (
    PayloadRef,
    object_key_for_sha256,
    prepare_json_payload,
    prepare_text_payload,
    sha256_bytes,
)
from dr_llm.streaming_log.serialization import canonical_json_bytes
from dr_llm.streaming_log.work import QueuedWorkMessage


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hi")],
    )


def _work_submitted_payload() -> WorkSubmittedPayload:
    return WorkSubmittedPayload(work_id="work-1", max_retries=0)


def test_event_envelope_requires_strict_event_type() -> None:
    event = EventEnvelope(
        event_type=StreamingLogEventType.work_submitted,
        producer=ProducerInfo(name="test", version="1"),
        idempotency_key="same",
        payload=_work_submitted_payload(),
    )

    assert event.event_type is StreamingLogEventType.work_submitted
    assert event.occurred_at.tzinfo is not None


def test_event_envelope_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        EventEnvelope(
            event_type=StreamingLogEventType.work_submitted,
            producer=ProducerInfo(),
            idempotency_key="same",
            payload=_work_submitted_payload(),
            occurred_at=datetime(2026, 1, 1),
        )


def test_idempotency_key_is_deterministic() -> None:
    first = idempotency_key("pool", 1, {"a": "b"})
    second = idempotency_key("pool", 1, {"a": "b"})
    different = idempotency_key("pool", 2, {"a": "b"})

    assert first == second
    assert first != different


def test_stable_json_bytes_are_shared_by_hashing_and_payloads() -> None:
    payload = {"b": 2, "a": 1}
    prepared = prepare_json_payload("request_json", payload)

    assert canonical_json_bytes(payload) == b'{"a":1,"b":2}'
    assert prepared.data == canonical_json_bytes(payload)
    assert prepared.sha256 == sha256_bytes(canonical_json_bytes(payload))
    assert stable_hash(payload) == sha256_bytes(canonical_json_bytes(payload))


def test_payload_hash_and_object_key_are_content_addressed() -> None:
    payload = prepare_text_payload("stdout", "hello")
    digest = sha256_bytes(b"hello")

    assert payload.sha256 == digest
    assert payload.object_key == f"sha256/{digest[:2]}/{digest}"
    assert payload.ref().model_dump(mode="json") == {
        "role": "stdout",
        "object_key": f"sha256/{digest[:2]}/{digest}",
        "sha256": digest,
        "size_bytes": 5,
        "content_type": "text/plain",
        "encoding": "utf-8",
        "compression": "none",
    }


def test_json_payload_serialization_is_stable() -> None:
    left = prepare_json_payload("request_json", {"b": 2, "a": 1})
    right = prepare_json_payload("request_json", {"a": 1, "b": 2})

    assert left.data == b'{"a":1,"b":2}'
    assert left.sha256 == right.sha256


def test_event_envelope_validates_payload_refs() -> None:
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate(
            {
                "event_type": StreamingLogEventType.work_submitted,
                "producer": ProducerInfo(name="test"),
                "idempotency_key": "same",
                "payload": _work_submitted_payload(),
                "payload_refs": [{"role": "request_json"}],
            }
        )


def test_event_envelope_restores_typed_payload_refs_from_json() -> None:
    ref = PayloadRef(
        role="request_json",
        object_key="sha256/00/test",
        sha256="0" * 64,
        size_bytes=5,
        content_type="application/json",
        encoding="utf-8",
    )
    event = EventEnvelope(
        event_type=StreamingLogEventType.work_submitted,
        producer=ProducerInfo(name="test"),
        idempotency_key="same",
        payload=_work_submitted_payload(),
        payload_refs=[ref],
    )

    restored = EventEnvelope.model_validate_json(event.json_bytes())

    assert restored.payload == _work_submitted_payload()
    assert restored.payload_refs == [ref]
    assert isinstance(restored.payload_refs[0], PayloadRef)


def test_event_envelope_rejects_mismatched_payload_model() -> None:
    with pytest.raises(ValueError, match="WorkSubmittedPayload"):
        EventEnvelope(
            event_type=StreamingLogEventType.work_submitted,
            producer=ProducerInfo(name="test"),
            idempotency_key="same",
            payload=ProducerLifecyclePayload(worker_id="worker-1"),
        )


def test_build_event_accepts_typed_payload_refs() -> None:
    ref = PayloadRef(
        role="request_json",
        object_key="sha256/00/test",
        sha256="0" * 64,
        size_bytes=5,
        content_type="application/json",
        encoding="utf-8",
    )

    event = build_event(
        StreamingLogEventType.work_submitted,
        producer=ProducerInfo(name="test"),
        idempotency_key="same",
        payload=_work_submitted_payload(),
        payload_refs=[ref],
    )

    assert event.payload_refs == [ref]


def test_object_key_rejects_invalid_digest() -> None:
    with pytest.raises(ValueError, match="64 hex"):
        object_key_for_sha256("nope")


def test_timestamp_is_normalized_to_utc() -> None:
    event = EventEnvelope(
        event_type=StreamingLogEventType.producer_started,
        producer=ProducerInfo(),
        idempotency_key="same",
        payload=ProducerLifecyclePayload(worker_id="worker-1"),
        occurred_at=datetime.now(UTC),
    )

    assert event.occurred_at.tzinfo is UTC


def test_event_context_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EventContext.model_validate({"run_id": "run-1", "extra_field": "nope"})


def test_event_context_builds_from_work_and_attempt() -> None:
    work = QueuedWorkMessage(
        work_id="work-1",
        request=_request(),
        run_id="run-1",
        correlation_id="corr-1",
        source="test-source",
    )

    context = EventContext.from_work(work)
    attempt_context = EventContext.from_work_attempt(
        work, attempt_id="attempt-1"
    )

    assert context == EventContext(
        run_id="run-1",
        work_id="work-1",
        correlation_id="corr-1",
        source="test-source",
    )
    assert attempt_context == EventContext(
        run_id="run-1",
        work_id="work-1",
        attempt_id="attempt-1",
        correlation_id="corr-1",
        source="test-source",
    )


def test_build_event_applies_context_to_envelope() -> None:
    event = build_event(
        StreamingLogEventType.attempt_started,
        producer=ProducerInfo(name="test"),
        idempotency_key="same",
        payload=AttemptStartedPayload(worker_id="worker-1", attempt=1),
        context=EventContext(
            run_id="run-1",
            work_id="work-1",
            attempt_id="attempt-1",
            causation_id="cause-1",
            correlation_id="corr-1",
            source="test-source",
        ),
    )

    assert event.run_id == "run-1"
    assert event.work_id == "work-1"
    assert event.attempt_id == "attempt-1"
    assert event.causation_id == "cause-1"
    assert event.correlation_id == "corr-1"
    assert event.source == "test-source"

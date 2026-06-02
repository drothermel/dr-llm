from __future__ import annotations

import asyncio

import pytest

from dr_llm.llm import CallMode, LlmRequest, Message, ProviderName
from dr_llm.streaming_log.client import (
    StreamingEventLog,
    StreamingLogConnection,
    StreamingPayloadStore,
    StreamingWorkQueue,
)
from dr_llm.streaming_log.errors import PayloadIntegrityError
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    StreamingLogEventType,
)
from dr_llm.streaming_log.payloads import prepare_text_payload
from dr_llm.streaming_log.work import QueuedWorkMessage
from tests.streaming_log.helpers import (
    FakeJetStream,
    FakeStreamingLogConnection,
)


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hi")],
    )


def _clients() -> tuple[
    StreamingPayloadStore, StreamingEventLog, StreamingWorkQueue, FakeJetStream
]:
    fake_js = FakeJetStream()
    connection: StreamingLogConnection = FakeStreamingLogConnection(fake_js)
    payload_store = StreamingPayloadStore(connection)
    event_log = StreamingEventLog(connection, payload_store)
    work_queue = StreamingWorkQueue(connection, event_log)
    return payload_store, event_log, work_queue, fake_js


def test_payload_store_writes_new_payload_and_returns_ref() -> None:
    payload_store, _, _, fake_js = _clients()
    payload = prepare_text_payload("stdout", "hello")

    ref = asyncio.run(payload_store.write_payload(payload))

    assert ref == payload.ref()
    assert fake_js.store.objects[payload.object_key] == b"hello"


def test_payload_store_allows_existing_identical_payload() -> None:
    payload_store, _, _, fake_js = _clients()
    payload = prepare_text_payload("stdout", "hello")
    fake_js.store.objects[payload.object_key] = payload.data

    ref = asyncio.run(payload_store.write_payload(payload))

    assert ref == payload.ref()
    assert fake_js.store.objects[payload.object_key] == b"hello"


def test_payload_store_rejects_existing_different_payload() -> None:
    payload_store, _, _, fake_js = _clients()
    payload = prepare_text_payload("stdout", "hello")
    fake_js.store.objects[payload.object_key] = b"different"

    with pytest.raises(PayloadIntegrityError, match="different bytes"):
        asyncio.run(payload_store.write_payload(payload))


def test_payload_store_reads_payload_ref_bytes() -> None:
    payload_store, _, _, fake_js = _clients()
    payload = prepare_text_payload("stdout", "hello")
    fake_js.store.objects[payload.object_key] = payload.data

    data = asyncio.run(payload_store.read_payload_ref(payload.ref()))

    assert data == b"hello"


def test_payload_store_rejects_payload_ref_hash_mismatch() -> None:
    payload_store, _, _, fake_js = _clients()
    payload = prepare_text_payload("stdout", "hello")
    ref = payload.ref().model_copy(update={"sha256": "0" * 64})
    fake_js.store.objects[payload.object_key] = payload.data

    with pytest.raises(PayloadIntegrityError, match="sha256 mismatch"):
        asyncio.run(payload_store.read_payload_ref(ref))


def test_payload_store_rejects_payload_ref_size_mismatch() -> None:
    payload_store, _, _, fake_js = _clients()
    payload = prepare_text_payload("stdout", "hello")
    ref = payload.ref().model_copy(update={"size_bytes": 6})
    fake_js.store.objects[payload.object_key] = payload.data

    with pytest.raises(PayloadIntegrityError, match="size mismatch"):
        asyncio.run(payload_store.read_payload_ref(ref))


def test_payload_store_rejects_empty_object_result() -> None:
    payload_store, _, _, fake_js = _clients()
    payload = prepare_text_payload("stdout", "hello")
    fake_js.store.objects[payload.object_key] = None

    with pytest.raises(PayloadIntegrityError, match="returned no bytes"):
        asyncio.run(payload_store.read_payload_ref(payload.ref()))


def test_work_queue_publishes_event_before_work_message() -> None:
    _, _, work_queue, fake_js = _clients()
    work = QueuedWorkMessage(work_id="work-1", request=_request())

    event = asyncio.run(work_queue.submit_work(work))

    assert event.event_type is StreamingLogEventType.work_submitted
    assert fake_js.published[0].subject == "drllm.events.work_submitted"
    assert fake_js.published[0].stream == "DRLLM_EVENTS"
    assert fake_js.published[1].subject == "drllm.work.llm"
    assert fake_js.published[1].stream == "DRLLM_WORK"
    assert fake_js.store.objects


def test_work_queue_rejects_retry_budget_that_exceeds_deliveries() -> None:
    _, _, work_queue, fake_js = _clients()
    work = QueuedWorkMessage(
        work_id="work-1", request=_request(), max_retries=3
    )

    with pytest.raises(ValueError, match="max_retries"):
        asyncio.run(work_queue.submit_work(work))

    assert fake_js.published == []
    assert fake_js.store.objects == {}


def test_event_log_contextual_publisher_applies_context_and_writes_payloads() -> (
    None
):
    _, event_log, _, fake_js = _clients()
    publisher = event_log.with_event_context(
        EventContext(
            run_id="run-1",
            work_id="work-1",
            attempt_id="attempt-1",
            correlation_id="corr-1",
            source="test-source",
        )
    )

    event = asyncio.run(
        publisher.publish_event_with_payloads(
            StreamingLogEventType.attempt_started,
            idempotency_key="attempt-started-1",
            payload={"attempt": 1},
            payloads=[prepare_text_payload("stdout", "hello")],
        )
    )
    published = EventEnvelope.model_validate_json(fake_js.published[0].payload)

    assert event.run_id == "run-1"
    assert event.work_id == "work-1"
    assert event.attempt_id == "attempt-1"
    assert event.correlation_id == "corr-1"
    assert event.source == "test-source"
    assert published.model_dump(mode="json") == event.model_dump(mode="json")
    assert list(fake_js.store.objects.values()) == [b"hello"]

from __future__ import annotations

import asyncio
from typing import Any, cast

from dr_llm.llm import CallMode, LlmRequest, Message, ProviderName
from dr_llm.streaming_log.client import StreamingLogClient
from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    StreamingLogEventType,
)
from dr_llm.streaming_log.payloads import prepare_text_payload
from dr_llm.streaming_log.work import QueuedWorkMessage


class FakeObjectResult:
    def __init__(self, data: bytes) -> None:
        self.data = data


class FakeObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def get(self, name: str):
        from nats.js.errors import ObjectNotFoundError

        if name not in self.objects:
            raise ObjectNotFoundError
        return FakeObjectResult(self.objects[name])

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data


class FakeJetStream:
    def __init__(self) -> None:
        self.store = FakeObjectStore()
        self.published: list[tuple[str, bytes, str | None]] = []

    async def object_store(self, bucket: str) -> FakeObjectStore:
        assert bucket == "DRLLM_PAYLOADS"
        return self.store

    async def publish(
        self, subject: str, payload: bytes, *, stream: str | None = None
    ) -> object:
        self.published.append((subject, payload, stream))
        return object()


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hi")],
    )


def test_submit_work_publishes_event_before_work_message() -> None:
    client = StreamingLogClient(StreamingLogConfig())
    fake_js = FakeJetStream()
    client._js = cast(Any, fake_js)

    work = QueuedWorkMessage(work_id="work-1", request=_request())

    event = asyncio.run(client.submit_work(work))

    assert event.event_type is StreamingLogEventType.work_submitted
    assert fake_js.published[0][0] == "drllm.events.work_submitted"
    assert fake_js.published[0][2] == "DRLLM_EVENTS"
    assert fake_js.published[1][0] == "drllm.work.llm"
    assert fake_js.published[1][2] == "DRLLM_WORK"
    assert fake_js.store.objects


def test_contextual_publisher_applies_context_and_writes_payloads() -> None:
    client = StreamingLogClient(StreamingLogConfig())
    fake_js = FakeJetStream()
    client._js = cast(Any, fake_js)
    publisher = client.with_event_context(
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
    published = EventEnvelope.model_validate_json(fake_js.published[0][1])

    assert event.run_id == "run-1"
    assert event.work_id == "work-1"
    assert event.attempt_id == "attempt-1"
    assert event.correlation_id == "corr-1"
    assert event.source == "test-source"
    assert published.model_dump(mode="json") == event.model_dump(mode="json")
    assert list(fake_js.store.objects.values()) == [b"hello"]

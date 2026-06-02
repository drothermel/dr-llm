from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from dr_llm.llm import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    ProviderName,
)
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
    build_event,
)
from dr_llm.streaming_log.work import QueuedWorkMessage
from dr_llm.streaming_log.workers import _process_message


class FakePublisher:
    def __init__(self, context: EventContext) -> None:
        self.context = context
        self.events: list[EventEnvelope] = []

    async def publish_event_with_payloads(
        self,
        event_type: StreamingLogEventType,
        *,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        payloads: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventEnvelope:
        del payloads
        event = build_event(
            event_type,
            producer=ProducerInfo(name="test"),
            idempotency_key=idempotency_key,
            payload=payload,
            context=self.context,
            metadata=metadata,
        )
        self.events.append(event)
        return event


class FakeClient:
    def __init__(self) -> None:
        self.publisher: FakePublisher | None = None

    def with_event_context(self, context: EventContext) -> FakePublisher:
        self.publisher = FakePublisher(context)
        return self.publisher


class FakeOrchestrator:
    def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            text="done",
            provider=str(request.provider),
            model=request.model,
            mode=request.mode,
        )


class FakeRegistry:
    def get(self, provider: str) -> FakeOrchestrator:
        assert provider == str(ProviderName.OPENAI)
        return FakeOrchestrator()


class FakeMessage:
    def __init__(self, work: QueuedWorkMessage) -> None:
        self.data = work.json_bytes()
        self.metadata = SimpleNamespace(num_delivered=2)
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hi")],
    )


def test_worker_lifecycle_uses_one_attempt_context() -> None:
    work = QueuedWorkMessage(
        work_id="work-1",
        request=_request(),
        run_id="run-1",
        correlation_id="corr-1",
        source="test-source",
    )
    client = FakeClient()
    msg = FakeMessage(work)

    asyncio.run(
        _process_message(
            client=cast(Any, client),
            registry=cast(Any, FakeRegistry()),
            worker_id="worker-1",
            msg=msg,
        )
    )

    assert client.publisher is not None
    assert msg.acked
    assert [event.event_type for event in client.publisher.events] == [
        StreamingLogEventType.attempt_started,
        StreamingLogEventType.provider_request_prepared,
        StreamingLogEventType.provider_response_received,
        StreamingLogEventType.attempt_succeeded,
        StreamingLogEventType.work_completed,
    ]
    for event in client.publisher.events:
        assert event.run_id == "run-1"
        assert event.work_id == "work-1"
        assert event.attempt_id == "work-1-2"
        assert event.correlation_id == "corr-1"
        assert event.source == "test-source"

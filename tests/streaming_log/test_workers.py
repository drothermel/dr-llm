from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    ProviderName,
)
from dr_llm.streaming_log.client import StreamingEventLog
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
    build_event,
)
from dr_llm.streaming_log.payloads import PreparedPayload
from dr_llm.streaming_log.work import QueuedWorkMessage
from dr_llm.streaming_log.workers import (
    StreamingRetryPolicy,
    StreamingWorkAttempt,
    StreamingWorkMessageHandler,
    StreamingWorkOutcomeType,
    StreamingWorkProcessor,
)


class PublishedEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event: EventEnvelope
    payload_roles: list[str] = Field(default_factory=list)


class FakePublisher:
    def __init__(self, context: EventContext) -> None:
        self.context = context
        self.published: list[PublishedEvent] = []

    @property
    def events(self) -> list[EventEnvelope]:
        return [record.event for record in self.published]

    async def publish_event_with_payloads(
        self,
        event_type: StreamingLogEventType,
        *,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        payloads: list[PreparedPayload] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventEnvelope:
        payloads = payloads or []
        event = build_event(
            event_type,
            producer=ProducerInfo(name="test"),
            idempotency_key=idempotency_key,
            payload=payload,
            payload_refs=[payload.ref() for payload in payloads],
            context=self.context,
            metadata=metadata,
        )
        self.published.append(
            PublishedEvent(
                event=event,
                payload_roles=[payload.role for payload in payloads],
            )
        )
        return event


class FakeEventLog:
    def __init__(self) -> None:
        self.publisher: FakePublisher | None = None

    def with_event_context(self, context: EventContext) -> FakePublisher:
        self.publisher = FakePublisher(context)
        return self.publisher


class FakeExecutor:
    def __init__(
        self,
        *,
        response: LlmResponse | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.response = response
        self.exc = exc
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        return self.response or _response(request)


class FakeMessage:
    def __init__(
        self, work: QueuedWorkMessage, *, num_delivered: int = 2
    ) -> None:
        self.data = work.json_bytes()
        self.metadata = SimpleNamespace(num_delivered=num_delivered)
        self.acked = False
        self.naked = False

    async def ack(self) -> None:
        self.acked = True

    async def nak(self) -> None:
        self.naked = True


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hi")],
    )


def _response(request: LlmRequest) -> LlmResponse:
    return LlmResponse(
        text="done",
        provider=str(request.provider),
        model=request.model,
        mode=request.mode,
    )


def _work(*, max_retries: int = 0) -> QueuedWorkMessage:
    return QueuedWorkMessage(
        work_id="work-1",
        request=_request(),
        run_id="run-1",
        correlation_id="corr-1",
        source="test-source",
        max_retries=max_retries,
    )


def _handler(
    *,
    event_log: FakeEventLog,
    executor: FakeExecutor,
) -> StreamingWorkMessageHandler:
    processor = StreamingWorkProcessor(executor=executor)
    return StreamingWorkMessageHandler(
        event_log=cast(StreamingEventLog, event_log),
        worker_id="worker-1",
        processor=processor,
    )


def _event_types(publisher: FakePublisher) -> list[StreamingLogEventType]:
    return [event.event_type for event in publisher.events]


def _published(
    publisher: FakePublisher, event_type: StreamingLogEventType
) -> PublishedEvent:
    for record in publisher.published:
        if record.event.event_type is event_type:
            return record
    raise AssertionError(f"missing event {event_type}")


def test_attempt_uses_delivery_metadata_and_work_context() -> None:
    work = _work()
    msg = FakeMessage(work, num_delivered=3)

    attempt = StreamingWorkAttempt.from_message(msg=msg, worker_id="worker-1")

    assert attempt.work == work
    assert attempt.worker_id == "worker-1"
    assert attempt.attempt == 3
    assert attempt.attempt_id == "work-1-3"
    assert attempt.event_context() == EventContext.from_work_attempt(
        work, attempt_id="work-1-3"
    )


def test_success_path_emits_lifecycle_payloads_and_acks() -> None:
    work = _work()
    event_log = FakeEventLog()
    msg = FakeMessage(work, num_delivered=2)

    outcome = asyncio.run(
        _handler(event_log=event_log, executor=FakeExecutor()).process_message(
            msg
        )
    )

    assert outcome.outcome_type is StreamingWorkOutcomeType.succeeded
    assert msg.acked
    assert not msg.naked
    assert event_log.publisher is not None
    assert _event_types(event_log.publisher) == [
        StreamingLogEventType.attempt_started,
        StreamingLogEventType.provider_request_prepared,
        StreamingLogEventType.provider_response_received,
        StreamingLogEventType.attempt_succeeded,
        StreamingLogEventType.work_completed,
    ]
    assert _published(
        event_log.publisher, StreamingLogEventType.provider_request_prepared
    ).payload_roles == ["request_json"]
    assert _published(
        event_log.publisher, StreamingLogEventType.provider_response_received
    ).payload_roles == ["response_json"]
    for event in event_log.publisher.events:
        assert event.run_id == "run-1"
        assert event.work_id == "work-1"
        assert event.attempt_id == "work-1-2"
        assert event.correlation_id == "corr-1"
        assert event.source == "test-source"


def test_retryable_failure_emits_retry_event_and_naks() -> None:
    work = _work(max_retries=2)
    event_log = FakeEventLog()
    msg = FakeMessage(work, num_delivered=1)

    outcome = asyncio.run(
        _handler(
            event_log=event_log,
            executor=FakeExecutor(exc=RuntimeError("provider down")),
        ).process_message(msg)
    )

    assert outcome.outcome_type is StreamingWorkOutcomeType.retry_scheduled
    assert outcome.next_attempt == 2
    assert not msg.acked
    assert msg.naked
    assert event_log.publisher is not None
    assert _event_types(event_log.publisher) == [
        StreamingLogEventType.attempt_started,
        StreamingLogEventType.provider_request_prepared,
        StreamingLogEventType.attempt_failed,
        StreamingLogEventType.work_retry_scheduled,
    ]
    failed = _published(
        event_log.publisher, StreamingLogEventType.attempt_failed
    )
    assert failed.payload_roles == ["error_detail"]
    assert failed.event.payload == {
        "error_type": "RuntimeError",
        "message": "provider down",
        "attempt": 1,
    }
    retry = _published(
        event_log.publisher, StreamingLogEventType.work_retry_scheduled
    )
    assert retry.event.payload == {"attempt": 1, "next_attempt": 2}


def test_terminal_failure_emits_completion_event_and_acks() -> None:
    work = _work(max_retries=1)
    event_log = FakeEventLog()
    msg = FakeMessage(work, num_delivered=2)

    outcome = asyncio.run(
        _handler(
            event_log=event_log,
            executor=FakeExecutor(exc=RuntimeError("still down")),
        ).process_message(msg)
    )

    assert outcome.outcome_type is StreamingWorkOutcomeType.failed
    assert msg.acked
    assert not msg.naked
    assert event_log.publisher is not None
    assert _event_types(event_log.publisher) == [
        StreamingLogEventType.attempt_started,
        StreamingLogEventType.provider_request_prepared,
        StreamingLogEventType.attempt_failed,
        StreamingLogEventType.work_completed,
    ]
    completed = _published(
        event_log.publisher, StreamingLogEventType.work_completed
    )
    assert completed.event.payload == {
        "status": StreamingWorkOutcomeType.failed,
        "error_type": "RuntimeError",
        "message": "still down",
        "attempt": 2,
    }


def test_retry_policy_retries_through_configured_max_retries() -> None:
    policy = StreamingRetryPolicy()
    work = _work(max_retries=1)
    first = StreamingWorkAttempt(
        work=work,
        worker_id="worker-1",
        attempt=1,
        attempt_id="work-1-1",
    )
    second = StreamingWorkAttempt(
        work=work,
        worker_id="worker-1",
        attempt=2,
        attempt_id="work-1-2",
    )

    retry = policy.failure_outcome(attempt=first, exc=RuntimeError("retry me"))
    failure = policy.failure_outcome(attempt=second, exc=RuntimeError("stop"))

    assert retry.outcome_type is StreamingWorkOutcomeType.retry_scheduled
    assert retry.next_attempt == 2
    assert failure.outcome_type is StreamingWorkOutcomeType.failed

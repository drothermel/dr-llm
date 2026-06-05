from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dr_llm.llm import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    ProviderName,
)
from dr_llm.streaming_log.client import StreamingEventLog
from dr_llm.streaming_log.events import EventContext, StreamingLogEventType
from dr_llm.streaming_log.work import QueuedWorkMessage
from dr_llm.streaming_log.workers import (
    StreamingRetryPolicy,
    StreamingWorkAttempt,
    StreamingWorkFailed,
    StreamingWorkMessageHandler,
    StreamingWorkOutcome,
    StreamingWorkOutcomeType,
    StreamingWorkProcessor,
    StreamingWorkRetryScheduled,
    StreamingWorkSucceeded,
)
from tests.streaming_log.helpers import (
    PublishCall,
    SpyStreamingEventLog,
    event_types,
    published_call,
)


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
    event_log: SpyStreamingEventLog,
    executor: FakeExecutor,
) -> StreamingWorkMessageHandler:
    processor = StreamingWorkProcessor(executor=executor)
    return StreamingWorkMessageHandler(
        event_log=cast(StreamingEventLog, event_log),
        worker_id="worker-1",
        processor=processor,
    )


def _process_message(
    *,
    work: QueuedWorkMessage | None = None,
    executor: FakeExecutor | None = None,
    num_delivered: int = 2,
) -> tuple[SpyStreamingEventLog, FakeMessage, StreamingWorkOutcome]:
    event_log = SpyStreamingEventLog()
    msg = FakeMessage(work or _work(), num_delivered=num_delivered)

    outcome = asyncio.run(
        _handler(
            event_log=event_log,
            executor=executor or FakeExecutor(),
        ).process_message(msg)
    )

    return event_log, msg, outcome


def _published_messages(event_log: SpyStreamingEventLog) -> list[PublishCall]:
    assert len(event_log.contextual_publishers) == 1
    return event_log.contextual_publishers[0].published


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


def test_success_path_returns_succeeded_and_acks() -> None:
    _, msg, outcome = _process_message()

    assert isinstance(outcome, StreamingWorkSucceeded)
    assert outcome.outcome_type is StreamingWorkOutcomeType.succeeded
    assert msg.acked
    assert not msg.naked


def test_success_path_emits_lifecycle_in_order() -> None:
    event_log, _, _ = _process_message()

    assert event_types(_published_messages(event_log)) == [
        StreamingLogEventType.attempt_started,
        StreamingLogEventType.provider_request_prepared,
        StreamingLogEventType.provider_response_received,
        StreamingLogEventType.attempt_succeeded,
        StreamingLogEventType.work_completed,
    ]


def test_success_path_records_request_and_response_payload_roles() -> None:
    event_log, _, _ = _process_message()
    calls = _published_messages(event_log)

    assert published_call(
        calls, StreamingLogEventType.provider_request_prepared
    ).payload_roles == ["request_json"]
    assert published_call(
        calls, StreamingLogEventType.provider_response_received
    ).payload_roles == ["response_json"]


def test_success_path_applies_work_context_to_all_events() -> None:
    event_log, _, _ = _process_message()

    for call in _published_messages(event_log):
        assert call.context == EventContext(
            run_id="run-1",
            work_id="work-1",
            attempt_id="work-1-2",
            correlation_id="corr-1",
            source="test-source",
        )


def test_retryable_failure_returns_retry_and_naks() -> None:
    _, msg, outcome = _process_message(
        work=_work(max_retries=2),
        executor=FakeExecutor(exc=RuntimeError("provider down")),
        num_delivered=1,
    )

    assert isinstance(outcome, StreamingWorkRetryScheduled)
    assert outcome.outcome_type is StreamingWorkOutcomeType.retry_scheduled
    assert outcome.next_attempt == 2
    assert not msg.acked
    assert msg.naked


def test_retryable_failure_emits_failure_and_retry_events() -> None:
    event_log, _, _ = _process_message(
        work=_work(max_retries=2),
        executor=FakeExecutor(exc=RuntimeError("provider down")),
        num_delivered=1,
    )
    calls = _published_messages(event_log)

    assert event_types(calls) == [
        StreamingLogEventType.attempt_started,
        StreamingLogEventType.provider_request_prepared,
        StreamingLogEventType.attempt_failed,
        StreamingLogEventType.work_retry_scheduled,
    ]
    failed = published_call(calls, StreamingLogEventType.attempt_failed)
    assert failed.payload_roles == ["error_detail"]
    assert failed.payload.model_dump(mode="json") == {
        "error_type": "RuntimeError",
        "message": "provider down",
        "attempt": 1,
    }
    retry = published_call(calls, StreamingLogEventType.work_retry_scheduled)
    assert retry.payload.model_dump(mode="json") == {
        "attempt": 1,
        "next_attempt": 2,
    }


def test_terminal_failure_returns_failed_and_acks() -> None:
    _, msg, outcome = _process_message(
        work=_work(max_retries=1),
        executor=FakeExecutor(exc=RuntimeError("still down")),
        num_delivered=2,
    )

    assert isinstance(outcome, StreamingWorkFailed)
    assert outcome.outcome_type is StreamingWorkOutcomeType.failed
    assert msg.acked
    assert not msg.naked


def test_terminal_failure_emits_completion_event() -> None:
    event_log, _, _ = _process_message(
        work=_work(max_retries=1),
        executor=FakeExecutor(exc=RuntimeError("still down")),
        num_delivered=2,
    )
    calls = _published_messages(event_log)

    assert event_types(calls) == [
        StreamingLogEventType.attempt_started,
        StreamingLogEventType.provider_request_prepared,
        StreamingLogEventType.attempt_failed,
        StreamingLogEventType.work_completed,
    ]
    completed = published_call(calls, StreamingLogEventType.work_completed)
    assert completed.payload.model_dump(mode="json", exclude_none=True) == {
        "status": "failed",
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

    assert isinstance(retry, StreamingWorkRetryScheduled)
    assert retry.outcome_type is StreamingWorkOutcomeType.retry_scheduled
    assert retry.next_attempt == 2
    assert isinstance(failure, StreamingWorkFailed)
    assert failure.outcome_type is StreamingWorkOutcomeType.failed


def test_retry_policy_terminal_fails_on_final_delivery() -> None:
    policy = StreamingRetryPolicy(max_deliveries=3)
    work = _work(max_retries=3)
    final_delivery = StreamingWorkAttempt(
        work=work,
        worker_id="worker-1",
        attempt=3,
        attempt_id="work-1-3",
    )

    outcome = policy.failure_outcome(
        attempt=final_delivery, exc=RuntimeError("stop")
    )

    assert isinstance(outcome, StreamingWorkFailed)
    assert outcome.outcome_type is StreamingWorkOutcomeType.failed
    assert outcome.attempt == 3


def test_success_outcome_requires_response() -> None:
    model = cast(Any, StreamingWorkSucceeded)

    with pytest.raises(ValidationError):
        model(attempt=1)


def test_success_outcome_has_no_error_payload_api() -> None:
    outcome = StreamingWorkSucceeded(attempt=1, response=_response(_request()))

    assert outcome.attempt == 1
    assert outcome.response == _response(_request())
    assert not hasattr(outcome, "error_payload")


def test_retry_outcome_requires_next_attempt() -> None:
    model = cast(Any, StreamingWorkRetryScheduled)

    with pytest.raises(ValidationError):
        model(
            attempt=1,
            error_type="RuntimeError",
            error_message="provider down",
        )


def test_failed_outcome_rejects_next_attempt() -> None:
    model = cast(Any, StreamingWorkFailed)

    with pytest.raises(ValidationError):
        model(
            attempt=1,
            error_type="RuntimeError",
            error_message="stop",
            next_attempt=2,
        )

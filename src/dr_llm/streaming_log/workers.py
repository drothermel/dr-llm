from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dr_llm.llm import (
    LlmRequest,
    LlmResponse,
    ProviderRegistry,
    build_default_registry,
)
from dr_llm.streaming_log.client import (
    StreamingEventLog,
    StreamingEventPublisher,
    StreamingLogConnection,
    StreamingPayloadStore,
    StreamingWorkQueue,
)
from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.events import (
    EventContext,
    StreamingLogEventType,
    idempotency_key,
)
from dr_llm.streaming_log.payloads import prepare_json_payload
from dr_llm.streaming_log.work import QueuedWorkMessage


class StreamingWorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: str = Field(default_factory=lambda: f"stream-{uuid4().hex[:8]}")
    max_messages: int | None = Field(default=None, ge=1)
    fetch_timeout_seconds: float = Field(default=1.0, gt=0)


class StreamingWorkOutcomeType(StrEnum):
    succeeded = "succeeded"
    retry_scheduled = "retry_scheduled"
    failed = "failed"


class StreamingWorkAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    work: QueuedWorkMessage
    worker_id: str
    attempt: int = Field(ge=1)
    attempt_id: str

    @classmethod
    def from_message(cls, *, msg: Any, worker_id: str) -> Self:
        work = QueuedWorkMessage.from_payload(msg.data)
        attempt = int(getattr(msg.metadata, "num_delivered", 1))
        return cls(
            work=work,
            worker_id=worker_id,
            attempt=attempt,
            attempt_id=cls.attempt_id_for(work=work, attempt=attempt),
        )

    @staticmethod
    def attempt_id_for(*, work: QueuedWorkMessage, attempt: int) -> str:
        return f"{work.work_id}-{attempt}"

    def event_context(self) -> EventContext:
        return EventContext.from_work_attempt(
            self.work, attempt_id=self.attempt_id
        )


class StreamingWorkOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_type: StreamingWorkOutcomeType
    attempt: int = Field(ge=1)
    response: LlmResponse | None = None
    error_type: str | None = None
    error_message: str | None = None
    next_attempt: int | None = Field(default=None, ge=2)

    @classmethod
    def succeeded(cls, *, attempt: int, response: LlmResponse) -> Self:
        return cls(
            outcome_type=StreamingWorkOutcomeType.succeeded,
            attempt=attempt,
            response=response,
        )

    @classmethod
    def retry_scheduled(
        cls, *, attempt: int, exc: Exception, next_attempt: int
    ) -> Self:
        return cls(
            outcome_type=StreamingWorkOutcomeType.retry_scheduled,
            attempt=attempt,
            error_type=type(exc).__name__,
            error_message=str(exc),
            next_attempt=next_attempt,
        )

    @classmethod
    def failed(cls, *, attempt: int, exc: Exception) -> Self:
        return cls(
            outcome_type=StreamingWorkOutcomeType.failed,
            attempt=attempt,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    @model_validator(mode="after")
    def _validate_outcome_payload(self) -> Self:
        if self.outcome_type is StreamingWorkOutcomeType.succeeded:
            if self.response is None:
                raise ValueError("succeeded outcomes require a response")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("succeeded outcomes cannot include errors")
            if self.next_attempt is not None:
                raise ValueError("succeeded outcomes cannot schedule retries")
            return self
        if self.response is not None:
            raise ValueError("failure outcomes cannot include a response")
        if self.error_type is None or self.error_message is None:
            raise ValueError("failure outcomes require error details")
        if (
            self.outcome_type is StreamingWorkOutcomeType.retry_scheduled
            and self.next_attempt is None
        ):
            raise ValueError("retry outcomes require next_attempt")
        if (
            self.outcome_type is StreamingWorkOutcomeType.failed
            and self.next_attempt is not None
        ):
            raise ValueError("failed outcomes cannot schedule retries")
        return self

    def error_payload(self) -> dict[str, Any]:
        if self.error_type is None or self.error_message is None:
            raise ValueError("outcome has no error payload")
        return {
            "error_type": self.error_type,
            "message": self.error_message,
            "attempt": self.attempt,
        }


class StreamingWorkExecutor(Protocol):
    async def generate(self, request: LlmRequest) -> LlmResponse: ...


class ProviderRegistryStreamingWorkExecutor:
    def __init__(self, registry: ProviderRegistry) -> None:
        self.registry = registry

    async def generate(self, request: LlmRequest) -> LlmResponse:
        orchestrator = self.registry.get(str(request.provider))
        return await asyncio.to_thread(orchestrator.generate, request)


class StreamingRetryPolicy:
    def failure_outcome(
        self, *, attempt: StreamingWorkAttempt, exc: Exception
    ) -> StreamingWorkOutcome:
        if attempt.attempt <= attempt.work.max_retries:
            return StreamingWorkOutcome.retry_scheduled(
                attempt=attempt.attempt,
                exc=exc,
                next_attempt=attempt.attempt + 1,
            )
        return StreamingWorkOutcome.failed(attempt=attempt.attempt, exc=exc)


class StreamingWorkLifecycleReporter:
    def __init__(
        self,
        *,
        publisher: StreamingEventPublisher,
        attempt: StreamingWorkAttempt,
    ) -> None:
        self.publisher = publisher
        self.attempt = attempt

    @classmethod
    def from_event_log(
        cls, *, event_log: StreamingEventLog, attempt: StreamingWorkAttempt
    ) -> Self:
        return cls(
            publisher=event_log.with_event_context(attempt.event_context()),
            attempt=attempt,
        )

    async def record_attempt_started(self) -> None:
        await self.publisher.publish_event_with_payloads(
            StreamingLogEventType.attempt_started,
            idempotency_key=self.idempotency_key_for("attempt_started"),
            payload={
                "worker_id": self.attempt.worker_id,
                "attempt": self.attempt.attempt,
            },
        )

    async def record_provider_request_prepared(self) -> None:
        request = self.attempt.work.request
        request_payload = request.model_dump(
            mode="json",
            exclude_none=True,
            exclude_computed_fields=True,
        )
        await self.publisher.publish_event_with_payloads(
            StreamingLogEventType.provider_request_prepared,
            idempotency_key=self.idempotency_key_for(
                "provider_request_prepared"
            ),
            payload={
                "provider": request.provider,
                "model": request.model,
                "mode": request.mode,
            },
            payloads=[prepare_json_payload("request_json", request_payload)],
        )

    async def record_success(self, outcome: StreamingWorkOutcome) -> None:
        response = self._response_for(outcome)
        response_payload = response.model_dump(
            mode="json",
            exclude_none=True,
            exclude_computed_fields=True,
        )
        await self.publisher.publish_event_with_payloads(
            StreamingLogEventType.provider_response_received,
            idempotency_key=self.idempotency_key_for(
                "provider_response_received"
            ),
            payload={
                "provider": response.provider,
                "model": response.model,
                "mode": response.mode,
                "finish_reason": response.finish_reason,
            },
            payloads=[prepare_json_payload("response_json", response_payload)],
        )
        await self.publisher.publish_event_with_payloads(
            StreamingLogEventType.attempt_succeeded,
            idempotency_key=self.idempotency_key_for("attempt_succeeded"),
            payload={"attempt": self.attempt.attempt},
        )
        await self.publisher.publish_event_with_payloads(
            StreamingLogEventType.work_completed,
            idempotency_key=idempotency_key(
                "work_completed", self.attempt.work.work_id
            ),
            payload={
                "status": StreamingWorkOutcomeType.succeeded,
                "attempt": self.attempt.attempt,
            },
        )

    async def record_failure(self, outcome: StreamingWorkOutcome) -> None:
        error_payload = outcome.error_payload()
        await self.publisher.publish_event_with_payloads(
            StreamingLogEventType.attempt_failed,
            idempotency_key=self.idempotency_key_for("attempt_failed"),
            payload=error_payload,
            payloads=[prepare_json_payload("error_detail", error_payload)],
        )
        if outcome.outcome_type is StreamingWorkOutcomeType.retry_scheduled:
            await self.record_retry_scheduled(outcome)
            return
        await self.record_work_failed(error_payload)

    async def record_retry_scheduled(
        self, outcome: StreamingWorkOutcome
    ) -> None:
        await self.publisher.publish_event_with_payloads(
            StreamingLogEventType.work_retry_scheduled,
            idempotency_key=self.idempotency_key_for("work_retry_scheduled"),
            payload={
                "attempt": self.attempt.attempt,
                "next_attempt": outcome.next_attempt,
            },
        )

    async def record_work_failed(self, error_payload: dict[str, Any]) -> None:
        await self.publisher.publish_event_with_payloads(
            StreamingLogEventType.work_completed,
            idempotency_key=idempotency_key(
                "work_completed", self.attempt.work.work_id
            ),
            payload={
                "status": StreamingWorkOutcomeType.failed,
                **error_payload,
            },
        )

    def idempotency_key_for(self, event_name: str) -> str:
        return idempotency_key(
            event_name, self.attempt.work.work_id, self.attempt.attempt
        )

    @staticmethod
    def _response_for(outcome: StreamingWorkOutcome) -> LlmResponse:
        if outcome.response is None:
            raise ValueError("outcome has no response")
        return outcome.response


class StreamingMessageAcknowledger:
    async def apply(self, *, msg: Any, outcome: StreamingWorkOutcome) -> None:
        if outcome.outcome_type is StreamingWorkOutcomeType.retry_scheduled:
            await msg.nak()
            return
        await msg.ack()


class StreamingWorkProcessor:
    def __init__(
        self,
        *,
        executor: StreamingWorkExecutor,
        retry_policy: StreamingRetryPolicy | None = None,
    ) -> None:
        self.executor = executor
        self.retry_policy = retry_policy or StreamingRetryPolicy()

    async def process(
        self,
        *,
        attempt: StreamingWorkAttempt,
        reporter: StreamingWorkLifecycleReporter,
    ) -> StreamingWorkOutcome:
        await reporter.record_attempt_started()
        await reporter.record_provider_request_prepared()
        try:
            response = await self.executor.generate(attempt.work.request)
        except Exception as exc:  # noqa: BLE001
            outcome = self.retry_policy.failure_outcome(
                attempt=attempt, exc=exc
            )
            await reporter.record_failure(outcome)
            return outcome
        outcome = StreamingWorkOutcome.succeeded(
            attempt=attempt.attempt, response=response
        )
        await reporter.record_success(outcome)
        return outcome


class StreamingWorkMessageHandler:
    def __init__(
        self,
        *,
        event_log: StreamingEventLog,
        worker_id: str,
        processor: StreamingWorkProcessor,
        acknowledger: StreamingMessageAcknowledger | None = None,
    ) -> None:
        self.event_log = event_log
        self.worker_id = worker_id
        self.processor = processor
        self.acknowledger = acknowledger or StreamingMessageAcknowledger()

    async def process_message(self, msg: Any) -> StreamingWorkOutcome:
        attempt = StreamingWorkAttempt.from_message(
            msg=msg, worker_id=self.worker_id
        )
        reporter = StreamingWorkLifecycleReporter.from_event_log(
            event_log=self.event_log, attempt=attempt
        )
        outcome = await self.processor.process(
            attempt=attempt, reporter=reporter
        )
        await self.acknowledger.apply(msg=msg, outcome=outcome)
        return outcome


async def run_streaming_worker(
    *,
    work_queue: StreamingWorkQueue | None = None,
    config: StreamingWorkerConfig | None = None,
    registry_factory: Callable[[], ProviderRegistry] = build_default_registry,
) -> None:
    config = config or StreamingWorkerConfig()
    work_queue, owned_connection = _worker_queue(work_queue)
    event_log = work_queue.event_log
    registry: ProviderRegistry | None = None
    try:
        if owned_connection is not None:
            await owned_connection.connect()
        registry = registry_factory()
        handler = _message_handler(
            event_log=event_log,
            worker_id=config.worker_id,
            registry=registry,
        )
        await _run_worker_session(
            work_queue=work_queue,
            event_log=event_log,
            config=config,
            handler=handler,
        )
    finally:
        if registry is not None:
            registry.close()
        if owned_connection is not None:
            await owned_connection.close()


def _worker_queue(
    work_queue: StreamingWorkQueue | None,
) -> tuple[StreamingWorkQueue, StreamingLogConnection | None]:
    if work_queue is not None:
        return work_queue, None
    connection = StreamingLogConnection(StreamingLogConfig())
    payload_store = StreamingPayloadStore(connection)
    event_log = StreamingEventLog(connection, payload_store)
    return StreamingWorkQueue(connection, event_log), connection


def _message_handler(
    *,
    event_log: StreamingEventLog,
    worker_id: str,
    registry: ProviderRegistry,
) -> StreamingWorkMessageHandler:
    executor = ProviderRegistryStreamingWorkExecutor(registry)
    processor = StreamingWorkProcessor(executor=executor)
    return StreamingWorkMessageHandler(
        event_log=event_log, worker_id=worker_id, processor=processor
    )


async def _run_worker_session(
    *,
    work_queue: StreamingWorkQueue,
    event_log: StreamingEventLog,
    config: StreamingWorkerConfig,
    handler: StreamingWorkMessageHandler,
) -> None:
    processed = 0
    try:
        await _publish_producer_started(
            event_log=event_log, worker_id=config.worker_id
        )
        sub = await work_queue.work_subscription()
        while _should_process_more(config=config, processed=processed):
            processed = await _process_next_batch(
                sub=sub,
                work_queue=work_queue,
                config=config,
                handler=handler,
                processed=processed,
            )
    finally:
        await _publish_producer_stopped(
            event_log=event_log, worker_id=config.worker_id
        )


async def _publish_producer_started(
    *, event_log: StreamingEventLog, worker_id: str
) -> None:
    await event_log.publish_event_with_payloads(
        StreamingLogEventType.producer_started,
        idempotency_key=idempotency_key("producer_started", worker_id),
        payload={"worker_id": worker_id},
    )


async def _publish_producer_stopped(
    *, event_log: StreamingEventLog, worker_id: str
) -> None:
    await event_log.publish_event_with_payloads(
        StreamingLogEventType.producer_stopped,
        idempotency_key=idempotency_key("producer_stopped", worker_id),
        payload={"worker_id": worker_id},
    )


def _should_process_more(
    *, config: StreamingWorkerConfig, processed: int
) -> bool:
    return config.max_messages is None or processed < config.max_messages


async def _process_next_batch(
    *,
    sub: Any,
    work_queue: StreamingWorkQueue,
    config: StreamingWorkerConfig,
    handler: StreamingWorkMessageHandler,
    processed: int,
) -> int:
    messages = await _fetch_messages(
        sub=sub, work_queue=work_queue, config=config
    )
    for msg in messages:
        await handler.process_message(msg)
        processed += 1
        if not _should_process_more(config=config, processed=processed):
            break
    return processed


async def _fetch_messages(
    *, sub: Any, work_queue: StreamingWorkQueue, config: StreamingWorkerConfig
) -> list[Any]:
    try:
        return await sub.fetch(
            work_queue.config.fetch_batch_size,
            timeout=config.fetch_timeout_seconds,
        )
    except TimeoutError:
        return []


__all__ = [
    "ProviderRegistryStreamingWorkExecutor",
    "StreamingMessageAcknowledger",
    "StreamingRetryPolicy",
    "StreamingWorkAttempt",
    "StreamingWorkExecutor",
    "StreamingWorkLifecycleReporter",
    "StreamingWorkMessageHandler",
    "StreamingWorkOutcome",
    "StreamingWorkOutcomeType",
    "StreamingWorkProcessor",
    "StreamingWorkerConfig",
    "run_streaming_worker",
]

from __future__ import annotations

from typing import Any, cast

from nats.js.errors import ObjectNotFoundError
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.streaming_log.client import StreamingLogConnection
from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.event_builders import StreamingEventPublishSpec
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
)


class PublishCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: StreamingLogEventType
    idempotency_key: str
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_roles: list[str] = Field(default_factory=list)
    context: EventContext | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    event: EventEnvelope


class PublishedMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str
    payload: bytes
    stream: str | None = None


class FakeObjectResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data: bytes | None


class FakeObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes | None] = {}

    async def get(self, name: str) -> FakeObjectResult:
        if name not in self.objects:
            raise ObjectNotFoundError
        return FakeObjectResult(data=self.objects[name])

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data


class FakeJetStream:
    def __init__(self) -> None:
        self.store = FakeObjectStore()
        self.published: list[PublishedMessage] = []

    async def object_store(self, bucket: str) -> FakeObjectStore:
        if bucket != "DRLLM_PAYLOADS":
            raise AssertionError(f"unexpected payload bucket {bucket!r}")
        return self.store

    async def publish(
        self, subject: str, payload: bytes, *, stream: str | None = None
    ) -> object:
        self.published.append(
            PublishedMessage(subject=subject, payload=payload, stream=stream)
        )
        return object()


class FakeStreamingLogConnection(StreamingLogConnection):
    def __init__(self, fake_js: FakeJetStream | None = None) -> None:
        super().__init__(StreamingLogConfig())
        self.fake_js = fake_js or FakeJetStream()

    @property
    def js(self) -> Any:
        return self.fake_js


class SpyEventPublisher:
    def __init__(self, context: EventContext | None = None) -> None:
        self.context = context
        self.published: list[PublishCall] = []

    @property
    def events(self) -> list[EventEnvelope]:
        return [call.event for call in self.published]

    async def publish_event_spec(
        self, spec: StreamingEventPublishSpec
    ) -> EventEnvelope:
        if self.context is None:
            return self.record_publish(spec)
        return self.record_publish(
            spec.model_copy(update={"context": self.context})
        )

    def record_publish(self, spec: StreamingEventPublishSpec) -> EventEnvelope:
        event = minimal_event(
            spec.event_type,
            idempotency_key=spec.idempotency_key,
            payload=spec.payload,
            context=spec.context,
            metadata=spec.metadata,
        )
        self.published.append(
            PublishCall(
                event_type=spec.event_type,
                idempotency_key=spec.idempotency_key,
                payload=spec.payload,
                payload_roles=[
                    payload_item.role for payload_item in spec.payloads
                ],
                context=spec.context,
                metadata=spec.metadata,
                event=event,
            )
        )
        return event


class SpyStreamingEventLog:
    def __init__(self, fail_on: StreamingLogEventType | None = None) -> None:
        self.fail_on = fail_on
        self.publisher = SpyEventPublisher()
        self.contextual_publishers: list[SpyEventPublisher] = []

    @property
    def published(self) -> list[PublishCall]:
        calls = list(self.publisher.published)
        for publisher in self.contextual_publishers:
            calls.extend(publisher.published)
        return calls

    @property
    def events(self) -> list[EventEnvelope]:
        return [call.event for call in self.published]

    async def publish_event_spec(
        self, spec: StreamingEventPublishSpec
    ) -> EventEnvelope:
        if spec.event_type == self.fail_on:
            raise RuntimeError(f"failed to publish {spec.event_type}")
        return self.publisher.record_publish(spec)

    def with_event_context(self, context: EventContext) -> SpyEventPublisher:
        publisher = SpyEventPublisher(context)
        self.contextual_publishers.append(publisher)
        return publisher


def minimal_event(
    event_type: StreamingLogEventType,
    *,
    idempotency_key: str,
    payload: dict[str, Any] | None = None,
    context: EventContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> EventEnvelope:
    context = context or EventContext()
    return EventEnvelope(
        event_type=event_type,
        producer=ProducerInfo(name="test"),
        idempotency_key=idempotency_key,
        payload=payload or {},
        run_id=context.run_id,
        work_id=context.work_id,
        attempt_id=context.attempt_id,
        causation_id=context.causation_id,
        correlation_id=context.correlation_id,
        source=context.source,
        metadata=metadata or {},
    )


def event_types(
    calls_or_events: list[PublishCall] | list[EventEnvelope],
) -> list[StreamingLogEventType]:
    return [
        item.event_type
        for item in cast(list[PublishCall | EventEnvelope], calls_or_events)
    ]


def published_call(
    calls: list[PublishCall], event_type: StreamingLogEventType
) -> PublishCall:
    for call in calls:
        if call.event_type is event_type:
            return call
    raise AssertionError(f"missing event {event_type}")

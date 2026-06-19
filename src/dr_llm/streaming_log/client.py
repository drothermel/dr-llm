from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Protocol, Self

import nats
from nats.aio.client import Client as NatsClient
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.client import JetStreamContext as LegacyJetStreamContext
from nats.js.errors import NotFoundError, ObjectNotFoundError

from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.errors import (
    PayloadIntegrityError,
    PayloadNotFoundError,
    PayloadReadError,
    StreamingLogError,
)
from dr_llm.streaming_log.event_builders import (
    StreamingEventPublishSpec,
    work_submitted_event,
)
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
    build_event,
)
from dr_llm.streaming_log.payloads import (
    PayloadRef,
    PreparedPayload,
    sha256_bytes,
)
from dr_llm.streaming_log.work import QueuedWorkMessage


class StreamingLogConnection:
    def __init__(self, config: StreamingLogConfig | None = None) -> None:
        self.config = config or StreamingLogConfig()
        self._nc: NatsClient | None = None
        self._js: JetStreamContext | LegacyJetStreamContext | None = None

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        await self.close()

    async def connect(self) -> None:
        if self._nc is not None:
            return
        self._nc = await nats.connect(self.config.nats_url)
        self._js = self._nc.jetstream()

    async def close(self) -> None:
        if self._nc is None:
            return
        await self._nc.close()
        self._nc = None
        self._js = None

    @property
    def js(self) -> JetStreamContext | LegacyJetStreamContext:
        if self._js is None:
            raise RuntimeError("StreamingLogConnection is not connected")
        return self._js


class StreamingPayloadReader(Protocol):
    async def read_payload_ref(self, payload_ref: PayloadRef) -> bytes: ...


class StreamingPayloadStore:
    def __init__(self, connection: StreamingLogConnection) -> None:
        self.connection = connection

    @property
    def config(self) -> StreamingLogConfig:
        return self.connection.config

    async def write_payloads(
        self, payloads: list[PreparedPayload]
    ) -> list[PayloadRef]:
        return await asyncio.gather(
            *(self.write_payload(payload) for payload in payloads)
        )

    async def write_payload(self, payload: PreparedPayload) -> PayloadRef:
        store = await self.connection.js.object_store(
            self.config.payload_bucket
        )
        try:
            existing = await store.get(payload.object_key)
        except (NotFoundError, ObjectNotFoundError):
            await store.put(payload.object_key, payload.data)
            return payload.ref()
        if existing.data != payload.data:
            raise PayloadIntegrityError(
                f"Object {payload.object_key!r} exists with different bytes"
            )
        return payload.ref()

    async def read_payload_ref(self, payload_ref: PayloadRef) -> bytes:
        try:
            store = await self.connection.js.object_store(
                self.config.payload_bucket
            )
            result = await store.get(payload_ref.object_key)
        except (NotFoundError, ObjectNotFoundError) as exc:
            raise PayloadNotFoundError(
                f"Object {payload_ref.object_key!r} was not found"
            ) from exc
        except StreamingLogError:
            raise
        except Exception as exc:
            raise PayloadReadError(
                f"Object {payload_ref.object_key!r} could not be read"
            ) from exc
        if result.data is None:
            raise PayloadIntegrityError(
                f"Object {payload_ref.object_key!r} returned no bytes"
            )
        data = bytes(result.data)
        _validate_payload_ref_bytes(payload_ref, data)
        return data


class StreamingEventPublisher(Protocol):
    async def publish_event_spec(
        self, spec: StreamingEventPublishSpec
    ) -> EventEnvelope: ...


class ContextualEventPublisher:
    def __init__(
        self, event_log: StreamingEventLog, context: EventContext
    ) -> None:
        self.event_log = event_log
        self.context = context

    async def publish_event_spec(
        self, spec: StreamingEventPublishSpec
    ) -> EventEnvelope:
        return await self.event_log.publish_event_spec(
            spec.model_copy(update={"context": self.context})
        )


class StreamingEventLog:
    def __init__(
        self,
        connection: StreamingLogConnection,
        payload_store: StreamingPayloadStore,
        *,
        producer: ProducerInfo | None = None,
    ) -> None:
        self.connection = connection
        self.payload_store = payload_store
        self.producer = producer or ProducerInfo(
            name=self.config.producer_name,
            version=self.config.producer_version,
        )

    @property
    def config(self) -> StreamingLogConfig:
        return self.connection.config

    async def publish_event(self, event: EventEnvelope) -> Any:
        return await self.connection.js.publish(
            self.event_subject(event.event_type),
            event.json_bytes(),
            stream=self.config.events_stream,
        )

    async def publish_event_spec(
        self, spec: StreamingEventPublishSpec
    ) -> EventEnvelope:
        payload_refs = await self.payload_store.write_payloads(spec.payloads)
        event = build_event(
            spec.event_type,
            producer=self.producer,
            idempotency_key=spec.idempotency_key,
            payload=spec.payload,
            payload_refs=payload_refs,
            context=spec.context,
            metadata=spec.metadata,
        )
        await self.publish_event(event)
        return event

    def with_event_context(
        self, context: EventContext
    ) -> ContextualEventPublisher:
        return ContextualEventPublisher(self, context)

    async def event_subscription(self, durable: str | None = None) -> Any:
        return await self.connection.js.pull_subscribe(
            self.event_subject_wildcard(),
            durable=durable or self.config.event_consumer,
            stream=self.config.events_stream,
        )

    async def replay_events(
        self,
        *,
        durable: str | None = None,
        batch_size: int | None = None,
        timeout: float = 1.0,
    ) -> AsyncIterator[EventEnvelope]:
        sub = await self.event_subscription(durable)
        messages = await sub.fetch(
            batch_size or self.config.fetch_batch_size,
            timeout=timeout,
        )
        for msg in messages:
            yield EventEnvelope.model_validate_json(msg.data)
            await msg.ack()

    def event_subject(self, event_type: StreamingLogEventType) -> str:
        return (
            f"{self._subject_prefix(self.config.events_subject)}{event_type}"
        )

    def event_subject_wildcard(self) -> str:
        return self.config.events_subject

    def _subject_prefix(self, wildcard: str) -> str:
        if not wildcard.endswith(">"):
            raise ValueError(
                f"expected wildcard subject ending in '>': {wildcard}"
            )
        return wildcard[:-1]


class StreamingWorkQueue:
    def __init__(
        self,
        connection: StreamingLogConnection,
        event_log: StreamingEventLog,
    ) -> None:
        self.connection = connection
        self.event_log = event_log

    @property
    def config(self) -> StreamingLogConfig:
        return self.connection.config

    async def submit_work(self, work: QueuedWorkMessage) -> EventEnvelope:
        if work.max_retries >= self.config.max_deliver:
            raise ValueError(
                "work.max_retries must be less than the streaming-log "
                f"max_deliver setting ({self.config.max_deliver})"
            )
        event = await self.event_log.publish_event_spec(
            work_submitted_event(work)
        )
        await self.connection.js.publish(
            self.config.llm_work_subject,
            work.json_bytes(),
            stream=self.config.work_stream,
        )
        return event

    async def work_subscription(self) -> Any:
        return await self.connection.js.pull_subscribe(
            self.config.llm_work_subject,
            durable=self.config.work_consumer,
            stream=self.config.work_stream,
        )

    async def fetch_work(
        self, *, batch_size: int | None = None, timeout: float = 1.0
    ) -> list[Msg]:
        sub = await self.work_subscription()
        return await sub.fetch(
            batch_size or self.config.fetch_batch_size,
            timeout=timeout,
        )


def _validate_payload_ref_bytes(payload_ref: PayloadRef, data: bytes) -> None:
    if len(data) != payload_ref.size_bytes:
        raise PayloadIntegrityError(
            f"Object {payload_ref.object_key!r} size mismatch: "
            f"{len(data)} != {payload_ref.size_bytes}"
        )
    digest = sha256_bytes(data)
    if digest != payload_ref.sha256:
        raise PayloadIntegrityError(
            f"Object {payload_ref.object_key!r} sha256 mismatch: "
            f"{digest} != {payload_ref.sha256}"
        )


__all__ = [
    "ContextualEventPublisher",
    "StreamingEventLog",
    "StreamingEventPublisher",
    "StreamingLogConnection",
    "StreamingPayloadReader",
    "StreamingPayloadStore",
    "StreamingWorkQueue",
]

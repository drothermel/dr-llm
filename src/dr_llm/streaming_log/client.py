from __future__ import annotations

from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self

import nats
from nats.aio.client import Client as NatsClient
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.client import JetStreamContext as LegacyJetStreamContext
from nats.js.errors import NotFoundError, ObjectNotFoundError

from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.errors import PayloadIntegrityError
from dr_llm.streaming_log.events import (
    EventContext,
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
    build_event,
    idempotency_key,
)
from dr_llm.streaming_log.payloads import (
    PayloadRef,
    PreparedPayload,
    prepare_json_payload,
)
from dr_llm.streaming_log.work import QueuedWorkMessage


class ContextualEventPublisher:
    def __init__(
        self, client: StreamingLogClient, context: EventContext
    ) -> None:
        self.client = client
        self.context = context

    async def publish_event_with_payloads(
        self,
        event_type: StreamingLogEventType,
        *,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        payloads: list[PreparedPayload] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventEnvelope:
        return await self.client.publish_event_with_payloads(
            event_type,
            idempotency_key=idempotency_key,
            payload=payload,
            payloads=payloads,
            context=self.context,
            metadata=metadata,
        )


class StreamingLogClient:
    def __init__(
        self,
        config: StreamingLogConfig | None = None,
        *,
        producer: ProducerInfo | None = None,
    ) -> None:
        self.config = config or StreamingLogConfig()
        self.producer = producer or ProducerInfo(
            name=self.config.producer_name,
            version=self.config.producer_version,
        )
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
            raise RuntimeError("StreamingLogClient is not connected")
        return self._js

    async def publish_event(self, event: EventEnvelope) -> Any:
        return await self.js.publish(
            self.event_subject(event.event_type),
            event.json_bytes(),
            stream=self.config.events_stream,
        )

    async def publish_event_with_payloads(
        self,
        event_type: StreamingLogEventType,
        *,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        payloads: list[PreparedPayload] | None = None,
        context: EventContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventEnvelope:
        payload_refs = await self.write_payloads(payloads or [])
        event = build_event(
            event_type,
            producer=self.producer,
            idempotency_key=idempotency_key,
            payload=payload,
            payload_refs=payload_refs,
            context=context,
            metadata=metadata,
        )
        await self.publish_event(event)
        return event

    def with_event_context(
        self, context: EventContext
    ) -> ContextualEventPublisher:
        return ContextualEventPublisher(self, context)

    async def write_payloads(
        self, payloads: list[PreparedPayload]
    ) -> list[PayloadRef]:
        refs: list[PayloadRef] = []
        for payload in payloads:
            refs.append(await self.write_payload(payload))
        return refs

    async def write_payload(self, payload: PreparedPayload) -> PayloadRef:
        store = await self.js.object_store(self.config.payload_bucket)
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
        store = await self.js.object_store(self.config.payload_bucket)
        result = await store.get(payload_ref.object_key)
        if result.data is None:
            raise PayloadIntegrityError(
                f"Object {payload_ref.object_key!r} returned no bytes"
            )
        return bytes(result.data)

    async def submit_work(self, work: QueuedWorkMessage) -> EventEnvelope:
        request_payload = prepare_json_payload(
            "request_json",
            work.request.model_dump(
                mode="json",
                exclude_none=True,
                exclude_computed_fields=True,
            ),
        )
        event = await self.publish_event_with_payloads(
            StreamingLogEventType.work_submitted,
            idempotency_key=idempotency_key("work_submitted", work.work_id),
            payload={
                "work_id": work.work_id,
                "run_id": work.run_id,
                "max_retries": work.max_retries,
                "metadata": work.metadata,
            },
            payloads=[request_payload],
            context=EventContext.from_work(work),
        )
        await self.js.publish(
            self.config.llm_work_subject,
            work.json_bytes(),
            stream=self.config.work_stream,
        )
        return event

    async def work_subscription(self) -> Any:
        return await self.js.pull_subscribe(
            self.config.llm_work_subject,
            durable=self.config.work_consumer,
            stream=self.config.work_stream,
        )

    async def event_subscription(self, durable: str | None = None) -> Any:
        return await self.js.pull_subscribe(
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
        messages = await sub.fetch(batch_size or self.config.fetch_batch_size)
        for msg in messages:
            yield EventEnvelope.model_validate_json(msg.data)
            await msg.ack()

    async def fetch_work(
        self, *, batch_size: int | None = None, timeout: float = 1.0
    ) -> list[Msg]:
        sub = await self.work_subscription()
        return await sub.fetch(
            batch_size or self.config.fetch_batch_size,
            timeout=timeout,
        )

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


__all__ = ["StreamingLogClient"]

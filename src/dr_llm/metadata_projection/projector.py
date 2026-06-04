from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.mapper import EventFactMapper
from dr_llm.metadata_projection.models import (
    MetadataProjectionCheckpoint,
    MetadataProjectionError,
    MetadataProjectionErrorKind,
    MetadataWritePlan,
)
from dr_llm.metadata_projection.store import MetadataStore
from dr_llm.streaming_log.client import StreamingLogConnection
from dr_llm.streaming_log.events import EventEnvelope


class MetadataEventDelivery(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    event: EventEnvelope
    stream_sequence: int = Field(ge=0)
    message: Any

    async def ack(self) -> None:
        await self.message.ack()


class MetadataProjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    processed_count: int = Field(ge=0)
    error_count: int = Field(ge=0)


class MetadataProjector:
    def __init__(
        self,
        *,
        config: MetadataProjectionConfig,
        store: MetadataStore,
        mapper: EventFactMapper | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.mapper = mapper or EventFactMapper(config)

    async def process_delivery(
        self, delivery: MetadataEventDelivery
    ) -> MetadataProjectionResult:
        plan = self.mapper.map_event(delivery.event)
        self.store.apply_write_plan(
            plan,
            checkpoint=checkpoint_for_delivery(self.config, delivery),
        )
        await delivery.ack()
        return MetadataProjectionResult(
            processed_count=1,
            error_count=len(plan.errors),
        )

    async def process_invalid_message(
        self,
        message: Any,
        *,
        stream_sequence: int,
        exc: Exception,
    ) -> MetadataProjectionResult:
        error = MetadataProjectionError(
            projection_version=self.config.projection_version,
            source_event_id="unknown",
            source_idempotency_key="unknown",
            error_kind=MetadataProjectionErrorKind.invalid_event,
            message=str(exc),
            stream_sequence=stream_sequence,
        )
        self.store.apply_write_plan(
            MetadataWritePlan(errors=[error]),
            checkpoint=MetadataProjectionCheckpoint(
                projection_version=self.config.projection_version,
                durable_consumer=self.config.durable_consumer,
                stream_sequence=stream_sequence,
            ),
        )
        await message.ack()
        return MetadataProjectionResult(processed_count=1, error_count=1)


async def run_metadata_projector(
    *,
    connection: StreamingLogConnection,
    config: MetadataProjectionConfig,
    max_messages: int | None = None,
    batch_size: int | None = None,
    from_start: bool = False,
) -> int:
    store = MetadataStore(config=config)
    store.initialize()
    projector = MetadataProjector(config=config, store=store)
    try:
        return await _consume_events(
            connection=connection,
            config=config,
            projector=projector,
            max_messages=max_messages,
            batch_size=batch_size or config.fetch_batch_size,
            from_start=from_start,
        )
    finally:
        store.close()


async def _consume_events(
    *,
    connection: StreamingLogConnection,
    config: MetadataProjectionConfig,
    projector: MetadataProjector,
    max_messages: int | None,
    batch_size: int,
    from_start: bool,
) -> int:
    sub = await connection.js.pull_subscribe(
        connection.config.events_subject,
        durable=_consumer_name(config, from_start=from_start),
        stream=connection.config.events_stream,
    )
    processed = 0
    while max_messages is None or processed < max_messages:
        messages = await sub.fetch(batch_size, timeout=1)
        for message in messages:
            await _process_message(projector, message)
            processed += 1
            if max_messages is not None and processed >= max_messages:
                break
    return processed


async def _process_message(projector: MetadataProjector, message: Any) -> None:
    stream_sequence = stream_sequence_for_message(message)
    try:
        delivery = MetadataEventDelivery(
            event=EventEnvelope.model_validate_json(message.data),
            stream_sequence=stream_sequence,
            message=message,
        )
    except Exception as exc:  # noqa: BLE001
        await projector.process_invalid_message(
            message, stream_sequence=stream_sequence, exc=exc
        )
        return
    await projector.process_delivery(delivery)


def checkpoint_for_delivery(
    config: MetadataProjectionConfig,
    delivery: MetadataEventDelivery,
) -> MetadataProjectionCheckpoint:
    return MetadataProjectionCheckpoint(
        projection_version=config.projection_version,
        durable_consumer=config.durable_consumer,
        stream_sequence=delivery.stream_sequence,
        event_id=delivery.event.event_id,
    )


def stream_sequence_for_message(message: Any) -> int:
    metadata = getattr(message, "metadata", None)
    sequence = getattr(metadata, "sequence", None)
    stream_sequence = getattr(sequence, "stream", None)
    if stream_sequence is None:
        raise ValueError("message metadata is missing stream sequence")
    return int(stream_sequence)


def _consumer_name(
    config: MetadataProjectionConfig, *, from_start: bool
) -> str:
    if not from_start:
        return config.durable_consumer
    return f"{config.durable_consumer}_replay_{uuid4().hex[:8]}"


__all__ = [
    "MetadataEventDelivery",
    "MetadataProjectionResult",
    "MetadataProjector",
    "checkpoint_for_delivery",
    "run_metadata_projector",
    "stream_sequence_for_message",
]

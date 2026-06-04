from __future__ import annotations

from typing import Any

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
from dr_llm.streaming_log.projection_runtime import (
    ProjectionEventDelivery,
    ProjectionMessageHandler,
    consume_projection_events,
    process_delivery_with_ack,
    process_invalid_message_with_ack,
    stream_sequence_for_message,
)


class MetadataEventDelivery(ProjectionEventDelivery):
    pass


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
        return await process_delivery_with_ack(
            delivery, self._process_delivery
        )

    async def _process_delivery(
        self, delivery: MetadataEventDelivery
    ) -> MetadataProjectionResult:
        plan = self.mapper.map_event(delivery.event)
        self.store.apply_write_plan(
            plan,
            checkpoint=checkpoint_for_delivery(self.config, delivery),
        )
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
        return await process_invalid_message_with_ack(
            message,
            stream_sequence=stream_sequence,
            exc=exc,
            process_invalid_message=self._process_invalid_message,
        )

    async def _process_invalid_message(
        self,
        message: Any,
        *,
        stream_sequence: int,
        exc: Exception,
    ) -> MetadataProjectionResult:
        del message
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
    return await consume_projection_events(
        connection=connection,
        durable_consumer=config.durable_consumer,
        handler=ProjectionMessageHandler(
            delivery_type=MetadataEventDelivery,
            process_delivery=projector.process_delivery,
            process_invalid_message=projector.process_invalid_message,
        ),
        max_messages=max_messages,
        batch_size=batch_size,
        from_start=from_start,
    )


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


__all__ = [
    "MetadataEventDelivery",
    "MetadataProjectionResult",
    "MetadataProjector",
    "checkpoint_for_delivery",
    "run_metadata_projector",
    "stream_sequence_for_message",
]

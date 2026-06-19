from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeAlias, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.streaming_log.client import StreamingLogConnection
from dr_llm.streaming_log.events import EventEnvelope


class ProjectionEventDelivery(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    event: EventEnvelope
    stream_sequence: int = Field(ge=0)
    message: Any

    async def ack(self) -> None:
        await self.message.ack()


DeliveryT = TypeVar("DeliveryT", bound=ProjectionEventDelivery)
ResultT = TypeVar("ResultT")


InvalidMessageHandler: TypeAlias = Callable[..., Awaitable[Any]]


class ProjectionMessageHandler(BaseModel, Generic[DeliveryT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    delivery_type: type[DeliveryT]
    process_delivery: Callable[[DeliveryT], Awaitable[Any]]
    process_invalid_message: InvalidMessageHandler | None = None


async def process_delivery_with_ack(
    delivery: DeliveryT,
    process_delivery: Callable[[DeliveryT], Awaitable[ResultT]],
) -> ResultT:
    result = await process_delivery(delivery)
    await delivery.ack()
    return result


async def process_invalid_message_with_ack(
    message: Any,
    *,
    stream_sequence: int,
    exc: Exception,
    process_invalid_message: InvalidMessageHandler,
) -> Any:
    result = await process_invalid_message(
        message, stream_sequence=stream_sequence, exc=exc
    )
    await message.ack()
    return result


async def consume_projection_events(
    *,
    connection: StreamingLogConnection,
    durable_consumer: str,
    handler: ProjectionMessageHandler[Any],
    max_messages: int | None,
    batch_size: int,
    from_start: bool = False,
    fetch_timeout: float = 1.0,
    idle_sleep: float = 0.1,
) -> int:
    sub = await connection.js.pull_subscribe(
        connection.config.events_subject,
        durable=projection_consumer_name(
            durable_consumer, from_start=from_start
        ),
        stream=connection.config.events_stream,
    )
    processed = 0
    while should_process_more(max_messages=max_messages, processed=processed):
        messages = await sub.fetch(batch_size, timeout=fetch_timeout)
        if not messages:
            await asyncio.sleep(idle_sleep)
            continue
        for message in messages:
            await process_projection_message(handler=handler, message=message)
            processed += 1
            if not should_process_more(
                max_messages=max_messages, processed=processed
            ):
                break
    return processed


async def process_projection_message(
    *, handler: ProjectionMessageHandler[DeliveryT], message: Any
) -> None:
    stream_sequence = stream_sequence_for_message(message)
    try:
        delivery = delivery_from_message(message, handler.delivery_type)
    except Exception as exc:  # noqa: BLE001
        if handler.process_invalid_message is None:
            raise
        await handler.process_invalid_message(
            message, stream_sequence=stream_sequence, exc=exc
        )
        return
    await handler.process_delivery(delivery)


def delivery_from_message(
    message: Any, delivery_type: type[DeliveryT]
) -> DeliveryT:
    return delivery_type(
        event=EventEnvelope.model_validate_json(message.data),
        stream_sequence=stream_sequence_for_message(message),
        message=message,
    )


def stream_sequence_for_message(message: Any) -> int:
    metadata = getattr(message, "metadata", None)
    sequence = getattr(metadata, "sequence", None)
    stream_sequence = getattr(sequence, "stream", None)
    if stream_sequence is None:
        raise ValueError("message metadata is missing stream sequence")
    return int(stream_sequence)


def projection_consumer_name(
    durable_consumer: str, *, from_start: bool = False
) -> str:
    if not from_start:
        return durable_consumer
    return f"{durable_consumer}_replay_{uuid4().hex[:8]}"


def should_process_more(*, max_messages: int | None, processed: int) -> bool:
    return max_messages is None or processed < max_messages


__all__ = [
    "InvalidMessageHandler",
    "ProjectionEventDelivery",
    "ProjectionMessageHandler",
    "consume_projection_events",
    "delivery_from_message",
    "process_delivery_with_ack",
    "process_invalid_message_with_ack",
    "process_projection_message",
    "projection_consumer_name",
    "should_process_more",
    "stream_sequence_for_message",
]

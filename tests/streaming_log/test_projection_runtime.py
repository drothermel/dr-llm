from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from dr_llm.streaming_log.events import (
    EventEnvelope,
    ProducerInfo,
    StreamingLogEventType,
    WorkSubmittedPayload,
)
from dr_llm.streaming_log.projection_runtime import (
    ProjectionEventDelivery,
    ProjectionMessageHandler,
    consume_projection_events,
    delivery_from_message,
    process_delivery_with_ack,
    process_invalid_message_with_ack,
    process_projection_message,
    projection_consumer_name,
    stream_sequence_for_message,
)


class FakeMessage:
    def __init__(
        self, data: bytes = b"not-json", *, stream_sequence: int = 1
    ) -> None:
        self.data = data
        self.metadata = SimpleNamespace(
            sequence=SimpleNamespace(stream=stream_sequence)
        )
        self.actions: list[str] = []

    async def ack(self) -> None:
        self.actions.append("ack")


class FakeSubscription:
    def __init__(self, batches: list[list[FakeMessage]]) -> None:
        self.batches = batches
        self.fetches: list[tuple[int, float]] = []

    async def fetch(
        self, batch_size: int, *, timeout: float
    ) -> list[FakeMessage]:
        self.fetches.append((batch_size, timeout))
        if not self.batches:
            return []
        return self.batches.pop(0)


class FakeJetStream:
    def __init__(self, subscription: FakeSubscription) -> None:
        self.subscription = subscription
        self.pull_subscribe_calls: list[dict[str, str]] = []

    async def pull_subscribe(
        self, subject: str, *, durable: str, stream: str
    ) -> FakeSubscription:
        self.pull_subscribe_calls.append(
            {"subject": subject, "durable": durable, "stream": stream}
        )
        return self.subscription


class FakeConnection:
    def __init__(self, subscription: FakeSubscription) -> None:
        self.config = SimpleNamespace(
            events_subject="events.>",
            events_stream="EVENTS",
        )
        self.js = FakeJetStream(subscription)


def test_stream_sequence_for_message_reads_metadata() -> None:
    message = FakeMessage(stream_sequence=42)

    assert stream_sequence_for_message(message) == 42


def test_stream_sequence_for_message_requires_stream_sequence() -> None:
    message = SimpleNamespace(metadata=SimpleNamespace(sequence=object()))

    with pytest.raises(ValueError, match="missing stream sequence"):
        stream_sequence_for_message(message)


def test_projection_consumer_name_uses_base_durable_by_default() -> None:
    assert projection_consumer_name("projection") == "projection"


def test_projection_consumer_name_generates_replay_durable() -> None:
    name = projection_consumer_name("projection", from_start=True)

    assert re.fullmatch(r"projection_replay_[0-9a-f]{8}", name)


def test_delivery_from_message_parses_event_and_sequence() -> None:
    event = _event()
    message = FakeMessage(event.json_bytes(), stream_sequence=7)

    delivery = delivery_from_message(message, ProjectionEventDelivery)

    assert delivery.event.event_id == event.event_id
    assert delivery.stream_sequence == 7
    assert delivery.message is message


def test_process_delivery_with_ack_acks_after_handler() -> None:
    message = FakeMessage(_event().json_bytes())
    delivery = delivery_from_message(message, ProjectionEventDelivery)

    async def handle(delivery: ProjectionEventDelivery) -> str:
        delivery.message.actions.append("handled")
        return "ok"

    result = _run(process_delivery_with_ack(delivery, handle))

    assert result == "ok"
    assert message.actions == ["handled", "ack"]


def test_process_invalid_message_with_ack_acks_after_handler() -> None:
    message = FakeMessage()

    async def handle(
        message: Any, *, stream_sequence: int, exc: Exception
    ) -> str:
        del stream_sequence, exc
        message.actions.append("invalid")
        return "recorded"

    result = _run(
        process_invalid_message_with_ack(
            message,
            stream_sequence=3,
            exc=ValueError("bad event"),
            process_invalid_message=handle,
        )
    )

    assert result == "recorded"
    assert message.actions == ["invalid", "ack"]


def test_process_projection_message_leaves_invalid_unacked_without_hook() -> (
    None
):
    message = FakeMessage()

    async def handle(delivery: ProjectionEventDelivery) -> None:
        del delivery

    with pytest.raises(ValidationError):
        _run(
            process_projection_message(
                handler=ProjectionMessageHandler(
                    delivery_type=ProjectionEventDelivery,
                    process_delivery=handle,
                ),
                message=message,
            )
        )

    assert message.actions == []


def test_process_projection_message_uses_invalid_hook() -> None:
    message = FakeMessage(stream_sequence=9)

    async def handle_delivery(delivery: ProjectionEventDelivery) -> None:
        del delivery

    async def handle_invalid(
        message: Any, *, stream_sequence: int, exc: Exception
    ) -> None:
        assert stream_sequence == 9
        assert "Invalid JSON" in str(exc)
        message.actions.append("invalid")
        await message.ack()

    _run(
        process_projection_message(
            handler=ProjectionMessageHandler(
                delivery_type=ProjectionEventDelivery,
                process_delivery=handle_delivery,
                process_invalid_message=handle_invalid,
            ),
            message=message,
        )
    )

    assert message.actions == ["invalid", "ack"]


def test_consume_projection_events_applies_subscription_and_max_messages() -> (
    None
):
    messages = [
        FakeMessage(
            _event(f"work-{index}").json_bytes(), stream_sequence=index
        )
        for index in range(1, 4)
    ]
    subscription = FakeSubscription([messages])
    connection: Any = FakeConnection(subscription)
    handled: list[int] = []

    async def handle(delivery: ProjectionEventDelivery) -> None:
        handled.append(delivery.stream_sequence)
        await delivery.ack()

    processed = _run(
        consume_projection_events(
            connection=connection,
            durable_consumer="projection",
            handler=ProjectionMessageHandler(
                delivery_type=ProjectionEventDelivery,
                process_delivery=handle,
            ),
            max_messages=2,
            batch_size=10,
        )
    )

    assert processed == 2
    assert handled == [1, 2]
    assert subscription.fetches == [(10, 1.0)]
    assert messages[0].actions == ["ack"]
    assert messages[1].actions == ["ack"]
    assert messages[2].actions == []
    assert connection.js.pull_subscribe_calls == [
        {
            "subject": "events.>",
            "durable": "projection",
            "stream": "EVENTS",
        }
    ]


def _event(work_id: str = "work-1") -> EventEnvelope:
    return EventEnvelope(
        event_type=StreamingLogEventType.work_submitted,
        producer=ProducerInfo(name="test"),
        idempotency_key=f"{work_id}-submitted",
        payload=WorkSubmittedPayload(work_id=work_id, max_retries=0),
        work_id=work_id,
    )


def _run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)

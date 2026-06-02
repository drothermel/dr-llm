from __future__ import annotations

import os
import asyncio
from uuid import uuid4

import pytest

from dr_llm.llm import CallMode, LlmRequest, Message, ProviderName
from dr_llm.streaming_log.bootstrap import bootstrap_streaming_log
from dr_llm.streaming_log.client import StreamingLogClient
from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.events import (
    ProducerInfo,
    StreamingLogEventType,
    build_event,
)
from dr_llm.streaming_log.payloads import prepare_text_payload
from dr_llm.streaming_log.work import QueuedWorkMessage

pytestmark = pytest.mark.integration


def _nats_url() -> str:
    value = os.getenv("DR_LLM_TEST_NATS_URL")
    if value is None:
        pytest.skip("Set DR_LLM_TEST_NATS_URL to run NATS integration tests")
    return value


def _config() -> StreamingLogConfig:
    suffix = uuid4().hex[:8].upper()
    return StreamingLogConfig(
        nats_url=_nats_url(),
        events_stream=f"DRLLM_EVENTS_{suffix}",
        work_stream=f"DRLLM_WORK_{suffix}",
        payload_bucket=f"DRLLM_PAYLOADS_{suffix}",
        events_subject=f"drllm.events.{suffix.lower()}.>",
        work_subject=f"drllm.work.{suffix.lower()}.>",
        llm_work_subject=f"drllm.work.{suffix.lower()}.llm",
        work_consumer=f"drllm_work_workers_{suffix.lower()}",
        event_consumer=f"drllm_events_replay_{suffix.lower()}",
    )


def _request() -> LlmRequest:
    return LlmRequest(
        provider=ProviderName.OPENAI,
        model="gpt-test",
        mode=CallMode.api,
        messages=[Message(role="user", content="hi")],
    )


def test_bootstrap_is_idempotent_and_event_replay_works() -> None:
    asyncio.run(_test_bootstrap_is_idempotent_and_event_replay_works())


async def _test_bootstrap_is_idempotent_and_event_replay_works() -> None:
    config = _config()

    first = await bootstrap_streaming_log(config)
    second = await bootstrap_streaming_log(config)

    assert first.events_stream == config.events_stream
    assert second.events_stream == config.events_stream

    async with StreamingLogClient(
        config, producer=ProducerInfo(name="test")
    ) as client:
        event = build_event(
            StreamingLogEventType.producer_started,
            producer=client.producer,
            idempotency_key="event-1",
            payload={"ok": True},
        )
        await client.publish_event(event)
        events = [
            item
            async for item in client.replay_events(
                durable=f"events_{uuid4().hex[:8]}"
            )
        ]

    assert event.event_id in {item.event_id for item in events}


def test_payload_is_written_before_event_publish() -> None:
    asyncio.run(_test_payload_is_written_before_event_publish())


async def _test_payload_is_written_before_event_publish() -> None:
    config = _config()
    await bootstrap_streaming_log(config)

    async with StreamingLogClient(config) as client:
        payload = prepare_text_payload("stdout", "hello")
        event = await client.publish_event_with_payloads(
            StreamingLogEventType.producer_started,
            idempotency_key="payload-event-1",
            payload={"ok": True},
            payloads=[payload],
        )
        stored = await client.read_payload_ref(event.payload_refs[0])

    assert stored == b"hello"


def test_work_messages_are_redelivered_when_unacked() -> None:
    asyncio.run(_test_work_messages_are_redelivered_when_unacked())


async def _test_work_messages_are_redelivered_when_unacked() -> None:
    config = _config()
    await bootstrap_streaming_log(config)

    async with StreamingLogClient(config) as client:
        work = QueuedWorkMessage(work_id="work-1", request=_request())
        await client.submit_work(work)
        first = await client.fetch_work(batch_size=1, timeout=2)
        assert len(first) == 1
        await first[0].nak()
        second = await client.fetch_work(batch_size=1, timeout=2)

    assert len(second) == 1
    assert QueuedWorkMessage.from_payload(second[0].data).work_id == "work-1"

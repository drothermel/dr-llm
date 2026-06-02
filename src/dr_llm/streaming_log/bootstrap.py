from __future__ import annotations

from typing import Any

import nats
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DeliverPolicy,
    ObjectStoreConfig,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)
from nats.js.errors import BucketNotFoundError, NotFoundError
from pydantic import BaseModel, ConfigDict

from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.errors import StreamingLogResourceError


class StreamingLogStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nats_url: str
    events_stream: str
    work_stream: str
    payload_bucket: str
    events_subjects: list[str]
    work_subjects: list[str]
    payload_objects: int


async def bootstrap_streaming_log(
    config: StreamingLogConfig | None = None,
) -> StreamingLogStatus:
    config = config or StreamingLogConfig()
    nc = await nats.connect(config.nats_url)
    try:
        js = nc.jetstream()
        await _ensure_stream(
            js,
            StreamConfig(
                name=config.events_stream,
                subjects=[config.events_subject],
                retention=RetentionPolicy.LIMITS,
                storage=StorageType.FILE,
            ),
        )
        await _ensure_stream(
            js,
            StreamConfig(
                name=config.work_stream,
                subjects=[config.work_subject],
                retention=RetentionPolicy.WORK_QUEUE,
                storage=StorageType.FILE,
            ),
        )
        await _ensure_consumer(
            js,
            config.work_stream,
            ConsumerConfig(
                durable_name=config.work_consumer,
                ack_policy=AckPolicy.EXPLICIT,
                deliver_policy=DeliverPolicy.ALL,
                ack_wait=config.ack_wait_seconds,
                max_deliver=config.max_deliver,
                filter_subject=config.llm_work_subject,
            ),
        )
        await _ensure_consumer(
            js,
            config.events_stream,
            ConsumerConfig(
                durable_name=config.event_consumer,
                ack_policy=AckPolicy.EXPLICIT,
                deliver_policy=DeliverPolicy.ALL,
                ack_wait=config.ack_wait_seconds,
                filter_subject=config.events_subject,
            ),
        )
        await _ensure_object_store(js, config)
        return await inspect_streaming_log(config, js=js)
    finally:
        await nc.close()


async def inspect_streaming_log(
    config: StreamingLogConfig | None = None,
    *,
    js: Any | None = None,
) -> StreamingLogStatus:
    config = config or StreamingLogConfig()
    owns_connection = js is None
    nc = None
    if js is None:
        nc = await nats.connect(config.nats_url)
        js = nc.jetstream()
    try:
        events_info = await js.stream_info(config.events_stream)
        work_info = await js.stream_info(config.work_stream)
        store = await js.object_store(config.payload_bucket)
        store_status = await store.status()
        return StreamingLogStatus(
            nats_url=config.nats_url,
            events_stream=config.events_stream,
            work_stream=config.work_stream,
            payload_bucket=config.payload_bucket,
            events_subjects=list(events_info.config.subjects or []),
            work_subjects=list(work_info.config.subjects or []),
            payload_objects=int(getattr(store_status, "size", 0)),
        )
    finally:
        if owns_connection and nc is not None:
            await nc.close()


async def _ensure_stream(js: Any, expected: StreamConfig) -> None:
    if expected.name is None:
        raise ValueError("stream config name is required")
    try:
        info = await js.stream_info(expected.name)
    except NotFoundError:
        await js.add_stream(expected)
        return
    existing_subjects = set(info.config.subjects or [])
    expected_subjects = set(expected.subjects or [])
    if existing_subjects != expected_subjects:
        raise StreamingLogResourceError(
            f"Stream {expected.name!r} has subjects "
            f"{sorted(existing_subjects)!r}, expected "
            f"{sorted(expected_subjects)!r}"
        )
    if info.config.retention != expected.retention:
        raise StreamingLogResourceError(
            f"Stream {expected.name!r} has retention "
            f"{info.config.retention!r}, expected {expected.retention!r}"
        )


async def _ensure_consumer(
    js: Any, stream_name: str, expected: ConsumerConfig
) -> None:
    if expected.durable_name is None:
        raise ValueError("consumer durable name is required")
    try:
        info = await js.consumer_info(stream_name, expected.durable_name)
    except NotFoundError:
        await js.add_consumer(stream_name, expected)
        return
    if info.config.ack_policy != expected.ack_policy:
        raise StreamingLogResourceError(
            f"Consumer {expected.durable_name!r} has ack policy "
            f"{info.config.ack_policy!r}, expected {expected.ack_policy!r}"
        )
    if info.config.filter_subject != expected.filter_subject:
        raise StreamingLogResourceError(
            f"Consumer {expected.durable_name!r} has filter subject "
            f"{info.config.filter_subject!r}, expected "
            f"{expected.filter_subject!r}"
        )


async def _ensure_object_store(js: Any, config: StreamingLogConfig) -> None:
    try:
        await js.object_store(config.payload_bucket)
    except BucketNotFoundError:
        await js.create_object_store(
            bucket=config.payload_bucket,
            config=ObjectStoreConfig(
                bucket=config.payload_bucket,
                storage=StorageType.FILE,
            ),
        )


__all__ = [
    "StreamingLogStatus",
    "bootstrap_streaming_log",
    "inspect_streaming_log",
]

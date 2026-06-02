"""Shared helpers for live streaming-log demo scripts."""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from uuid import uuid4

import nats
from nats.errors import TimeoutError as NatsTimeoutError
from pydantic import BaseModel, ConfigDict

from dr_llm.demo.requirements import ensure_docker_available
from dr_llm.project.docker_runner import call_docker
from dr_llm.streaming_log import (
    EventEnvelope,
    StreamingEventLog,
    StreamingPayloadReader,
)
from dr_llm.streaming_log.config import StreamingLogConfig

DEFAULT_NATS_DOCKER_REASON = (
    "This demo creates a NATS JetStream container when no --nats-url is "
    "provided."
)
DEFAULT_NATS_DOCKER_RECOVERY_HINT = (
    "Install Docker, start the daemon, or pass --nats-url to use an "
    "existing NATS server."
)


class DemoNatsLease(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nats_url: str
    container_name: str | None = None
    should_destroy_container: bool = False


class PayloadVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    role: str
    object_key: str
    size_bytes: int


def demo_streaming_log_config(
    *,
    nats_url: str,
    suffix: str | None = None,
) -> StreamingLogConfig:
    resolved_suffix = (suffix or uuid4().hex[:8]).lower()
    stream_suffix = resolved_suffix.upper()
    return StreamingLogConfig(
        nats_url=nats_url,
        events_stream=f"DRLLM_DEMO_EVENTS_{stream_suffix}",
        work_stream=f"DRLLM_DEMO_WORK_{stream_suffix}",
        payload_bucket=f"DRLLM_DEMO_PAYLOADS_{stream_suffix}",
        events_subject=f"drllm.demo.{resolved_suffix}.events.>",
        work_subject=f"drllm.demo.{resolved_suffix}.work.>",
        llm_work_subject=f"drllm.demo.{resolved_suffix}.work.llm",
        work_consumer=f"drllm_demo_work_{resolved_suffix}",
        event_consumer=f"drllm_demo_events_{resolved_suffix}",
    )


def prepare_demo_nats(
    *,
    nats_url: str | None,
    keep_nats: bool = False,
    container_prefix: str = "dr_llm_demo_nats",
) -> DemoNatsLease:
    if nats_url is not None:
        return DemoNatsLease(nats_url=nats_url)

    ensure_docker_available(
        reason=DEFAULT_NATS_DOCKER_REASON,
        recovery_hint=DEFAULT_NATS_DOCKER_RECOVERY_HINT,
    )
    container_name = f"{container_prefix}_{uuid4().hex[:8]}"
    call_docker(
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        "127.0.0.1::4222",
        "nats",
        "-js",
    )
    port_line = call_docker("port", container_name, "4222/tcp").stdout.strip()
    port = port_line.rsplit(":", 1)[-1]
    return DemoNatsLease(
        nats_url=f"nats://127.0.0.1:{port}",
        container_name=container_name,
        should_destroy_container=not keep_nats,
    )


def cleanup_demo_nats(lease: DemoNatsLease) -> None:
    if lease.container_name is None or not lease.should_destroy_container:
        return
    call_docker("rm", "-f", lease.container_name, check=False)


async def wait_for_nats(
    nats_url: str,
    *,
    timeout_seconds: float = 20.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_error: Exception | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            nc = await nats.connect(nats_url, connect_timeout=1)
            await nc.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.2)
    raise RuntimeError(f"NATS was not ready at {nats_url}: {last_error}")


async def collect_streaming_log_events(
    event_log: StreamingEventLog,
    *,
    expected_min_events: int = 1,
    idle_timeout_seconds: float = 0.5,
    max_idle_fetches: int = 2,
) -> list[EventEnvelope]:
    sub = await event_log.event_subscription(
        durable=f"{event_log.config.event_consumer}_demo_{uuid4().hex[:8]}"
    )
    events: list[EventEnvelope] = []
    idle_fetches = 0
    while idle_fetches < max_idle_fetches:
        try:
            messages = await sub.fetch(
                event_log.config.fetch_batch_size,
                timeout=idle_timeout_seconds,
            )
        except NatsTimeoutError:
            if len(events) >= expected_min_events:
                idle_fetches += 1
                continue
            raise
        if not messages:
            if len(events) < expected_min_events:
                raise RuntimeError(
                    "NATS fetch returned no messages before collecting "
                    f"{expected_min_events} expected events"
                )
            idle_fetches += 1
            continue
        idle_fetches = 0
        for msg in messages:
            events.append(EventEnvelope.model_validate_json(msg.data))
            await msg.ack()
    return events


async def verify_payload_refs(
    payload_reader: StreamingPayloadReader,
    events: list[EventEnvelope],
) -> list[PayloadVerification]:
    verified: list[PayloadVerification] = []
    for event in events:
        for ref in event.payload_refs:
            data = await payload_reader.read_payload_ref(ref)
            digest = hashlib.sha256(data).hexdigest()
            if digest != ref.sha256:
                raise RuntimeError(
                    f"Payload hash mismatch for {ref.object_key}: "
                    f"{digest} != {ref.sha256}"
                )
            if len(data) != ref.size_bytes:
                raise RuntimeError(
                    f"Payload size mismatch for {ref.object_key}: "
                    f"{len(data)} != {ref.size_bytes}"
                )
            verified.append(
                PayloadVerification(
                    event_id=event.event_id,
                    role=ref.role,
                    object_key=ref.object_key,
                    size_bytes=ref.size_bytes,
                )
            )
    return verified


def summarize_events(events: list[EventEnvelope]) -> Counter[str]:
    return Counter(str(event.event_type) for event in events)


__all__ = [
    "DemoNatsLease",
    "PayloadVerification",
    "cleanup_demo_nats",
    "collect_streaming_log_events",
    "demo_streaming_log_config",
    "prepare_demo_nats",
    "summarize_events",
    "verify_payload_refs",
    "wait_for_nats",
]

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm import LlmResponse, ProviderRegistry, build_default_registry
from dr_llm.streaming_log.client import (
    ContextualEventPublisher,
    StreamingLogClient,
)
from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.events import (
    EventContext,
    StreamingLogEventType,
    idempotency_key,
)
from dr_llm.streaming_log.payloads import prepare_json_payload
from dr_llm.streaming_log.work import QueuedWorkMessage

logger = logging.getLogger(__name__)


class StreamingWorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: str = Field(default_factory=lambda: f"stream-{uuid4().hex[:8]}")
    max_messages: int | None = Field(default=None, ge=1)
    fetch_timeout_seconds: float = Field(default=1.0, gt=0)


async def run_streaming_worker(
    *,
    client: StreamingLogClient | None = None,
    config: StreamingWorkerConfig | None = None,
    registry_factory: Callable[[], ProviderRegistry] = build_default_registry,
) -> None:
    config = config or StreamingWorkerConfig()
    owns_client = client is None
    if client is None:
        client = StreamingLogClient(StreamingLogConfig())
    if owns_client:
        await client.connect()
    registry = registry_factory()
    processed = 0
    try:
        await client.publish_event_with_payloads(
            StreamingLogEventType.producer_started,
            idempotency_key=idempotency_key(
                "producer_started", config.worker_id
            ),
            payload={"worker_id": config.worker_id},
        )
        sub = await client.work_subscription()
        while config.max_messages is None or processed < config.max_messages:
            try:
                messages = await sub.fetch(
                    client.config.fetch_batch_size,
                    timeout=config.fetch_timeout_seconds,
                )
            except TimeoutError:
                continue
            for msg in messages:
                await _process_message(
                    client=client,
                    registry=registry,
                    worker_id=config.worker_id,
                    msg=msg,
                )
                processed += 1
                if (
                    config.max_messages is not None
                    and processed >= config.max_messages
                ):
                    break
    finally:
        try:
            await client.publish_event_with_payloads(
                StreamingLogEventType.producer_stopped,
                idempotency_key=idempotency_key(
                    "producer_stopped", config.worker_id
                ),
                payload={"worker_id": config.worker_id},
            )
        finally:
            registry.close()
            if owns_client:
                await client.close()


async def _process_message(
    *,
    client: StreamingLogClient,
    registry: ProviderRegistry,
    worker_id: str,
    msg,
) -> None:
    work = QueuedWorkMessage.from_payload(msg.data)
    attempt = int(getattr(msg.metadata, "num_delivered", 1))
    attempt_id = f"{work.work_id}-{attempt}"
    publisher = client.with_event_context(
        EventContext.from_work_attempt(work, attempt_id=attempt_id)
    )
    await publisher.publish_event_with_payloads(
        StreamingLogEventType.attempt_started,
        idempotency_key=idempotency_key(
            "attempt_started", work.work_id, attempt
        ),
        payload={"worker_id": worker_id, "attempt": attempt},
    )
    request_payload = work.request.model_dump(
        mode="json",
        exclude_none=True,
        exclude_computed_fields=True,
    )
    await publisher.publish_event_with_payloads(
        StreamingLogEventType.provider_request_prepared,
        idempotency_key=idempotency_key(
            "provider_request_prepared", work.work_id, attempt
        ),
        payload={
            "provider": work.request.provider,
            "model": work.request.model,
            "mode": work.request.mode,
        },
        payloads=[prepare_json_payload("request_json", request_payload)],
    )
    try:
        response = await _generate_response(registry, work)
    except Exception as exc:  # noqa: BLE001
        await _handle_failure(
            publisher=publisher,
            work=work,
            msg=msg,
            exc=exc,
            attempt=attempt,
        )
        return
    await _handle_success(
        publisher=publisher,
        work=work,
        msg=msg,
        response=response,
        attempt=attempt,
    )


async def _generate_response(
    registry: ProviderRegistry, work: QueuedWorkMessage
) -> LlmResponse:
    orchestrator = registry.get(str(work.request.provider))
    return await asyncio.to_thread(orchestrator.generate, work.request)


async def _handle_success(
    *,
    publisher: ContextualEventPublisher,
    work: QueuedWorkMessage,
    msg,
    response: LlmResponse,
    attempt: int,
) -> None:
    response_payload = response.model_dump(
        mode="json",
        exclude_none=True,
        exclude_computed_fields=True,
    )
    await publisher.publish_event_with_payloads(
        StreamingLogEventType.provider_response_received,
        idempotency_key=idempotency_key(
            "provider_response_received", work.work_id, attempt
        ),
        payload={
            "provider": response.provider,
            "model": response.model,
            "mode": response.mode,
            "finish_reason": response.finish_reason,
        },
        payloads=[prepare_json_payload("response_json", response_payload)],
    )
    await publisher.publish_event_with_payloads(
        StreamingLogEventType.attempt_succeeded,
        idempotency_key=idempotency_key(
            "attempt_succeeded", work.work_id, attempt
        ),
        payload={"attempt": attempt},
    )
    await publisher.publish_event_with_payloads(
        StreamingLogEventType.work_completed,
        idempotency_key=idempotency_key("work_completed", work.work_id),
        payload={"status": "succeeded", "attempt": attempt},
    )
    await msg.ack()


async def _handle_failure(
    *,
    publisher: ContextualEventPublisher,
    work: QueuedWorkMessage,
    msg,
    exc: Exception,
    attempt: int,
) -> None:
    error_payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "attempt": attempt,
    }
    await publisher.publish_event_with_payloads(
        StreamingLogEventType.attempt_failed,
        idempotency_key=idempotency_key(
            "attempt_failed", work.work_id, attempt
        ),
        payload=error_payload,
        payloads=[prepare_json_payload("error_detail", error_payload)],
    )
    if attempt <= work.max_retries:
        await publisher.publish_event_with_payloads(
            StreamingLogEventType.work_retry_scheduled,
            idempotency_key=idempotency_key(
                "work_retry_scheduled", work.work_id, attempt
            ),
            payload={"attempt": attempt, "next_attempt": attempt + 1},
        )
        await msg.nak()
        return
    await publisher.publish_event_with_payloads(
        StreamingLogEventType.work_completed,
        idempotency_key=idempotency_key("work_completed", work.work_id),
        payload={"status": "failed", **error_payload},
    )
    await msg.ack()


__all__ = ["StreamingWorkerConfig", "run_streaming_worker"]

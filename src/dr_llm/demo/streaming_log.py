"""Shared helpers for live streaming-log demo scripts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import nats
from nats.errors import TimeoutError as NatsTimeoutError
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.demo.cli_calls import (
    list_models_json,
    show_model_json,
    sync_models_json,
)
from dr_llm.demo.console import ok, warn
from dr_llm.demo.demo_models import DEMO_QUERY_DEFAULT_MODELS
from dr_llm.llm import (
    LlmRequest,
    Message,
    ProviderAvailabilityStatus,
    ProviderName,
    build_default_registry,
)
from dr_llm.demo.requirements import ensure_docker_available
from dr_llm.project.docker_runner import call_docker
from dr_llm.streaming_log import (
    EventEnvelope,
    StreamingEventLog,
    StreamingLogEventType,
    StreamingLogConnection,
    StreamingLogStatus,
    StreamingPayloadReader,
    StreamingPayloadStore,
    StreamingWorkQueue,
    bootstrap_streaming_log,
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
DEFAULT_PROVIDER_PREFERENCE = [
    ProviderName.OPENAI,
    ProviderName.GOOGLE,
    ProviderName.GLM,
    ProviderName.OPENROUTER,
    ProviderName.MINIMAX,
    ProviderName.KIMI_CODE,
    ProviderName.CODEX,
    ProviderName.CLAUDE_CODE,
    ProviderName.ANTHROPIC,
]
SUCCESSFUL_WORK_LIFECYCLE_EVENTS = [
    StreamingLogEventType.work_submitted,
    StreamingLogEventType.attempt_started,
    StreamingLogEventType.provider_request_prepared,
    StreamingLogEventType.provider_response_received,
    StreamingLogEventType.attempt_succeeded,
    StreamingLogEventType.work_completed,
]
PRODUCER_LIFECYCLE_EVENTS = [
    StreamingLogEventType.producer_started,
    StreamingLogEventType.producer_stopped,
]


class DemoProviderWorkFailedError(RuntimeError):
    pass


class DemoNatsLease(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nats_url: str
    container_name: str | None = None
    should_destroy_container: bool = False


class DemoNatsOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nats_url: str | None = None
    keep_nats: bool = False
    container_prefix: str = "dr_llm_demo_nats"


class StreamingLogDemoRuntimeOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nats: DemoNatsOptions = Field(default_factory=DemoNatsOptions)
    suffix: str | None = None


class StreamingLogDemoRuntime(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lease: DemoNatsLease
    config: StreamingLogConfig
    status: StreamingLogStatus

    @property
    def cleanup_command(self) -> str | None:
        if self.lease.container_name is None:
            return None
        return f"docker rm -f {self.lease.container_name}"


class StreamingLogDemoSession(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    runtime: StreamingLogDemoRuntime
    connection: StreamingLogConnection
    payload_store: StreamingPayloadStore
    event_log: StreamingEventLog
    work_queue: StreamingWorkQueue


class PayloadVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    role: str
    object_key: str
    size_bytes: int


class DemoProviderCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str


class DemoResponsePayloadVerification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text_preview: str


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


@asynccontextmanager
async def open_streaming_log_demo_runtime(
    options: StreamingLogDemoRuntimeOptions | None = None,
) -> AsyncIterator[StreamingLogDemoSession]:
    resolved_options = options or StreamingLogDemoRuntimeOptions()
    lease = prepare_demo_nats(
        nats_url=resolved_options.nats.nats_url,
        keep_nats=resolved_options.nats.keep_nats,
        container_prefix=resolved_options.nats.container_prefix,
    )
    connection: StreamingLogConnection | None = None
    try:
        await wait_for_nats(lease.nats_url)
        config = demo_streaming_log_config(
            nats_url=lease.nats_url,
            suffix=resolved_options.suffix,
        )
        status = await bootstrap_streaming_log(config)
        connection = StreamingLogConnection(config)
        await connection.connect()
        payload_store = StreamingPayloadStore(connection)
        event_log = StreamingEventLog(connection, payload_store)
        work_queue = StreamingWorkQueue(connection, event_log)
        yield StreamingLogDemoSession(
            runtime=StreamingLogDemoRuntime(
                lease=lease,
                config=config,
                status=status,
            ),
            connection=connection,
            payload_store=payload_store,
            event_log=event_log,
            work_queue=work_queue,
        )
    finally:
        if connection is not None:
            await connection.close()
        cleanup_demo_nats(lease)


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


def available_demo_providers_or_raise(
    statuses: list[ProviderAvailabilityStatus],
) -> list[ProviderAvailabilityStatus]:
    available: list[ProviderAvailabilityStatus] = []
    for status in statuses:
        if status.available:
            available.append(status)
            continue
        reasons = [f"{env_var} not set" for env_var in status.missing_env_vars]
        reasons.extend(
            f"'{executable}' CLI not found"
            for executable in status.missing_executables
        )
        warn(f"{status.provider}: {', '.join(reasons)}")
    if not available:
        raise RuntimeError(
            "No providers available. Set provider API keys or install a "
            "supported provider CLI."
        )
    return available


def demo_provider_names(
    available: list[ProviderAvailabilityStatus],
    *,
    requested_provider: str | ProviderName | None,
    preference: list[ProviderName] | None = None,
) -> list[str]:
    available_by_name = {status.provider: status for status in available}
    if requested_provider is not None:
        provider = str(requested_provider)
        if provider not in available_by_name:
            raise RuntimeError(
                f"Requested provider {provider!r} is not available. "
                f"Available providers: {sorted(available_by_name)}"
            )
        return [provider]

    providers: list[str] = []
    for provider in preference or DEFAULT_PROVIDER_PREFERENCE:
        if str(provider) in available_by_name:
            providers.append(str(provider))
    providers.extend(
        status.provider
        for status in available
        if status.provider not in providers
    )
    return providers


def demo_provider_candidates(
    *,
    requested_provider: str | ProviderName | None = None,
    requested_model: str | None = None,
) -> list[DemoProviderCandidate]:
    registry = build_default_registry()
    try:
        available = available_demo_providers_or_raise(
            registry.availability_statuses()
        )
        provider_names = demo_provider_names(
            available,
            requested_provider=requested_provider,
        )
        candidates: list[DemoProviderCandidate] = []
        for provider in provider_names:
            candidate = demo_provider_candidate(
                provider,
                requested_model=requested_model,
            )
            if candidate is not None:
                candidates.append(candidate)
        if candidates:
            return candidates
        raise RuntimeError("No provider candidate could resolve a model")
    finally:
        registry.close()


def demo_provider_candidate(
    provider: str,
    *,
    requested_model: str | None,
) -> DemoProviderCandidate | None:
    if requested_model is not None:
        return DemoProviderCandidate(provider=provider, model=requested_model)
    try:
        return DemoProviderCandidate(
            provider=provider,
            model=resolve_demo_model(provider),
        )
    except Exception as exc:  # noqa: BLE001
        warn(f"Skipping provider {provider}: could not resolve model: {exc}")
        return None


def resolve_demo_model(
    provider: str, *, requested_model: str | None = None
) -> str:
    sync_models_json(provider)
    if requested_model is not None:
        return requested_model

    models = list_models_json(provider)
    if not models:
        raise RuntimeError(f"No models found for {provider}")
    model_ids = [str(model["model"]) for model in models]
    default_model = DEMO_QUERY_DEFAULT_MODELS.get(ProviderName(provider))
    if default_model is not None and default_model in model_ids:
        return default_model
    if default_model is not None:
        warn(
            f"default model {default_model!r} not found; using {model_ids[0]!r}"
        )
    return model_ids[0]


def build_live_demo_request(
    *,
    prompt: str,
    provider: str,
    model: str,
) -> LlmRequest:
    registry = build_default_registry()
    try:
        info = show_model_json(provider, model)
        display = info.get("display_name", model)
        ok(f"Model info: {display}")
        orchestrator = registry.get(provider)
        defaults = orchestrator.request_defaults(model)
        return orchestrator.build_request(
            model=model,
            messages=[Message(role="user", content=prompt)],
            max_tokens=defaults.max_tokens,
            effort=defaults.effort,
            reasoning=defaults.reasoning,
        )
    finally:
        registry.close()


def verify_successful_work_lifecycle(
    events: list[EventEnvelope],
    *,
    work_id: str,
    require_producer_lifecycle: bool = False,
    reject_attempt_failed: bool = True,
    success_message: str | None = None,
) -> None:
    work_events = work_events_for(events, work_id=work_id)
    work_counts = Counter(event.event_type for event in work_events)
    producer_counts = Counter(event.event_type for event in events)
    expected = list(SUCCESSFUL_WORK_LIFECYCLE_EVENTS)
    if require_producer_lifecycle:
        expected.extend(PRODUCER_LIFECYCLE_EVENTS)

    missing = [
        event_type
        for event_type in SUCCESSFUL_WORK_LIFECYCLE_EVENTS
        if work_counts[event_type] < 1
    ]
    if require_producer_lifecycle:
        missing.extend(
            event_type
            for event_type in PRODUCER_LIFECYCLE_EVENTS
            if producer_counts[event_type] < 1
        )
    if missing:
        if work_counts[StreamingLogEventType.attempt_failed] > 0:
            raise DemoProviderWorkFailedError(
                "Provider work failed before a successful response; "
                "missing lifecycle events: "
                + ", ".join(str(event_type) for event_type in missing)
                + failure_details_suffix(work_events)
            )
        raise RuntimeError(
            "Missing expected lifecycle events: "
            + ", ".join(str(event_type) for event_type in missing)
        )
    if (
        reject_attempt_failed
        and work_counts[StreamingLogEventType.attempt_failed] > 0
    ):
        raise DemoProviderWorkFailedError(
            "attempt_failed was emitted for the selected work"
            + failure_details_suffix(work_events)
        )
    completed = single_work_event(
        work_events,
        StreamingLogEventType.work_completed,
        work_id=work_id,
    )
    status = getattr(completed.payload, "status", None)
    if status != "succeeded":
        raise DemoProviderWorkFailedError(
            f"work_completed status is {status!r}, expected 'succeeded'"
            + failure_details_suffix(work_events)
        )
    ok(
        success_message
        or "Lifecycle events verified: "
        + ", ".join(str(event_type) for event_type in expected)
    )


async def verify_response_payload(
    payload_reader: StreamingPayloadReader,
    events: list[EventEnvelope],
    *,
    work_id: str,
) -> DemoResponsePayloadVerification:
    response_event = single_work_event(
        work_events_for(events, work_id=work_id),
        StreamingLogEventType.provider_response_received,
        work_id=work_id,
    )
    refs = [
        ref
        for ref in response_event.payload_refs
        if ref.role == "response_json"
    ]
    if len(refs) != 1:
        raise RuntimeError(
            f"Expected one response_json payload for {work_id}, found {len(refs)}"
        )
    data = await payload_reader.read_payload_ref(refs[0])
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise RuntimeError("response_json payload is not a JSON object")
    response_payload: dict[str, Any] = payload
    text = response_payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("response_json payload has no non-empty text field")
    return DemoResponsePayloadVerification(
        text_preview=text[:120].replace("\n", " ")
    )


def work_events_for(
    events: list[EventEnvelope],
    *,
    work_id: str,
) -> list[EventEnvelope]:
    return [event for event in events if event.work_id == work_id]


def single_work_event(
    events: list[EventEnvelope],
    event_type: StreamingLogEventType,
    *,
    work_id: str,
) -> EventEnvelope:
    matches = [event for event in events if event.event_type is event_type]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {event_type} event for {work_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def failure_details_suffix(events: list[EventEnvelope]) -> str:
    details = attempt_failure_details(events)
    if not details:
        return ""
    return "; failure_details=" + "; ".join(details)


def attempt_failure_details(events: list[EventEnvelope]) -> list[str]:
    details: list[str] = []
    for event in events:
        if event.event_type is not StreamingLogEventType.attempt_failed:
            continue
        details.append(
            f"{getattr(event.payload, 'error_type', 'unknown')}: "
            f"{getattr(event.payload, 'message', '')}"
        )
    return details


async def payload_attempt_failure_details(
    payload_reader: StreamingPayloadReader,
    events: list[EventEnvelope],
) -> list[str]:
    details: list[str] = []
    for event in events:
        if event.event_type is not StreamingLogEventType.attempt_failed:
            continue
        for ref in event.payload_refs:
            if ref.role != "error_detail":
                continue
            data = await payload_reader.read_payload_ref(ref)
            detail = json.loads(data)
            summary = f"{detail.get('error_type')}: {detail.get('message')}"
            details.append(summary)
            warn("Provider attempt failed: " + summary)
    return details


def summarize_events(events: list[EventEnvelope]) -> Counter[str]:
    return Counter(str(event.event_type) for event in events)


__all__ = [
    "attempt_failure_details",
    "available_demo_providers_or_raise",
    "build_live_demo_request",
    "DemoNatsLease",
    "DemoNatsOptions",
    "DemoProviderCandidate",
    "DemoProviderWorkFailedError",
    "DemoResponsePayloadVerification",
    "demo_provider_candidate",
    "demo_provider_candidates",
    "demo_provider_names",
    "PayloadVerification",
    "payload_attempt_failure_details",
    "StreamingLogDemoRuntime",
    "StreamingLogDemoRuntimeOptions",
    "StreamingLogDemoSession",
    "cleanup_demo_nats",
    "collect_streaming_log_events",
    "demo_streaming_log_config",
    "failure_details_suffix",
    "open_streaming_log_demo_runtime",
    "prepare_demo_nats",
    "resolve_demo_model",
    "single_work_event",
    "summarize_events",
    "verify_response_payload",
    "verify_payload_refs",
    "verify_successful_work_lifecycle",
    "wait_for_nats",
    "work_events_for",
]

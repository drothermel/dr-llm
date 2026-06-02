#!/usr/bin/env python3
"""Demo: submit one real LLM request and process it through the streaming log.

Usage:
  uv run python scripts/demo-streaming-log-worker.py
  uv run python scripts/demo-streaming-log-worker.py --provider openai --model gpt-4o-mini
  uv run python scripts/demo-streaming-log-worker.py --keep-nats

Prerequisites:
  1. At least one real provider available through API keys or CLI tools.
  2. Docker running, unless --nats-url points at an existing NATS server.

The demo:
  - Auto-detects an available provider/model, falling back on provider failures
  - Starts a temporary NATS JetStream server when --nats-url is omitted
  - Submits a real work message
  - Runs the async streaming-log worker for one message
  - Replays and verifies lifecycle events and payload references
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
import typer

from dr_llm.demo import (
    DEMO_QUERY_DEFAULT_MODELS,
    DemoPrompts,
    collect_streaming_log_events,
    command_hint,
    DemoNatsOptions,
    fail,
    list_models_json,
    ok,
    open_streaming_log_demo_runtime,
    show_model_json,
    step,
    summarize_events,
    StreamingLogDemoRuntime,
    StreamingLogDemoRuntimeOptions,
    sync_models_json,
    verify_payload_refs,
    warn,
)
from dr_llm.llm import (
    LlmRequest,
    Message,
    ProviderAvailabilityStatus,
    ProviderName,
    build_default_registry,
)
from dr_llm.streaming_log import (
    EventEnvelope,
    QueuedWorkMessage,
    StreamingPayloadStore,
    StreamingWorkerConfig,
    run_streaming_worker,
)

app = typer.Typer()

PROVIDER_PREFERENCE = [
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


class ProviderWorkFailedError(RuntimeError):
    pass


class WorkerDemoOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nats: DemoNatsOptions = Field(default_factory=DemoNatsOptions)
    prompt: str
    max_retries: int = 0
    provider: ProviderName | None = None
    model: str | None = None


class WorkerAttemptOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nats: DemoNatsOptions = Field(default_factory=DemoNatsOptions)
    max_retries: int
    provider: str
    model: str
    request: LlmRequest
    attempt_index: int
    attempt_count: int


WorkerDemoOptions.model_rebuild(
    _types_namespace={
        "DemoNatsOptions": DemoNatsOptions,
        "ProviderName": ProviderName,
    }
)
WorkerAttemptOptions.model_rebuild(
    _types_namespace={
        "DemoNatsOptions": DemoNatsOptions,
        "LlmRequest": LlmRequest,
    }
)


async def _run_worker_demo(options: WorkerDemoOptions) -> None:
    step("1. Detecting live provider")
    registry = build_default_registry()
    try:
        statuses = registry.availability_statuses()
        available = _available_providers_or_exit(statuses)
        candidate_providers = _candidate_providers(
            available,
            requested_provider=options.provider,
        )
    finally:
        registry.close()

    allow_fallback = options.provider is None and options.model is None
    if not allow_fallback:
        candidate_providers = candidate_providers[:1]
    failures: list[str] = []
    for index, candidate_provider in enumerate(candidate_providers, start=1):
        provider_name, model_name, request = _build_live_request(
            prompt=options.prompt,
            provider=candidate_provider,
            requested_model=options.model,
        )
        ok(f"Using {provider_name}/{model_name}")
        try:
            await _run_worker_attempt(
                WorkerAttemptOptions(
                    nats=options.nats,
                    max_retries=options.max_retries,
                    provider=provider_name,
                    model=model_name,
                    request=request,
                    attempt_index=index,
                    attempt_count=len(candidate_providers),
                )
            )
            if failures:
                ok(
                    "Fallback succeeded after skipped provider attempts: "
                    + "; ".join(failures)
                )
            return
        except ProviderWorkFailedError as exc:
            failures.append(f"{provider_name}/{model_name}: {exc}")
            if not allow_fallback or index == len(candidate_providers):
                raise
            warn(
                f"{provider_name}/{model_name} failed during live execution; "
                "trying the next available provider"
            )
    raise RuntimeError("No live provider attempt completed successfully")


async def _run_worker_attempt(options: WorkerAttemptOptions) -> None:
    step(
        f"2. Preparing NATS for provider attempt "
        f"{options.attempt_index}/{options.attempt_count}"
    )
    runtime: StreamingLogDemoRuntime | None = None
    events: list[EventEnvelope] = []
    verified_payloads = []
    try:
        async with open_streaming_log_demo_runtime(
            StreamingLogDemoRuntimeOptions(nats=options.nats)
        ) as session:
            runtime = session.runtime
            _print_runtime_summary(runtime)

            step("3. Submitting work")
            work = QueuedWorkMessage(
                request=options.request,
                source="demo-streaming-log-worker",
                metadata={
                    "demo": "streaming-log-worker",
                    "provider": options.provider,
                    "model": options.model,
                },
                max_retries=options.max_retries,
            )
            submitted = await session.work_queue.submit_work(work)
            ok(
                f"Submitted work_id={work.work_id} "
                f"event_id={submitted.event_id}"
            )

            step("4. Running async streaming worker")
            await run_streaming_worker(
                work_queue=session.work_queue,
                config=StreamingWorkerConfig(
                    worker_id="demo-streaming-worker",
                    max_messages=1,
                ),
            )
            ok("Worker processed one live message")

            step("5. Replaying and verifying events")
            events = await collect_streaming_log_events(
                session.event_log,
                expected_min_events=1,
            )
            counts = summarize_events(events)
            verified_payloads = await verify_payload_refs(
                session.payload_store, events
            )
            _print_worker_event_summary(events)
            await _print_failure_details(session.payload_store, events)
            _verify_worker_lifecycle(counts)

        ok(f"Verified {len(events)} replayed events")
        ok(f"Verified {len(verified_payloads)} payload references")
        ok("Streaming-log worker demo verified live execution")
    finally:
        _print_cleanup_hint(runtime)


def _print_runtime_summary(runtime: StreamingLogDemoRuntime) -> None:
    ok(f"NATS ready at {runtime.lease.nats_url}")
    ok(f"Using event stream {runtime.config.events_stream}")
    ok(f"Using work stream {runtime.config.work_stream}")
    ok(f"Using payload bucket {runtime.config.payload_bucket}")


def _print_cleanup_hint(runtime: StreamingLogDemoRuntime | None) -> None:
    if runtime is None or runtime.lease.should_destroy_container:
        return
    if runtime.cleanup_command is not None:
        command_hint("Destroy NATS", runtime.cleanup_command)


def _build_live_request(
    *, prompt: str, provider: str, requested_model: str | None
) -> tuple[str, str, LlmRequest]:
    registry = build_default_registry()
    try:
        model = _resolve_model(
            provider,
            requested_model=requested_model,
        )
        info = show_model_json(provider, model)
        display = info.get("display_name", model)
        ok(f"Model info: {display}")
        orchestrator = registry.get(provider)
        defaults = orchestrator.request_defaults(model)
        request = orchestrator.build_request(
            model=model,
            messages=[Message(role="user", content=prompt)],
            max_tokens=defaults.max_tokens,
            effort=defaults.effort,
            reasoning=defaults.reasoning,
        )
        return provider, model, request
    finally:
        registry.close()


def _candidate_providers(
    available: list[ProviderAvailabilityStatus],
    *,
    requested_provider: ProviderName | None,
) -> list[str]:
    available_by_name = {status.provider: status for status in available}
    if requested_provider is not None:
        provider = str(requested_provider)
        if provider not in available_by_name:
            raise RuntimeError(
                f"Requested provider {provider!r} is not available"
            )
        return [provider]

    providers: list[str] = []
    for provider in PROVIDER_PREFERENCE:
        if str(provider) in available_by_name:
            providers.append(str(provider))
    providers.extend(
        status.provider
        for status in available
        if status.provider not in providers
    )
    return providers


def _available_providers_or_exit(
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
        raise RuntimeError("No live providers available")
    return available


def _resolve_model(provider: str, *, requested_model: str | None) -> str:
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


async def _print_failure_details(
    payload_store: StreamingPayloadStore,
    events: list[EventEnvelope],
) -> list[str]:
    details: list[str] = []
    for event in events:
        if str(event.event_type) != "attempt_failed":
            continue
        for ref in event.payload_refs:
            if ref.role != "error_detail":
                continue
            data = await payload_store.read_payload_ref(ref)
            detail = json.loads(data)
            summary = f"{detail.get('error_type')}: {detail.get('message')}"
            details.append(summary)
            warn("Provider attempt failed: " + summary)
    return details


def _verify_worker_lifecycle(counts: Counter[str]) -> None:
    expected = [
        "work_submitted",
        "producer_started",
        "attempt_started",
        "provider_request_prepared",
        "provider_response_received",
        "attempt_succeeded",
        "work_completed",
        "producer_stopped",
    ]
    missing = [event_type for event_type in expected if counts[event_type] < 1]
    if missing:
        if counts["attempt_failed"] > 0:
            raise ProviderWorkFailedError(
                "Provider work failed before a successful response; "
                "see attempt_failed details above. Missing lifecycle events: "
                + ", ".join(missing)
            )
        raise RuntimeError(
            "Missing expected lifecycle events: " + ", ".join(missing)
        )
    ok("Lifecycle events verified: " + ", ".join(expected))


def _print_worker_event_summary(events: list[EventEnvelope]) -> None:
    step("7. Event sequence")
    for event in events:
        roles = [ref.role for ref in event.payload_refs]
        print(
            f"  {event.event_type} "
            f"work_id={event.work_id or '-'} "
            f"attempt_id={event.attempt_id or '-'} "
            f"payload_roles={roles or '-'}"
        )


@app.command()
def main(
    nats_url: Annotated[
        str | None,
        typer.Option(
            help=(
                "NATS URL. If omitted, a temporary Docker NATS server is "
                "created."
            )
        ),
    ] = None,
    keep_nats: Annotated[
        bool,
        typer.Option(
            "--keep-nats",
            help="Keep the auto-created NATS container for inspection.",
        ),
    ] = False,
    prompt: Annotated[
        str,
        typer.Option(help="Prompt to submit as live streaming work."),
    ] = DemoPrompts.TWO_PLUS_TWO,
    max_retries: Annotated[
        int,
        typer.Option(help="Retry count for the submitted work message."),
    ] = 0,
    provider: Annotated[
        ProviderName | None,
        typer.Option(
            help="Provider to use. Defaults to a preferred available provider."
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            help="Model to use. Defaults to the demo model for the provider."
        ),
    ] = None,
) -> None:
    try:
        asyncio.run(
            _run_worker_demo(
                WorkerDemoOptions(
                    nats=DemoNatsOptions(
                        nats_url=nats_url,
                        keep_nats=keep_nats,
                    ),
                    prompt=prompt,
                    max_retries=max_retries,
                    provider=provider,
                    model=model,
                )
            )
        )
    except Exception as exc:
        fail(str(exc))
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()

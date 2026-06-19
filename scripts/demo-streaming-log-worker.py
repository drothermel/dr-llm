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
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
import typer

from dr_llm.demo import (
    DemoPrompts,
    DemoProviderWorkFailedError,
    collect_streaming_log_events,
    command_hint,
    build_live_demo_request,
    DemoNatsOptions,
    fail,
    ok,
    open_streaming_log_demo_runtime,
    payload_attempt_failure_details,
    step,
    StreamingLogDemoRuntime,
    StreamingLogDemoRuntimeOptions,
    demo_provider_candidates,
    verify_payload_refs,
    verify_response_payload,
    verify_successful_work_lifecycle,
    warn,
)
from dr_llm.llm import (
    LlmRequest,
    ProviderName,
)
from dr_llm.streaming_log import (
    EventEnvelope,
    QueuedWorkMessage,
    StreamingWorkerConfig,
    run_streaming_worker,
)

app = typer.Typer()


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
    candidates = demo_provider_candidates(
        requested_provider=options.provider,
        requested_model=options.model,
    )
    allow_fallback = options.provider is None and options.model is None
    if not allow_fallback:
        candidates = candidates[:1]
    failures: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        request = build_live_demo_request(
            prompt=options.prompt,
            provider=candidate.provider,
            model=candidate.model,
        )
        ok(f"Using {candidate.provider}/{candidate.model}")
        try:
            await _run_worker_attempt(
                WorkerAttemptOptions(
                    nats=options.nats,
                    max_retries=options.max_retries,
                    provider=candidate.provider,
                    model=candidate.model,
                    request=request,
                    attempt_index=index,
                    attempt_count=len(candidates),
                )
            )
            if failures:
                ok(
                    "Fallback succeeded after skipped provider attempts: "
                    + "; ".join(failures)
                )
            return
        except DemoProviderWorkFailedError as exc:
            failures.append(f"{candidate.provider}/{candidate.model}: {exc}")
            if not allow_fallback or index == len(candidates):
                raise
            warn(
                f"{candidate.provider}/{candidate.model} failed during live execution; "
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
            verified_payloads = await verify_payload_refs(
                session.payload_store, events
            )
            _print_worker_event_summary(events)
            await payload_attempt_failure_details(
                session.payload_store, events
            )
            verify_successful_work_lifecycle(
                events,
                work_id=work.work_id,
                require_producer_lifecycle=True,
                reject_attempt_failed=options.max_retries == 0,
            )
            response = await verify_response_payload(
                session.payload_store, events, work_id=work.work_id
            )
            ok(
                "Response payload verified: "
                f"text_preview={response.text_preview!r}"
            )

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

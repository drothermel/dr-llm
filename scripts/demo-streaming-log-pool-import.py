#!/usr/bin/env python3
"""Demo: import a real existing pool into the NATS streaming log.

Usage:
  uv run python scripts/demo-streaming-log-pool-import.py \
    --dsn postgresql://postgres:postgres@localhost:5433/dr_llm_test \
    --pool-name demo_pool_fill

Prerequisites:
  1. A real existing Postgres-backed pool.
  2. Docker running, unless --nats-url points at an existing NATS server.

The demo:
  - Starts a temporary NATS JetStream server when --nats-url is omitted
  - Bootstraps isolated demo streaming-log resources
  - Imports the requested pool using snapshot import semantics
  - Replays the event log and verifies counts and payload references
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
import typer

from dr_llm.demo import (
    collect_streaming_log_events,
    command_hint,
    DemoNatsOptions,
    fail,
    ok,
    open_streaming_log_demo_runtime,
    step,
    summarize_events,
    StreamingLogDemoRuntime,
    StreamingLogDemoRuntimeOptions,
    verify_payload_refs,
)
from dr_llm.pool import DbConfig, DbRuntime, PoolReader
from dr_llm.streaming_log import EventEnvelope, StreamingLogEventType
from dr_llm.streaming_log.ingest_pools import PoolImportResult, ingest_pool

app = typer.Typer()


class PoolImportDemoOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dsn: str
    pool_name: str
    nats: DemoNatsOptions = Field(default_factory=DemoNatsOptions)
    source_id: str | None = None
    sample_limit: int | None = None
    event_sample_limit: int = 5


PoolImportDemoOptions.model_rebuild(
    _types_namespace={"DemoNatsOptions": DemoNatsOptions}
)


async def _run_import_demo(options: PoolImportDemoOptions) -> None:
    step("1. Inspecting source pool")
    source_total = _print_source_pool_summary(
        dsn=options.dsn,
        pool_name=options.pool_name,
    )
    expected_import_count = (
        min(source_total, options.sample_limit)
        if options.sample_limit is not None
        else source_total
    )
    if options.sample_limit is not None:
        ok(f"Limiting import to {options.sample_limit} source samples")

    runtime: StreamingLogDemoRuntime | None = None
    events: list[EventEnvelope] = []
    verified_payloads = []
    try:
        step("2. Preparing streaming-log runtime")
        async with open_streaming_log_demo_runtime(
            StreamingLogDemoRuntimeOptions(nats=options.nats)
        ) as session:
            runtime = session.runtime
            _print_runtime_summary(runtime)

            step("3. Importing pool")
            result = await ingest_pool(
                event_log=session.event_log,
                dsn=options.dsn,
                pool_name=options.pool_name,
                source_id=options.source_id,
                sample_limit=options.sample_limit,
            )
            ok(
                f"Imported {result.imported_count} samples "
                f"with {len(result.event_ids)} emitted events"
            )

            step("4. Replaying and verifying events")
            events = await collect_streaming_log_events(
                session.event_log,
                expected_min_events=result.imported_count + 2,
            )
            _verify_import_events(
                expected_import_count=expected_import_count,
                result=result,
                events=events,
                pool_name=options.pool_name,
                source_id=options.source_id or options.dsn,
            )
            verified_payloads = await verify_payload_refs(
                session.payload_store, events
            )

        ok(f"Verified {len(events)} replayed events")
        ok(f"Verified {len(verified_payloads)} payload references")
        _print_event_sample(events, options.event_sample_limit)
        ok("Streaming-log pool import demo verified live data")
    finally:
        _print_cleanup_hint(runtime)


def _print_runtime_summary(runtime: StreamingLogDemoRuntime) -> None:
    ok(f"NATS ready at {runtime.lease.nats_url}")
    ok(f"Using event stream {runtime.config.events_stream}")
    ok(f"Using work stream {runtime.config.work_stream}")
    ok(f"Using payload bucket {runtime.config.payload_bucket}")
    ok(f"Events subjects: {', '.join(runtime.status.events_subjects)}")
    ok(f"Work subjects: {', '.join(runtime.status.work_subjects)}")


def _print_cleanup_hint(runtime: StreamingLogDemoRuntime | None) -> None:
    if runtime is None or runtime.lease.should_destroy_container:
        return
    if runtime.cleanup_command is not None:
        command_hint("Destroy NATS", runtime.cleanup_command)


def _print_source_pool_summary(*, dsn: str, pool_name: str) -> int:
    runtime = DbRuntime(DbConfig(dsn=dsn, min_pool_size=1, max_pool_size=4))
    try:
        reader = PoolReader.open(pool_name, runtime=runtime)
        progress = reader.progress()
        ok(f"Pool {pool_name!r} found")
        print(
            "  source counts: "
            f"total={progress.total} "
            f"incomplete={progress.incomplete} "
            f"complete={progress.complete} "
            f"error={progress.error} "
            f"leased={progress.leased}"
        )
        return progress.total
    finally:
        runtime.close()


def _verify_import_events(
    *,
    expected_import_count: int,
    result: PoolImportResult,
    events: list[EventEnvelope],
    pool_name: str,
    source_id: str,
) -> None:
    import_events = _events_for_pool_import(
        events, pool_name=pool_name, source_id=source_id
    )
    event_ids = [event.event_id for event in import_events]
    if event_ids != result.event_ids:
        raise RuntimeError(
            "Replayed import event IDs do not match importer result: "
            f"{event_ids!r} != {result.event_ids!r}"
        )
    counts = summarize_events(import_events)
    sample_events = counts["pool_sample_imported"]
    if result.imported_count != expected_import_count:
        raise RuntimeError(
            "Importer returned "
            f"{result.imported_count} rows, expected {expected_import_count}"
        )
    if sample_events != expected_import_count:
        raise RuntimeError(
            "Replayed "
            f"{sample_events} sample events, expected {expected_import_count}"
        )
    if counts["pool_import_started"] != 1:
        raise RuntimeError("Expected exactly one pool_import_started event")
    if counts["pool_import_completed"] != 1:
        raise RuntimeError("Expected exactly one pool_import_completed event")
    if counts["pool_import_failed"] != 0:
        raise RuntimeError("Expected no pool_import_failed events")
    completed = _single_import_event(
        import_events,
        StreamingLogEventType.pool_import_completed,
        pool_name=pool_name,
    )
    completed_count = getattr(completed.payload, "imported_count", None)
    if completed_count != result.imported_count:
        raise RuntimeError(
            "pool_import_completed imported_count mismatch: "
            f"{completed_count!r} != {result.imported_count!r}"
        )
    for event in import_events:
        if str(event.event_type) != "pool_sample_imported":
            continue
        _verify_sample_import_event(
            event, pool_name=pool_name, source_id=source_id
        )
    ok(
        "Event counts verified: "
        f"started={counts['pool_import_started']} "
        f"samples={sample_events} "
        f"completed={counts['pool_import_completed']}"
    )


def _events_for_pool_import(
    events: list[EventEnvelope], *, pool_name: str, source_id: str
) -> list[EventEnvelope]:
    return [
        event
        for event in events
        if getattr(event.payload, "pool_name", None) == pool_name
        and getattr(event.payload, "source_id", None) == source_id
    ]


def _single_import_event(
    events: list[EventEnvelope],
    event_type: StreamingLogEventType,
    *,
    pool_name: str,
) -> EventEnvelope:
    matches = [event for event in events if event.event_type == event_type]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {event_type} event for {pool_name}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _verify_sample_import_event(
    event: EventEnvelope, *, pool_name: str, source_id: str
) -> None:
    payload = event.payload
    if getattr(payload, "pool_name", None) != pool_name:
        raise RuntimeError("pool_sample_imported pool_name mismatch")
    if getattr(payload, "source_id", None) != source_id:
        raise RuntimeError("pool_sample_imported source_id mismatch")
    sample_id = getattr(payload, "sample_id", "")
    if not isinstance(sample_id, str) or not sample_id:
        raise RuntimeError("pool_sample_imported has no sample_id")
    row_state_hash = getattr(payload, "row_state_hash", "")
    if not isinstance(row_state_hash, str) or not row_state_hash:
        raise RuntimeError("pool_sample_imported has no row_state_hash")
    roles = {ref.role for ref in event.payload_refs}
    required_roles = {"pool_schema", "request_json", "metadata_json"}
    missing = sorted(required_roles - roles)
    if missing:
        raise RuntimeError(
            "pool_sample_imported missing payload roles: " + ", ".join(missing)
        )


def _print_event_sample(events: list[EventEnvelope], limit: int) -> None:
    step("6. Event sample")
    for event in events[:limit]:
        roles = [ref.role for ref in event.payload_refs]
        print(
            f"  {event.event_type} "
            f"event_id={event.event_id} "
            f"work_id={event.work_id or '-'} "
            f"payload_roles={roles or '-'}"
        )


@app.command()
def main(
    dsn: Annotated[str, typer.Option(help="PostgreSQL DSN.")],
    pool_name: Annotated[
        str,
        typer.Option(help="Existing pool name to import."),
    ],
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
    source_id: Annotated[
        str | None,
        typer.Option(help="Stable source database identifier."),
    ] = None,
    sample_limit: Annotated[
        int | None,
        typer.Option(help="Import at most this many source samples."),
    ] = None,
    event_sample_limit: Annotated[
        int,
        typer.Option(help="Number of replayed events to print."),
    ] = 5,
) -> None:
    try:
        asyncio.run(
            _run_import_demo(
                PoolImportDemoOptions(
                    dsn=dsn,
                    pool_name=pool_name,
                    nats=DemoNatsOptions(
                        nats_url=nats_url,
                        keep_nats=keep_nats,
                    ),
                    source_id=source_id,
                    sample_limit=sample_limit,
                    event_sample_limit=event_sample_limit,
                )
            )
        )
    except Exception as exc:
        fail(str(exc))
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()

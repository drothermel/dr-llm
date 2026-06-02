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

import typer

from dr_llm.demo import (
    cleanup_demo_nats,
    collect_streaming_log_events,
    command_hint,
    demo_streaming_log_config,
    fail,
    ok,
    prepare_demo_nats,
    step,
    summarize_events,
    verify_payload_refs,
    wait_for_nats,
)
from dr_llm.pool import DbConfig, DbRuntime, PoolReader
from dr_llm.streaming_log import StreamingLogClient
from dr_llm.streaming_log.bootstrap import bootstrap_streaming_log
from dr_llm.streaming_log.ingest_pools import ingest_pool

app = typer.Typer()


async def _run_import_demo(
    *,
    dsn: str,
    pool_name: str,
    nats_url: str | None,
    keep_nats: bool,
    source_id: str | None,
    sample_limit: int | None,
    event_sample_limit: int,
) -> None:
    step("1. Inspecting source pool")
    source_total = _print_source_pool_summary(dsn=dsn, pool_name=pool_name)
    expected_import_count = (
        min(source_total, sample_limit)
        if sample_limit is not None
        else source_total
    )
    if sample_limit is not None:
        ok(f"Limiting import to {sample_limit} source samples")

    step("2. Preparing NATS")
    lease = prepare_demo_nats(nats_url=nats_url, keep_nats=keep_nats)
    ok(f"NATS ready at {lease.nats_url}")
    await wait_for_nats(lease.nats_url)
    config = demo_streaming_log_config(nats_url=lease.nats_url)
    ok(f"Using event stream {config.events_stream}")
    ok(f"Using work stream {config.work_stream}")
    ok(f"Using payload bucket {config.payload_bucket}")

    try:
        step("3. Bootstrapping streaming-log resources")
        status = await bootstrap_streaming_log(config)
        ok(f"Events subjects: {', '.join(status.events_subjects)}")
        ok(f"Work subjects: {', '.join(status.work_subjects)}")

        step("4. Importing pool")
        async with StreamingLogClient(config) as client:
            result = await ingest_pool(
                client=client,
                dsn=dsn,
                pool_name=pool_name,
                source_id=source_id,
                sample_limit=sample_limit,
            )
            ok(
                f"Imported {result.imported_count} samples "
                f"with {len(result.event_ids)} emitted events"
            )

            step("5. Replaying and verifying events")
            events = await collect_streaming_log_events(
                client,
                expected_min_events=result.imported_count + 2,
            )
            counts = summarize_events(events)
            _verify_import_counts(
                expected_import_count=expected_import_count,
                imported_count=result.imported_count,
                counts=counts,
            )
            verified_payloads = await verify_payload_refs(client, events)

        ok(f"Verified {len(events)} replayed events")
        ok(f"Verified {len(verified_payloads)} payload references")
        _print_event_sample(events, event_sample_limit)
        ok("Streaming-log pool import demo verified live data")
    finally:
        if lease.should_destroy_container and lease.container_name is not None:
            step("Destroying temporary NATS")
            cleanup_demo_nats(lease)
        elif lease.container_name is not None:
            command_hint(
                "Destroy NATS",
                f"docker rm -f {lease.container_name}",
            )


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


def _verify_import_counts(
    *,
    expected_import_count: int,
    imported_count: int,
    counts,
) -> None:
    sample_events = counts["pool_sample_imported"]
    if imported_count != expected_import_count:
        raise RuntimeError(
            "Importer returned "
            f"{imported_count} rows, expected {expected_import_count}"
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
    ok(
        "Event counts verified: "
        f"started={counts['pool_import_started']} "
        f"samples={sample_events} "
        f"completed={counts['pool_import_completed']}"
    )


def _print_event_sample(events, limit: int) -> None:
    step("6. Event sample")
    for event in events[:limit]:
        roles = [
            str(ref.get("role"))
            for ref in event.payload_refs
            if isinstance(ref, dict)
        ]
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
                dsn=dsn,
                pool_name=pool_name,
                nats_url=nats_url,
                keep_nats=keep_nats,
                source_id=source_id,
                sample_limit=sample_limit,
                event_sample_limit=event_sample_limit,
            )
        )
    except Exception as exc:
        fail(str(exc))
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()

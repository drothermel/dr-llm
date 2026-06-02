#!/usr/bin/env python3
"""Demo: project synthetic streaming-log payloads into artifact storage.

Usage:
  uv run python scripts/demo-artifact-projection.py
  uv run python scripts/demo-artifact-projection.py --keep-nats
  uv run python scripts/demo-artifact-projection.py --artifact-root /tmp/dr-llm-artifacts

Prerequisites:
  1. Docker running, unless --nats-url points at an existing NATS server.

The demo:
  - Starts a temporary NATS JetStream server when --nats-url is omitted
  - Publishes one event with duplicate artifact-bearing payload refs
  - Runs the artifact projector for one event
  - Verifies idempotent projection, finalized readback, and open-reference cleanup
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from dr_llm.artifact_projection import (
    ArtifactProjectionConfig,
    ArtifactReader,
    ArtifactStore,
)
from dr_llm.artifact_projection.projector import run_artifact_projector
from dr_llm.demo import (
    cleanup_demo_nats,
    command_hint,
    demo_streaming_log_config,
    fail,
    ok,
    prepare_demo_nats,
    step,
    wait_for_nats,
)
from dr_llm.streaming_log import (
    StreamingEventLog,
    StreamingLogConnection,
    StreamingPayloadStore,
    StreamingLogEventType,
    bootstrap_streaming_log,
)
from dr_llm.streaming_log.payloads import prepare_json_payload

app = typer.Typer()


async def _run_artifact_demo(
    *,
    nats_url: str | None,
    keep_nats: bool,
    artifact_root: Path,
) -> None:
    step("1. Preparing NATS")
    lease = prepare_demo_nats(nats_url=nats_url, keep_nats=keep_nats)
    ok(f"NATS ready at {lease.nats_url}")
    await wait_for_nats(lease.nats_url)
    streaming_config = demo_streaming_log_config(nats_url=lease.nats_url)
    artifact_config = ArtifactProjectionConfig(artifact_root=artifact_root)
    ok(f"Using event stream {streaming_config.events_stream}")
    ok(f"Using payload bucket {streaming_config.payload_bucket}")
    ok(f"Using artifact root {artifact_config.artifact_root}")

    try:
        step("2. Bootstrapping streaming-log resources")
        await bootstrap_streaming_log(streaming_config)

        async with StreamingLogConnection(streaming_config) as connection:
            payload_store = StreamingPayloadStore(connection)
            event_log = StreamingEventLog(connection, payload_store)
            payload = prepare_json_payload(
                "response_json",
                {"demo": "artifact-projection", "ok": True},
            )

            step("3. Publishing duplicate artifact payload refs")
            event = await event_log.publish_event_with_payloads(
                StreamingLogEventType.provider_response_received,
                idempotency_key="demo-artifact-projection-1",
                payload={"provider": "demo"},
                payloads=[payload, payload],
            )
            ok(f"Published event_id={event.event_id}")

            step("4. Running artifact projector")
            processed = await run_artifact_projector(
                connection=connection,
                config=artifact_config,
                max_messages=1,
            )
            if processed != 1:
                raise RuntimeError(f"Processed {processed} events, expected 1")

        step("5. Verifying artifact index and readback")
        store = ArtifactStore(config=artifact_config)
        store.initialize()
        references = store.index.list_references()
        summary = store.index.summary(
            projection_version=artifact_config.projection_version,
            durable_consumer=artifact_config.durable_consumer,
        )
        if len(references) != 1:
            raise RuntimeError(
                f"Projected {len(references)} artifacts, expected 1"
            )
        if summary.open_artifact_count != 0:
            raise RuntimeError(
                "Expected no open references after projector finalization"
            )
        data = ArtifactReader(artifact_config).read_json(references[0])
        if data != {"demo": "artifact-projection", "ok": True}:
            raise RuntimeError(f"Unexpected artifact payload: {data!r}")
        ok(
            "Artifact projection verified: "
            f"finalized={summary.artifact_count} "
            f"open={summary.open_artifact_count}"
        )
    finally:
        if lease.should_destroy_container and lease.container_name is not None:
            step("Destroying temporary NATS")
            cleanup_demo_nats(lease)
        elif lease.container_name is not None:
            command_hint(
                "Destroy NATS",
                f"docker rm -f {lease.container_name}",
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
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            help=("Artifact root. If omitted, a temporary directory is used.")
        ),
    ] = None,
) -> None:
    try:
        if artifact_root is not None:
            asyncio.run(
                _run_artifact_demo(
                    nats_url=nats_url,
                    keep_nats=keep_nats,
                    artifact_root=artifact_root,
                )
            )
            return
        with tempfile.TemporaryDirectory(
            prefix="dr_llm_artifacts_"
        ) as artifact_dir:
            asyncio.run(
                _run_artifact_demo(
                    nats_url=nats_url,
                    keep_nats=keep_nats,
                    artifact_root=Path(artifact_dir),
                )
            )
    except Exception as exc:
        fail(str(exc))
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()

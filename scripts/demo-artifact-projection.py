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

from pydantic import BaseModel, ConfigDict, Field
import typer

from dr_llm.artifact_projection import (
    ArtifactProjectionConfig,
    ArtifactReader,
    ArtifactStore,
)
from dr_llm.artifact_projection.projector import run_artifact_projector
from dr_llm.demo import (
    command_hint,
    DemoNatsOptions,
    fail,
    ok,
    open_streaming_log_demo_runtime,
    step,
    StreamingLogDemoRuntime,
    StreamingLogDemoRuntimeOptions,
)
from dr_llm.streaming_log import (
    ProviderResponseReceivedPayload,
    StreamingEventPublishSpec,
    StreamingLogEventType,
)
from dr_llm.streaming_log.payloads import prepare_json_payload

app = typer.Typer()


class ArtifactProjectionDemoOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nats: DemoNatsOptions = Field(default_factory=DemoNatsOptions)
    artifact_root: Path


ArtifactProjectionDemoOptions.model_rebuild(
    _types_namespace={
        "DemoNatsOptions": DemoNatsOptions,
        "Path": Path,
    }
)


async def _run_artifact_demo(options: ArtifactProjectionDemoOptions) -> None:
    artifact_config = ArtifactProjectionConfig(
        artifact_root=options.artifact_root
    )
    ok(f"Using artifact root {artifact_config.artifact_root}")

    runtime: StreamingLogDemoRuntime | None = None
    try:
        step("1. Preparing streaming-log runtime")
        async with open_streaming_log_demo_runtime(
            StreamingLogDemoRuntimeOptions(nats=options.nats)
        ) as session:
            runtime = session.runtime
            _print_runtime_summary(runtime)
            payload = prepare_json_payload(
                "response_json",
                {"demo": "artifact-projection", "ok": True},
            )

            step("2. Publishing duplicate artifact payload refs")
            event = await session.event_log.publish_event_spec(
                StreamingEventPublishSpec(
                    event_type=StreamingLogEventType.provider_response_received,
                    idempotency_key="demo-artifact-projection-1",
                    payload=ProviderResponseReceivedPayload(
                        provider="demo",
                        model="demo-model",
                        mode="api",
                    ),
                    payloads=[payload, payload],
                )
            )
            ok(f"Published event_id={event.event_id}")

            step("3. Running artifact projector")
            processed = await run_artifact_projector(
                connection=session.connection,
                config=artifact_config,
                max_messages=1,
            )
            if processed != 1:
                raise RuntimeError(f"Processed {processed} events, expected 1")

        step("4. Verifying artifact index and readback")
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
        _print_cleanup_hint(runtime)


def _print_runtime_summary(runtime: StreamingLogDemoRuntime) -> None:
    ok(f"NATS ready at {runtime.lease.nats_url}")
    ok(f"Using event stream {runtime.config.events_stream}")
    ok(f"Using payload bucket {runtime.config.payload_bucket}")


def _print_cleanup_hint(runtime: StreamingLogDemoRuntime | None) -> None:
    if runtime is None or runtime.lease.should_destroy_container:
        return
    if runtime.cleanup_command is not None:
        command_hint("Destroy NATS", runtime.cleanup_command)


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
                    ArtifactProjectionDemoOptions(
                        nats=DemoNatsOptions(
                            nats_url=nats_url,
                            keep_nats=keep_nats,
                        ),
                        artifact_root=artifact_root,
                    )
                )
            )
            return
        with tempfile.TemporaryDirectory(
            prefix="dr_llm_artifacts_"
        ) as artifact_dir:
            asyncio.run(
                _run_artifact_demo(
                    ArtifactProjectionDemoOptions(
                        nats=DemoNatsOptions(
                            nats_url=nats_url,
                            keep_nats=keep_nats,
                        ),
                        artifact_root=Path(artifact_dir),
                    )
                )
            )
    except Exception as exc:
        fail(str(exc))
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()

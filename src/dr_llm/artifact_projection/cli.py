from __future__ import annotations

import asyncio
import sys
from typing import Annotated

import typer
from rich.console import Console

from dr_llm.artifact_projection.config import ArtifactProjectionConfig
from dr_llm.artifact_projection.index import ArtifactIndex
from dr_llm.artifact_projection.shards import ArtifactReader
from dr_llm.artifact_projection.store import ArtifactStore
from dr_llm.streaming_log.client import StreamingLogConnection
from dr_llm.streaming_log.config import StreamingLogConfig


artifact_projection_app = typer.Typer(
    help="Artifact projection storage commands"
)
console = Console()


@artifact_projection_app.command("init")
def init() -> None:
    """Create artifact projection directories and sidecar schema."""
    store = ArtifactStore(config=ArtifactProjectionConfig())
    store.initialize()
    console.print_json(data=_store_summary(store))


@artifact_projection_app.command("inspect")
def inspect() -> None:
    """Print artifact projection storage status."""
    config = ArtifactProjectionConfig()
    with ArtifactIndex(config.index_path) as index:
        index.initialize()
        summary = index.summary(
            projection_version=config.projection_version,
            durable_consumer=config.durable_consumer,
        )
    console.print_json(
        data={
            "artifact_root": str(config.artifact_root),
            "projection_version": config.projection_version,
            **summary.model_dump(mode="json"),
        }
    )


@artifact_projection_app.command("run")
def run(
    max_messages: Annotated[
        int | None,
        typer.Option("--max-messages", help="Stop after this many events."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="JetStream fetch batch size."),
    ] = None,
    flush_on_exit: Annotated[
        bool,
        typer.Option("--flush-on-exit/--no-flush-on-exit"),
    ] = True,
) -> None:
    """Run the artifact projector against DRLLM_EVENTS."""
    processed = asyncio.run(
        _run_projector(
            max_messages=max_messages,
            batch_size=batch_size,
            flush_on_exit=flush_on_exit,
        )
    )
    console.print_json(data={"processed": processed})


@artifact_projection_app.command("flush")
def flush() -> None:
    """Finalize any in-process artifact shard."""
    store = ArtifactStore(config=ArtifactProjectionConfig())
    store.initialize()
    manifest = store.finalize()
    console.print_json(
        data={
            "finalized": manifest is not None,
            "manifest": (
                manifest.model_dump(mode="json")
                if manifest is not None
                else None
            ),
        }
    )


@artifact_projection_app.command("rebuild-index")
def rebuild_index() -> None:
    """Rebuild the sidecar index from finalized manifests."""
    store = ArtifactStore(config=ArtifactProjectionConfig())
    store.rebuild_index()
    console.print_json(data=_store_summary(store))


@artifact_projection_app.command("verify")
def verify() -> None:
    """Verify basic index and projection-error status."""
    config = ArtifactProjectionConfig()
    with ArtifactIndex(config.index_path) as index:
        index.initialize()
        summary = index.summary(
            projection_version=config.projection_version,
            durable_consumer=config.durable_consumer,
        )
    console.print_json(data=summary.model_dump(mode="json"))
    if summary.error_count:
        raise typer.Exit(1)


@artifact_projection_app.command("read")
def read(
    artifact_id: Annotated[str, typer.Argument(help="Artifact ID to read.")],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output format: bytes, text, or json.",
        ),
    ] = "text",
) -> None:
    """Read one finalized artifact."""
    config = ArtifactProjectionConfig()
    with ArtifactIndex(config.index_path) as index:
        reference = index.get_finalized_reference(artifact_id)
    if reference is None:
        raise typer.BadParameter(f"unknown artifact ID {artifact_id!r}")
    reader = ArtifactReader(config)
    if output == "bytes":
        sys.stdout.buffer.write(reader.read_bytes(reference))
        return
    if output == "json":
        console.print_json(data=reader.read_json(reference))
        return
    if output != "text":
        raise typer.BadParameter("output must be bytes, text, or json")
    console.print(reader.read_text(reference))


async def _run_projector(
    *,
    max_messages: int | None,
    batch_size: int | None,
    flush_on_exit: bool,
) -> int:
    async with StreamingLogConnection(StreamingLogConfig()) as connection:
        from dr_llm.artifact_projection.projector import (
            run_artifact_projector,
        )

        return await run_artifact_projector(
            connection=connection,
            config=ArtifactProjectionConfig(),
            max_messages=max_messages,
            batch_size=batch_size,
            flush_on_exit=flush_on_exit,
        )


def _store_summary(store: ArtifactStore) -> dict[str, object]:
    summary = store.index.summary(
        projection_version=store.config.projection_version,
        durable_consumer=store.config.durable_consumer,
    )
    return {
        "artifact_root": str(store.config.artifact_root),
        "projection_version": store.config.projection_version,
        **summary.model_dump(mode="json"),
    }


__all__ = ["artifact_projection_app"]

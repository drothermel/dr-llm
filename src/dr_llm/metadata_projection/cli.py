from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dr_llm.metadata_projection.artifact_links import (
    ArtifactAttachmentPlanner,
    load_index_references,
)
from dr_llm.metadata_projection.config import MetadataProjectionConfig
from dr_llm.metadata_projection.models import MetadataProjectionCheckpoint
from dr_llm.metadata_projection.projector import run_metadata_projector
from dr_llm.metadata_projection.store import MetadataStore
from dr_llm.streaming_log.client import StreamingLogConnection
from dr_llm.streaming_log.config import StreamingLogConfig


metadata_projection_app = typer.Typer(
    help="Metadata projection catalog commands"
)
console = Console()


@metadata_projection_app.command("init")
def init() -> None:
    """Create metadata projection tables."""
    store = MetadataStore(config=MetadataProjectionConfig())
    try:
        store.initialize()
        console.print_json(data=store.summary().model_dump(mode="json"))
    finally:
        store.close()


@metadata_projection_app.command("run")
def run(
    max_messages: Annotated[
        int | None,
        typer.Option("--max-messages", help="Stop after this many events."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="JetStream fetch batch size."),
    ] = None,
    from_start: Annotated[
        bool,
        typer.Option("--from-start", help="Use a fresh replay consumer."),
    ] = False,
    artifact_index_path: Annotated[
        Path | None,
        typer.Option("--artifact-index-path", help="Artifact sidecar index."),
    ] = None,
) -> None:
    """Run the metadata projector against DRLLM_EVENTS."""
    config = _config(artifact_index_path=artifact_index_path)
    processed = asyncio.run(
        _run_projector(
            config=config,
            max_messages=max_messages,
            batch_size=batch_size,
            from_start=from_start,
        )
    )
    console.print_json(data={"processed": processed})


@metadata_projection_app.command("attach-artifacts")
def attach_artifacts(
    artifact_index_path: Annotated[
        Path | None,
        typer.Option("--artifact-index-path", help="Artifact sidecar index."),
    ] = None,
) -> None:
    """Attach finalized artifact references to metadata facts."""
    config = _config(artifact_index_path=artifact_index_path)
    attached = attach_finalized_artifacts(config)
    console.print_json(data={"attached": attached})


@metadata_projection_app.command("inspect")
def inspect() -> None:
    """Print metadata projection status."""
    store = MetadataStore(config=MetadataProjectionConfig())
    try:
        store.initialize()
        console.print_json(data=store.summary().model_dump(mode="json"))
    finally:
        store.close()


@metadata_projection_app.command("verify")
def verify() -> None:
    """Verify metadata projection consistency."""
    store = MetadataStore(config=MetadataProjectionConfig())
    try:
        store.initialize()
        result = store.verify()
        console.print_json(data=result.model_dump(mode="json"))
    finally:
        store.close()
    if not result.passed:
        raise typer.Exit(1)


@metadata_projection_app.command("rebuild")
def rebuild(
    max_messages: Annotated[
        int | None,
        typer.Option("--max-messages", help="Stop after this many events."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", help="JetStream fetch batch size."),
    ] = None,
    artifact_index_path: Annotated[
        Path | None,
        typer.Option("--artifact-index-path", help="Artifact sidecar index."),
    ] = None,
) -> None:
    """Clear metadata rows, replay events, then attach artifacts."""
    config = _config(artifact_index_path=artifact_index_path)
    store = MetadataStore(config=config)
    try:
        store.initialize()
        store.clear_rebuildable_rows()
    finally:
        store.close()
    processed = asyncio.run(
        _run_projector(
            config=config,
            max_messages=max_messages,
            batch_size=batch_size,
            from_start=True,
        )
    )
    attached = attach_finalized_artifacts(config)
    console.print_json(data={"processed": processed, "attached": attached})


def attach_finalized_artifacts(config: MetadataProjectionConfig) -> int:
    references = load_index_references(config.artifact_index_path)
    planner = ArtifactAttachmentPlanner(config)
    plan = planner.plan_references(references)
    store = MetadataStore(config=config)
    try:
        store.initialize()
        store.apply_write_plan(
            plan,
            checkpoint=MetadataProjectionCheckpoint(
                projection_version=config.projection_version,
                durable_consumer=config.artifact_attach_consumer,
                stream_sequence=len(references),
            ),
        )
    finally:
        store.close()
    return len(references)


async def _run_projector(
    *,
    config: MetadataProjectionConfig,
    max_messages: int | None,
    batch_size: int | None,
    from_start: bool,
) -> int:
    async with StreamingLogConnection(StreamingLogConfig()) as connection:
        return await run_metadata_projector(
            connection=connection,
            config=config,
            max_messages=max_messages,
            batch_size=batch_size,
            from_start=from_start,
        )


def _config(
    *, artifact_index_path: Path | None = None
) -> MetadataProjectionConfig:
    config = MetadataProjectionConfig()
    if artifact_index_path is None:
        return config
    return config.model_copy(
        update={"artifact_index_path": artifact_index_path}
    )


__all__ = [
    "attach_finalized_artifacts",
    "metadata_projection_app",
]

#!/usr/bin/env python3
"""Demo: verify streaming log, artifact projection, and metadata projection.

Usage:
  uv run python scripts/demo-metadata-projection-e2e.py
  uv run python scripts/demo-metadata-projection-e2e.py --keep-project --keep-nats

Prerequisites:
  1. Docker running, unless --dsn and --nats-url point at live systems.
  2. At least one real provider available through env vars or provider CLI.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
import typer

from dr_llm.artifact_projection import ArtifactProjectionConfig
from dr_llm.artifact_projection.projector import run_artifact_projector
from dr_llm.demo import (
    DEMO_QUERY_DEFAULT_MODELS,
    DemoDsnLease,
    DemoNatsOptions,
    DemoPrompts,
    cleanup_demo_dsn,
    collect_streaming_log_events,
    command_hint,
    fail,
    list_models_json,
    ok,
    open_streaming_log_demo_runtime,
    prepare_demo_dsn,
    step,
    StreamingLogDemoRuntime,
    StreamingLogDemoRuntimeOptions,
    sync_models_json,
    warn,
)
from dr_llm.llm import Message, ProviderName, build_default_registry
from dr_llm.metadata_projection import MetadataProjectionConfig, MetadataStore
from dr_llm.metadata_projection.cli import attach_finalized_artifacts
from dr_llm.metadata_projection.projector import run_metadata_projector
from dr_llm.streaming_log import QueuedWorkMessage, StreamingWorkerConfig
from dr_llm.streaming_log.workers import run_streaming_worker


app = typer.Typer()


class MetadataProjectionE2EOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dsn: str | None = None
    project_name: str | None = None
    keep_project: bool = False
    nats: DemoNatsOptions = Field(default_factory=DemoNatsOptions)
    artifact_root: Path
    provider: str | None = None
    model: str | None = None


MetadataProjectionE2EOptions.model_rebuild(
    _types_namespace={"DemoNatsOptions": DemoNatsOptions, "Path": Path}
)


async def _run_demo(options: MetadataProjectionE2EOptions) -> None:
    lease = prepare_demo_dsn(
        dsn=options.dsn,
        project_prefix="demo_metadata_projection",
        project_name=options.project_name,
        keep_project=options.keep_project,
    )
    runtime: StreamingLogDemoRuntime | None = None
    try:
        artifact_config = ArtifactProjectionConfig(
            artifact_root=options.artifact_root
        )
        metadata_config = MetadataProjectionConfig(
            database_dsn=lease.dsn,
            artifact_index_path=artifact_config.index_path,
        )
        _initialize_metadata(metadata_config)

        step("1. Preparing streaming-log runtime")
        async with open_streaming_log_demo_runtime(
            StreamingLogDemoRuntimeOptions(nats=options.nats)
        ) as session:
            runtime = session.runtime
            provider, model = _resolve_provider_model(options)
            ok(f"Using provider/model {provider}/{model}")

            step("2. Submitting real queued work")
            work = _queued_work(provider=provider, model=model)
            await session.work_queue.submit_work(work)
            ok(f"Submitted work_id={work.work_id}")

            step("3. Running streaming worker")
            await run_streaming_worker(
                work_queue=session.work_queue,
                config=StreamingWorkerConfig(
                    worker_id="metadata-demo-worker",
                    max_messages=1,
                ),
            )

            step("4. Replaying emitted events")
            events = await collect_streaming_log_events(
                session.event_log, expected_min_events=6
            )
            ok(f"Collected {len(events)} streaming events")

            step("5. Running artifact projection")
            artifact_count = await run_artifact_projector(
                connection=session.connection,
                config=artifact_config,
                max_messages=len(events),
            )
            ok(f"Artifact projector consumed {artifact_count} events")

            step("6. Running metadata projection")
            projected_count = await run_metadata_projector(
                connection=session.connection,
                config=metadata_config,
                max_messages=len(events),
            )
            ok(f"Metadata projector consumed {projected_count} events")

            step("7. Attaching finalized artifacts")
            attached = attach_finalized_artifacts(metadata_config)
            ok(f"Attached {attached} artifacts")

            step("8. Verifying replay and rebuild")
            _verify_projection(metadata_config, expected_events=len(events))
            await _verify_replay(
                metadata_config=metadata_config,
                connection=session.connection,
                event_count=len(events),
            )
            await _verify_rebuild(
                metadata_config=metadata_config,
                connection=session.connection,
                event_count=len(events),
            )

        ok("Metadata projection end-to-end demo verified live systems")
    finally:
        _print_cleanup_hints(runtime=runtime, lease=lease)
        cleanup_demo_dsn(lease)


def _initialize_metadata(config: MetadataProjectionConfig) -> None:
    store = MetadataStore(config=config)
    try:
        store.initialize()
    finally:
        store.close()


def _resolve_provider_model(
    options: MetadataProjectionE2EOptions,
) -> tuple[str, str]:
    registry = build_default_registry()
    try:
        available = [
            status
            for status in registry.availability_statuses()
            if status.available
        ]
        if not available:
            raise RuntimeError(
                "No providers available. Set provider API keys or install a "
                "supported provider CLI."
            )
        available_names = {status.provider for status in available}
        provider = options.provider or available[0].provider
        if provider not in available_names:
            raise RuntimeError(
                f"Requested provider {provider!r} is not available. "
                f"Available providers: {sorted(available_names)}"
            )
        model = options.model or _default_model(provider)
        return provider, model
    finally:
        registry.close()


def _default_model(provider: str) -> str:
    provider_name = ProviderName(provider)
    default_model = DEMO_QUERY_DEFAULT_MODELS.get(provider_name)
    sync_models_json(provider_name)
    model_ids = [item["model"] for item in list_models_json(provider_name)]
    if default_model in model_ids:
        return default_model
    if model_ids:
        warn(f"Default model unavailable for {provider}; using {model_ids[0]}")
        return model_ids[0]
    raise RuntimeError(f"No models found for provider {provider!r}")


def _queued_work(*, provider: str, model: str) -> QueuedWorkMessage:
    registry = build_default_registry()
    try:
        request = registry.get(provider).build_request(
            model=model,
            messages=[Message(role="user", content=str(DemoPrompts.EXACT_OK))],
            max_tokens=64,
        )
    finally:
        registry.close()
    return QueuedWorkMessage(
        request=request,
        run_id="metadata-demo-run",
        source="metadata-projection-e2e",
        metadata={"demo": "metadata-projection-e2e"},
        max_retries=0,
    )


def _verify_projection(
    config: MetadataProjectionConfig, *, expected_events: int
) -> None:
    store = MetadataStore(config=config)
    try:
        summary = store.summary()
        result = store.verify()
    finally:
        store.close()
    if not result.passed:
        raise RuntimeError(f"Metadata verification failed: {result.problems}")
    if summary.assertion_count < expected_events:
        raise RuntimeError(
            f"Projected {summary.assertion_count} assertions, "
            f"expected at least {expected_events}"
        )
    ok(
        "Metadata verified: "
        f"entities={summary.entity_count} "
        f"assertions={summary.assertion_count} "
        f"errors={summary.error_count}"
    )


async def _verify_replay(
    *,
    metadata_config: MetadataProjectionConfig,
    connection,
    event_count: int,
) -> None:
    before = _summary_counts(metadata_config)
    await run_metadata_projector(
        connection=connection,
        config=metadata_config,
        max_messages=event_count,
        from_start=True,
    )
    after = _summary_counts(metadata_config)
    if after != before:
        raise RuntimeError(
            f"Replay changed metadata counts: {before} -> {after}"
        )
    ok("Replay idempotency verified")


async def _verify_rebuild(
    *,
    metadata_config: MetadataProjectionConfig,
    connection,
    event_count: int,
) -> None:
    store = MetadataStore(config=metadata_config)
    try:
        before = store.summary()
        store.clear_rebuildable_rows()
    finally:
        store.close()
    await run_metadata_projector(
        connection=connection,
        config=metadata_config,
        max_messages=event_count,
        from_start=True,
    )
    attach_finalized_artifacts(metadata_config)
    after = _summary_counts(metadata_config)
    expected = (
        before.assertion_count,
        before.role_count,
        before.error_count,
    )
    if after != expected:
        raise RuntimeError(
            f"Rebuild changed metadata counts: {expected} -> {after}"
        )
    ok("Rebuild determinism verified")


def _summary_counts(config: MetadataProjectionConfig) -> tuple[int, int, int]:
    store = MetadataStore(config=config)
    try:
        summary = store.summary()
    finally:
        store.close()
    return summary.assertion_count, summary.role_count, summary.error_count


def _print_cleanup_hints(
    *, runtime: StreamingLogDemoRuntime | None, lease: DemoDsnLease
) -> None:
    if runtime is not None and runtime.cleanup_command is not None:
        command_hint("Destroy NATS", runtime.cleanup_command)
    if lease.project_name is not None and not lease.should_destroy_project:
        command_hint(
            "Destroy Postgres",
            "uv run dr-llm project destroy "
            f"{lease.project_name} --yes-really-delete-everything",
        )


@app.command()
def main(
    dsn: Annotated[str | None, typer.Option(help="PostgreSQL DSN.")] = None,
    project_name: Annotated[
        str | None,
        typer.Option(help="Name for auto-created demo project."),
    ] = None,
    keep_project: Annotated[
        bool,
        typer.Option("--keep-project", help="Keep auto-created Postgres."),
    ] = False,
    nats_url: Annotated[
        str | None,
        typer.Option(help="NATS URL. If omitted, Docker NATS is created."),
    ] = None,
    keep_nats: Annotated[
        bool,
        typer.Option("--keep-nats", help="Keep auto-created NATS."),
    ] = False,
    artifact_root: Annotated[
        Path | None,
        typer.Option(help="Artifact root. If omitted, temp dir is used."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(help="Provider to use. Defaults to first available."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="Model to use. Defaults to demo provider model."),
    ] = None,
) -> None:
    try:
        if artifact_root is not None:
            asyncio.run(
                _run_demo(
                    MetadataProjectionE2EOptions(
                        dsn=dsn,
                        project_name=project_name,
                        keep_project=keep_project,
                        nats=DemoNatsOptions(
                            nats_url=nats_url,
                            keep_nats=keep_nats,
                        ),
                        artifact_root=artifact_root,
                        provider=provider,
                        model=model,
                    )
                )
            )
            return
        with tempfile.TemporaryDirectory(
            prefix="dr_llm_metadata_e2e_"
        ) as root:
            asyncio.run(
                _run_demo(
                    MetadataProjectionE2EOptions(
                        dsn=dsn,
                        project_name=project_name,
                        keep_project=keep_project,
                        nats=DemoNatsOptions(
                            nats_url=nats_url,
                            keep_nats=keep_nats,
                        ),
                        artifact_root=Path(root),
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

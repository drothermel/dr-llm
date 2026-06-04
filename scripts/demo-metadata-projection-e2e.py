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
from typing import Annotated, Any, cast

from pydantic import BaseModel, ConfigDict, Field
import typer

from dr_llm.artifact_projection import (
    ArtifactProjectionConfig,
    ArtifactReader,
    ArtifactStore,
)
from dr_llm.artifact_projection.projector import run_artifact_projector
from dr_llm.demo import (
    DemoDsnLease,
    DemoNatsOptions,
    DemoProviderCandidate,
    DemoPrompts,
    cleanup_demo_dsn,
    collect_streaming_log_events,
    command_hint,
    demo_provider_candidates,
    fail,
    ok,
    open_streaming_log_demo_runtime,
    prepare_demo_dsn,
    step,
    StreamingLogDemoRuntime,
    StreamingLogDemoRuntimeOptions,
    warn,
    verify_successful_work_lifecycle,
)
from dr_llm.llm import CallMode, Message, build_default_registry
from dr_llm.metadata_projection import MetadataProjectionConfig, MetadataStore
from dr_llm.metadata_projection.cli import attach_finalized_artifacts
from dr_llm.metadata_projection.projector import run_metadata_projector
from dr_llm.streaming_log import (
    EventEnvelope,
    QueuedWorkMessage,
    StreamingWorkerConfig,
)
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


class SuccessfulProviderRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: DemoProviderCandidate
    work: QueuedWorkMessage
    events: list[EventEnvelope]


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

            step("2. Running real provider work")
            provider_run = await _run_successful_provider_work(
                options=options,
                session=session,
            )
            events = provider_run.events
            work = provider_run.work
            ok(
                "Successful provider lifecycle: "
                f"{provider_run.candidate.provider}/"
                f"{provider_run.candidate.model} "
                f"work_id={work.work_id}"
            )

            step("3. Running artifact projection")
            artifact_count = await run_artifact_projector(
                connection=session.connection,
                config=artifact_config,
                max_messages=len(events),
            )
            ok(f"Artifact projector consumed {artifact_count} events")
            response_artifact = _verify_response_artifact(
                artifact_config=artifact_config,
                work_id=work.work_id,
            )
            ok(
                "Response artifact verified: "
                f"artifact_id={response_artifact['artifact_id']} "
                f"text_preview={response_artifact['text_preview']!r}"
            )

            step("4. Running metadata projection")
            projected_count = await run_metadata_projector(
                connection=session.connection,
                config=metadata_config,
                max_messages=len(events),
            )
            ok(f"Metadata projector consumed {projected_count} events")

            step("5. Attaching finalized artifacts")
            attached = attach_finalized_artifacts(metadata_config)
            ok(f"Attached {attached} artifacts")

            step("6. Verifying metadata, replay, and rebuild")
            _verify_projection(
                metadata_config,
                expected_events=len(events),
                expected_work_id=work.work_id,
            )
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


def _queued_work(*, provider: str, model: str) -> QueuedWorkMessage:
    registry = build_default_registry()
    try:
        orchestrator = registry.get(provider)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                Message(role="user", content=str(DemoPrompts.EXACT_OK))
            ],
        }
        if orchestrator.mode is CallMode.api:
            request_kwargs["max_tokens"] = 64
        request = orchestrator.build_request(**request_kwargs)
    finally:
        registry.close()
    return QueuedWorkMessage(
        request=request,
        run_id="metadata-demo-run",
        source="metadata-projection-e2e",
        metadata={"demo": "metadata-projection-e2e"},
        max_retries=0,
    )


async def _run_successful_provider_work(
    *, options: MetadataProjectionE2EOptions, session: Any
) -> SuccessfulProviderRun:
    failures: list[str] = []
    for idx, candidate in enumerate(
        demo_provider_candidates(
            requested_provider=options.provider,
            requested_model=options.model,
        ),
        start=1,
    ):
        ok(
            "Trying provider candidate "
            f"{idx}: {candidate.provider}/{candidate.model}"
        )
        try:
            work = _queued_work(
                provider=candidate.provider, model=candidate.model
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{candidate.provider}/{candidate.model}: {exc}")
            warn(f"Skipping provider candidate before submit: {exc}")
            continue
        await session.work_queue.submit_work(work)
        await run_streaming_worker(
            work_queue=session.work_queue,
            config=StreamingWorkerConfig(
                worker_id=f"metadata-demo-worker-{idx}",
                max_messages=1,
            ),
        )
        events = await collect_streaming_log_events(
            session.event_log,
            expected_min_events=6,
        )
        _print_event_sequence(events)
        try:
            verify_successful_work_lifecycle(
                events,
                work_id=work.work_id,
                success_message="Successful lifecycle events verified",
            )
            return SuccessfulProviderRun(
                candidate=candidate,
                work=work,
                events=events,
            )
        except RuntimeError as exc:
            failures.append(f"{candidate.provider}/{candidate.model}: {exc}")
            warn(f"Provider candidate failed success checks: {exc}")
            if options.provider is not None:
                break
    raise RuntimeError(
        "No provider candidate produced a successful lifecycle. "
        + "; ".join(failures)
    )


def _print_event_sequence(events: list[EventEnvelope]) -> None:
    ok("Streaming event sequence:")
    for event in events:
        roles = [ref.role for ref in event.payload_refs]
        print(
            f"  {event.event_type} "
            f"work_id={event.work_id or '-'} "
            f"attempt_id={event.attempt_id or '-'} "
            f"roles={roles or '-'}"
        )


def _verify_response_artifact(
    *, artifact_config: ArtifactProjectionConfig, work_id: str
) -> dict[str, str]:
    store = ArtifactStore(config=artifact_config)
    store.initialize()
    references = [
        reference
        for reference in store.index.list_references()
        if reference.source_ref.payload_role == "response_json"
        and reference.event_context.work_id == work_id
    ]
    if len(references) != 1:
        raise RuntimeError(
            f"Expected one response_json artifact for {work_id}, "
            f"found {len(references)}"
        )
    response = ArtifactReader(artifact_config).read_json(references[0])
    if not isinstance(response, dict):
        raise RuntimeError("Response artifact is not a JSON object")
    response_payload = cast("dict[str, Any]", response)
    text = response_payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Response artifact has no non-empty text field")
    return {
        "artifact_id": references[0].artifact_id,
        "text_preview": text[:120].replace("\n", " "),
    }


def _verify_projection(
    config: MetadataProjectionConfig,
    *,
    expected_events: int,
    expected_work_id: str,
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
    if not _metadata_has_successful_work(config, work_id=expected_work_id):
        raise RuntimeError(
            f"Metadata projection lacks successful work_completed for {expected_work_id}"
        )
    ok(
        "Metadata verified: "
        f"entities={summary.entity_count} "
        f"assertions={summary.assertion_count} "
        f"errors={summary.error_count}"
    )


def _metadata_has_successful_work(
    config: MetadataProjectionConfig, *, work_id: str
) -> bool:
    from sqlalchemy import select

    from dr_llm.metadata_projection.schema import metadata_assertions

    store = MetadataStore(config=config)
    try:
        with store.runtime.connect() as conn:
            row = conn.execute(
                select(metadata_assertions.c.metadata_json).where(
                    metadata_assertions.c.assertion_type == "work_completed",
                    metadata_assertions.c.status == "succeeded",
                    metadata_assertions.c.metadata_json["event"][
                        "work_id"
                    ].astext
                    == work_id,
                )
            ).first()
    finally:
        store.close()
    return row is not None


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

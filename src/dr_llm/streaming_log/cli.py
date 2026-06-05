from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from dr_llm.streaming_log.bootstrap import (
    bootstrap_streaming_log,
    inspect_streaming_log,
)
from dr_llm.streaming_log.client import (
    StreamingEventLog,
    StreamingLogConnection,
    StreamingPayloadStore,
)
from dr_llm.streaming_log.config import StreamingLogConfig
from dr_llm.streaming_log.ingest_pools import (
    PoolImportResult,
    ingest_pool,
    ingest_pools,
)
from dr_llm.streaming_log.workers import (
    StreamingWorkerConfig,
    run_streaming_worker,
)

streaming_log_app = typer.Typer(help="NATS JetStream streaming-log commands")
console = Console()


@streaming_log_app.command("bootstrap")
def bootstrap() -> None:
    """Create or validate streaming-log NATS resources."""
    status = asyncio.run(bootstrap_streaming_log())
    console.print_json(status.model_dump_json())


@streaming_log_app.command("inspect")
def inspect() -> None:
    """Print streaming-log NATS resource status."""
    status = asyncio.run(inspect_streaming_log())
    console.print_json(status.model_dump_json())


@streaming_log_app.command("ingest-pool")
def ingest_pool_command(
    dsn: str = typer.Option(..., "--dsn", help="Postgres DSN."),
    pool_name: str = typer.Option(..., "--pool-name", help="Pool to import."),
    source_id: str | None = typer.Option(
        None, "--source-id", help="Stable source database identifier."
    ),
    sample_limit: int | None = typer.Option(
        None,
        "--sample-limit",
        help="Import at most this many samples from the pool.",
    ),
) -> None:
    """Import one existing pool as reconstructed snapshot facts."""
    result = asyncio.run(
        _ingest_one_pool(
            dsn=dsn,
            pool_name=pool_name,
            source_id=source_id,
            sample_limit=sample_limit,
        )
    )
    console.print_json(result.model_dump_json())


@streaming_log_app.command("ingest-pools")
def ingest_pools_command(
    dsn: str = typer.Option(..., "--dsn", help="Postgres DSN."),
    source_id: str | None = typer.Option(
        None, "--source-id", help="Stable source database identifier."
    ),
    sample_limit: int | None = typer.Option(
        None,
        "--sample-limit",
        help="Import at most this many samples from each pool.",
    ),
) -> None:
    """Import all discovered pools as reconstructed snapshot facts."""
    results = asyncio.run(
        _ingest_all_pools(
            dsn=dsn,
            source_id=source_id,
            sample_limit=sample_limit,
        )
    )
    console.print_json(
        data={"pools": [result.model_dump(mode="json") for result in results]}
    )


@streaming_log_app.command("run-worker")
def run_worker(
    worker_id: str | None = typer.Option(
        None, "--worker-id", help="Producer/worker identifier."
    ),
    max_messages: int | None = typer.Option(
        None,
        "--max-messages",
        help="Stop after processing this many work messages.",
    ),
) -> None:
    """Run an async JetStream-backed LLM worker."""
    if worker_id is None:
        config = StreamingWorkerConfig(max_messages=max_messages)
    else:
        config = StreamingWorkerConfig(
            worker_id=worker_id, max_messages=max_messages
        )
    asyncio.run(run_streaming_worker(config=config))


async def _ingest_one_pool(
    *,
    dsn: str,
    pool_name: str,
    source_id: str | None,
    sample_limit: int | None,
) -> PoolImportResult:
    async with StreamingLogConnection(StreamingLogConfig()) as connection:
        event_log = _event_log(connection)
        return await ingest_pool(
            event_log=event_log,
            dsn=dsn,
            pool_name=pool_name,
            source_id=source_id,
            sample_limit=sample_limit,
        )


async def _ingest_all_pools(
    *, dsn: str, source_id: str | None, sample_limit: int | None
) -> list[PoolImportResult]:
    async with StreamingLogConnection(StreamingLogConfig()) as connection:
        event_log = _event_log(connection)
        return await ingest_pools(
            event_log=event_log,
            dsn=dsn,
            source_id=source_id,
            sample_limit=sample_limit,
        )


def _event_log(connection: StreamingLogConnection) -> StreamingEventLog:
    payload_store = StreamingPayloadStore(connection)
    return StreamingEventLog(connection, payload_store)


__all__ = ["streaming_log_app"]

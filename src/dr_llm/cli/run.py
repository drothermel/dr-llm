from __future__ import annotations

import typer

from dr_llm.storage.models import RunStatus

from . import common

run_app = typer.Typer(help="Run lifecycle commands")


@run_app.command("start")
def run_start(
    run_type: str = typer.Option("generic"),
    status: RunStatus = typer.Option(RunStatus.running),
    run_id: str | None = typer.Option(None),
    metadata_json: str | None = typer.Option(None),
    parameters_json: str | None = typer.Option(
        None, help="Optional JSON object of run parameters."
    ),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Start or upsert a run record."""
    metadata = (
        common._parse_json(metadata_json, arg_name="metadata_json", expected=dict) or {}
    )
    parameters = (
        common._parse_json(parameters_json, arg_name="parameters_json", expected=dict)
        or {}
    )

    repository = common._repo(dsn, min_pool_size, max_pool_size)
    try:
        persisted_run_id = repository.start_run(
            run_type=run_type,
            status=status,
            metadata=metadata,
            run_id=run_id,
        )
        written = repository.upsert_run_parameters(
            run_id=persisted_run_id, parameters=parameters
        )
        common._emit({"run_id": persisted_run_id, "parameters_written": written})
    finally:
        repository.close()


@run_app.command("finish")
def run_finish(
    run_id: str = typer.Option(...),
    status: RunStatus = typer.Option(...),
    metadata_json: str | None = typer.Option(None),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Finish a run record."""
    metadata = common._parse_json(
        metadata_json, arg_name="metadata_json", expected=dict
    )
    repository = common._repo(dsn, min_pool_size, max_pool_size)
    try:
        repository.finish_run(run_id=run_id, status=status, metadata=metadata)
        common._emit({"run_id": run_id, "status": status.value})
    finally:
        repository.close()


@run_app.command("list-calls")
def run_list_calls(
    run_id: str | None = typer.Option(None, help="Filter by run ID."),
    limit: int = typer.Option(100),
    offset: int = typer.Option(0),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """List recorded LLM calls, optionally filtered by run."""
    repository = common._repo(dsn, min_pool_size, max_pool_size)
    try:
        calls = repository.list_calls(run_id=run_id, limit=limit, offset=offset)
        common._emit(
            {
                "calls": [
                    call.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                    for call in calls
                ],
                "count": len(calls),
            }
        )
    finally:
        repository.close()

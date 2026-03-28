from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from dr_llm.catalog.models import ModelCatalogSyncResult
from dr_llm.client import LlmClient
from dr_llm.project.cli import project_app
from dr_llm.project.docker import get_project
from dr_llm.providers import (
    ProviderAvailabilityStatus,
    build_default_registry,
    supported_provider_statuses,
)
from dr_llm.storage import PostgresRepository, StorageConfig
from dr_llm.types import (
    LlmRequest,
    Message,
    ModelCatalogEntry,
    ModelCatalogQuery,
    ReasoningConfig,
    RunStatus,
)

app = typer.Typer()
run_app = typer.Typer(help="Run lifecycle commands")
models_app = typer.Typer(help="Model catalog commands")

app.add_typer(run_app, name="run")
app.add_typer(models_app, name="models")
app.add_typer(project_app, name="project")


@app.callback()
def main(
    project: str | None = typer.Option(None, help="Use a named project's database."),
) -> None:
    """dr-llm CLI"""
    if project is not None:
        info = get_project(project)
        if info is None:
            typer.secho(f"Project '{project}' not found", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        if info.status != "running":
            typer.secho(
                f"Project '{project}' is {info.status} - start it first",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        os.environ["DR_LLM_DATABASE_URL"] = info.dsn


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _provider_requirements_text(status: ProviderAvailabilityStatus) -> str:
    if status.available:
        return "ready"
    parts = [f"env: {name}" for name in status.missing_env_vars]
    parts.extend(f"exe: {name}" for name in status.missing_executables)
    return ", ".join(parts)


def _render_providers_table(statuses: list[ProviderAvailabilityStatus]) -> None:
    table = Table(title="Providers")
    table.add_column("Provider", style="bold")
    table.add_column("Available")
    table.add_column("Structured")
    table.add_column("Missing Requirements", overflow="fold")

    for status in statuses:
        available_text = "[green]yes[/green]" if status.available else "[red]no[/red]"
        structured_text = (
            "[green]yes[/green]"
            if status.supports_structured_output
            else "[red]no[/red]"
        )
        table.add_row(
            status.provider,
            available_text,
            structured_text,
            _provider_requirements_text(status),
        )

    available_count = sum(1 for status in statuses if status.available)
    console = Console()
    console.print(table)
    console.print(
        f"Available: {available_count}/{len(statuses)} supported providers are ready."
    )


def _sync_failure_summary(result: ModelCatalogSyncResult) -> str:
    error = result.error.splitlines()[0] if result.error else "unknown error"
    return f"{result.provider} ({error})"


def _sync_failure_message(result: ModelCatalogSyncResult) -> str:
    return result.error.splitlines()[0] if result.error else "unknown error"


def _render_models_sync_summary(results: list[ModelCatalogSyncResult]) -> None:
    failures = [result for result in results if not result.success]
    successes = [result for result in results if result.success]

    if failures:
        if len(failures) == 1 and not successes:
            typer.secho(
                f"Model sync failed for {failures[0].provider}: {_sync_failure_message(failures[0])}",
                fg=typer.colors.RED,
                err=True,
            )
        else:
            failure_text = ", ".join(
                _sync_failure_summary(result) for result in failures
            )
            typer.secho(
                f"Model sync failed for {len(failures)}/{len(results)} providers: {failure_text}",
                fg=typer.colors.RED,
                err=True,
            )
        raise typer.Exit(code=1)

    total_entries = sum(result.entry_count for result in successes)
    if len(successes) == 1:
        result = successes[0]
        typer.echo(f"Synced {result.entry_count} models for {result.provider}.")
        return
    typer.echo(f"Synced {total_entries} models across {len(successes)} providers.")


def _models_list_header(items: list[ModelCatalogEntry], provider: str | None) -> str:
    count = len(items)
    if provider is not None:
        return f"{provider} Models (Showing {count} out of {{total_count}})"
    providers = {item.provider for item in items}
    if len(providers) <= 1:
        return f"Models (Showing {count} out of {{total_count}})"
    return f"Models (Showing {count} out of {{total_count}} across {len(providers)} providers)"


def _render_models_list(
    items: list[ModelCatalogEntry], provider: str | None, total_count: int
) -> None:
    if not items:
        if total_count > 0:
            if provider is not None:
                typer.echo(
                    f"No models found on this page for {provider}. {total_count} matching models exist."
                )
                return
            typer.echo(
                f"No models found on this page. {total_count} matching models exist."
            )
            return
        if provider is not None:
            typer.echo(f"No models found for {provider}.")
            return
        typer.echo("No models found.")
        return

    typer.echo(_models_list_header(items, provider).format(total_count=total_count))
    include_provider = provider is None and len({item.provider for item in items}) > 1
    for item in items:
        label = f"{item.provider}: {item.model}" if include_provider else item.model
        if item.display_name and item.display_name != item.model:
            label = f"{label} ({item.display_name})"
        typer.echo(f"- {label}")


def _parse_json(
    value: str | None, *, arg_name: str, expected: type | tuple[type, ...] | None = None
) -> Any:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON for {arg_name}: {exc}") from exc
    if expected is not None and not isinstance(parsed, expected):
        expected_names = (
            ", ".join(t.__name__ for t in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise typer.BadParameter(f"{arg_name} must decode to {expected_names}")
    return parsed


def _load_messages(
    messages_file: Path | None,
    messages: list[str],
    *,
    require_nonempty: bool = True,
) -> list[Message]:
    result: list[Message] = []
    if messages_file is not None:
        try:
            payload = json.loads(messages_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"messages-file is not valid JSON: {exc}") from exc
        if isinstance(payload, dict):
            payload = payload.get("messages")
        if not isinstance(payload, list):
            raise typer.BadParameter(
                "messages-file must be a JSON list or an object with a 'messages' list"
            )
        for item in payload:
            if not isinstance(item, dict):
                raise typer.BadParameter("messages-file entries must be JSON objects")
            try:
                result.append(Message(**item))
            except ValidationError as exc:
                raise typer.BadParameter(
                    f"Invalid message in messages-file: {exc}"
                ) from exc

    result.extend(Message(role="user", content=content) for content in messages)
    if require_nonempty and not result:
        raise typer.BadParameter(
            "At least one message is required (use --message or --messages-file)"
        )
    return result


def _repo(
    dsn: str | None, min_pool_size: int, max_pool_size: int
) -> PostgresRepository:
    if dsn is None:
        cfg = StorageConfig(min_pool_size=min_pool_size, max_pool_size=max_pool_size)
    else:
        cfg = StorageConfig(
            dsn=dsn,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
        )
    return PostgresRepository(cfg)


@app.command("providers")
def providers(
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Emit JSON output.",
    ),
) -> None:
    """List supported providers and whether they are available locally."""
    registry = build_default_registry()
    statuses = supported_provider_statuses(registry)
    if json_output:
        _emit(
            {
                "providers": [
                    status.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_computed_fields=True,
                    )
                    for status in statuses
                ]
            }
        )
        return
    _render_providers_table(statuses)


@models_app.command("sync")
def models_sync(
    provider: str | None = typer.Option(None, help="Optional provider key."),
    verbose: bool = typer.Option(
        False,
        "--verbose/--no-verbose",
        help="Emit full JSON sync results instead of the concise summary.",
    ),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Sync provider model catalog into PostgreSQL."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        client = LlmClient(registry=build_default_registry(), repository=repository)
        results = client.sync_models_detailed(provider=provider)
        exit_code = 1 if any(not result.success for result in results) else 0
        if verbose:
            _emit(
                {
                    "results": [
                        result.model_dump(
                            mode="json",
                            exclude_none=True,
                            exclude_computed_fields=True,
                        )
                        for result in results
                    ]
                }
            )
            raise typer.Exit(exit_code)
        _render_models_sync_summary(results)
    finally:
        repository.close()


@models_app.command("list")
def models_list(
    provider: str | None = typer.Option(None, help="Optional provider filter."),
    supports_reasoning: bool | None = typer.Option(
        None, help="Optional reasoning support filter."
    ),
    model_contains: str | None = typer.Option(None, help="Substring model filter."),
    limit: int = typer.Option(20),
    offset: int = typer.Option(0),
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Emit JSON output.",
    ),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """List models from stored catalog."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        registry = build_default_registry()
        if provider is not None:
            try:
                provider = registry.get(provider).name
            except KeyError:
                pass
        client = LlmClient(registry=registry, repository=repository)
        items = client.list_models(
            ModelCatalogQuery(
                provider=provider,
                supports_reasoning=supports_reasoning,
                model_contains=model_contains,
                limit=limit,
                offset=offset,
            )
        )
        if json_output:
            _emit(
                {
                    "models": [
                        item.model_dump(
                            mode="json",
                            exclude_none=True,
                            exclude_computed_fields=True,
                        )
                        for item in items
                    ]
                }
            )
        else:
            total_count = client.count_models(
                ModelCatalogQuery(
                    provider=provider,
                    supports_reasoning=supports_reasoning,
                    model_contains=model_contains,
                )
            )
            _render_models_list(items, provider, total_count)
            static_providers: set[str] = set()
            for item in items:
                if item.source_quality == "static":
                    static_providers.add(item.provider)
            for sp in sorted(static_providers):
                docs_url = ""
                for item in items:
                    if item.provider == sp and item.metadata.get("docs_url"):
                        docs_url = item.metadata["docs_url"]
                        break
                msg = f"\nNote: {sp} models are from a static list and may be out of date."
                if docs_url:
                    msg += f"\nSee {docs_url} for the latest models."
                typer.echo(msg, err=True)
    finally:
        repository.close()


@models_app.command("show")
def models_show(
    provider: str = typer.Option(...),
    model: str = typer.Option(...),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Show one model from stored catalog."""
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        client = LlmClient(registry=build_default_registry(), repository=repository)
        item = client.show_model(provider=provider, model=model)
        if item is None:
            typer.secho(
                f"Model not found for provider={provider!r} model={model!r}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        _emit(
            item.model_dump(
                mode="json", exclude_none=True, exclude_computed_fields=True
            )
        )
    finally:
        repository.close()


@app.command("query")
def query(
    provider: str = typer.Option(..., help="Provider key registered in dr-llm."),
    model: str = typer.Option(..., help="Model identifier for the provider."),
    message: list[str] = typer.Option(
        None, "--message", help="User message. Repeatable."
    ),
    messages_file: Path | None = typer.Option(
        None, help="Path to JSON messages payload."
    ),
    temperature: float | None = typer.Option(None),
    top_p: float | None = typer.Option(None),
    max_tokens: int | None = typer.Option(None),
    reasoning_json: str | None = typer.Option(
        None,
        help='JSON reasoning config (e.g. {"effort":"high"} or {"max_tokens":2000}).',
    ),
    metadata_json: str | None = typer.Option(None, help="JSON object metadata."),
    run_id: str | None = typer.Option(None),
    external_call_id: str | None = typer.Option(None),
    record: bool = typer.Option(True, "--record/--no-record"),
    dsn: str | None = typer.Option(None, envvar="DR_LLM_DATABASE_URL"),
    min_pool_size: int = typer.Option(4),
    max_pool_size: int = typer.Option(64),
) -> None:
    """Execute a single LLM query through the unified provider interface."""
    metadata = _parse_json(metadata_json, arg_name="metadata_json", expected=dict) or {}
    reasoning_payload = _parse_json(
        reasoning_json, arg_name="reasoning_json", expected=dict
    )
    try:
        reasoning = (
            ReasoningConfig(**reasoning_payload)
            if isinstance(reasoning_payload, dict)
            else None
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    messages_payload = _load_messages(messages_file, message or [])

    repository: PostgresRepository | None = None
    try:
        if record:
            repository = _repo(dsn, min_pool_size, max_pool_size)
        client = LlmClient(registry=build_default_registry(), repository=repository)
        try:
            request = LlmRequest(
                provider=provider,
                model=model,
                messages=messages_payload,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
                metadata=metadata,
            )
        except ValidationError as exc:
            raise typer.BadParameter(str(exc)) from exc
        response = client.query(
            request,
            run_id=run_id,
            external_call_id=external_call_id,
            metadata=metadata,
        )
        _emit(response.model_dump(mode="json", exclude_computed_fields=True))
    finally:
        if repository is not None:
            repository.close()


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
    metadata = _parse_json(metadata_json, arg_name="metadata_json", expected=dict) or {}
    parameters = (
        _parse_json(parameters_json, arg_name="parameters_json", expected=dict) or {}
    )

    repository = _repo(dsn, min_pool_size, max_pool_size)
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
        _emit({"run_id": persisted_run_id, "parameters_written": written})
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
    metadata = _parse_json(metadata_json, arg_name="metadata_json", expected=dict)
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        repository.finish_run(run_id=run_id, status=status, metadata=metadata)
        _emit({"run_id": run_id, "status": status.value})
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
    repository = _repo(dsn, min_pool_size, max_pool_size)
    try:
        calls = repository.list_calls(run_id=run_id, limit=limit, offset=offset)
        _emit(
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


if __name__ == "__main__":
    app()

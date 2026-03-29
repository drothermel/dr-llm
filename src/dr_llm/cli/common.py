from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from dr_llm.catalog.models import ModelCatalogEntry, ModelCatalogSyncResult
from dr_llm.pool.db import PoolDb
from dr_llm.pool.runtime import DbConfig
from dr_llm.providers.models import Message
from dr_llm.providers.provider_config import ProviderAvailabilityStatus


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


def _repo(dsn: str | None, min_pool_size: int, max_pool_size: int) -> PoolDb:
    kwargs: dict[str, object] = {
        "min_pool_size": min_pool_size,
        "max_pool_size": max_pool_size,
    }
    if dsn is not None:
        kwargs["dsn"] = dsn
    return PoolDb(DbConfig(**kwargs))

from __future__ import annotations

import typer

from dr_llm.catalog.models import ModelCatalogQuery
from dr_llm.client import LlmClient
from dr_llm.providers import build_default_registry

from . import common

models_app = typer.Typer(help="Model catalog commands")


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
    repository = common._repo(dsn, min_pool_size, max_pool_size)
    try:
        client = LlmClient(registry=build_default_registry(), repository=repository)
        results = client.sync_models_detailed(provider=provider)
        exit_code = 1 if any(not result.success for result in results) else 0
        if verbose:
            common._emit(
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
        common._render_models_sync_summary(results)
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
    repository = common._repo(dsn, min_pool_size, max_pool_size)
    try:
        registry = build_default_registry()
        if provider is not None:
            try:
                provider = registry.get(provider).name
            except KeyError:
                pass
        client = LlmClient(registry=registry, repository=repository)
        base_query = ModelCatalogQuery(
            provider=provider,
            supports_reasoning=supports_reasoning,
            model_contains=model_contains,
        )
        list_query = ModelCatalogQuery(
            provider=base_query.provider,
            supports_reasoning=base_query.supports_reasoning,
            model_contains=base_query.model_contains,
            limit=limit,
            offset=offset,
        )
        items = client.list_models(list_query)
        if json_output:
            common._emit(
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
            total_count = client.count_models(base_query)
            common._render_models_list(items, provider, total_count)
            static_providers = {
                item.provider for item in items if item.source_quality == "static"
            }
            for provider_name in sorted(static_providers):
                docs_url = ""
                for item in items:
                    if (
                        item.provider == provider_name
                        and item.metadata.get("docs_url") is not None
                    ):
                        docs_url = item.metadata["docs_url"]
                        break
                message = f"\nNote: {provider_name} models are from a static list and may be out of date."
                if docs_url:
                    message += f"\nSee {docs_url} for the latest models."
                typer.echo(message, err=True)
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
    repository = common._repo(dsn, min_pool_size, max_pool_size)
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
        common._emit(
            item.model_dump(
                mode="json", exclude_none=True, exclude_computed_fields=True
            )
        )
    finally:
        repository.close()

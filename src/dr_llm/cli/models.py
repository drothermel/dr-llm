from __future__ import annotations

from contextlib import suppress

import typer

from dr_llm.catalog.file_store import FileCatalogStore
from dr_llm.catalog.models import ModelCatalogQuery
from dr_llm.catalog.service import ModelCatalogService
from dr_llm.providers import build_default_registry
from dr_llm.providers.registry import ProviderRegistry

from . import common

models_app = typer.Typer(help="Model catalog commands")


def _catalog_service(
    registry: ProviderRegistry | None = None,
) -> tuple[ModelCatalogService, ProviderRegistry]:
    reg = registry or build_default_registry()
    svc = ModelCatalogService(registry=reg, repository=FileCatalogStore())
    return svc, reg


@models_app.command("sync")
def models_sync(
    provider: str | None = typer.Option(None, help="Optional provider key."),
    verbose: bool = typer.Option(
        False,
        "--verbose/--no-verbose",
        help="Emit full JSON sync results instead of the concise summary.",
    ),
) -> None:
    """Sync provider model catalog."""
    svc, _ = _catalog_service()
    results = svc.sync_models_detailed(provider=provider)
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
) -> None:
    """List models from stored catalog."""
    svc, registry = _catalog_service()
    if provider is not None:
        with suppress(KeyError):
            provider = registry.get(provider).name
    base_query = ModelCatalogQuery(
        provider=provider,
        supports_reasoning=supports_reasoning,
        model_contains=model_contains,
    )
    list_query = base_query.model_copy(update={"limit": limit, "offset": offset})
    items = svc.list_models(list_query)
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
        total_count = svc.count_models(base_query)
        common._render_models_list(items, provider, total_count)
        static_docs: dict[str, str] = {}
        for item in items:
            if item.source_quality == "static" and item.provider not in static_docs:
                static_docs[item.provider] = item.metadata.get("docs_url", "")
        for provider_name in sorted(static_docs):
            message = f"\nNote: {provider_name} models are from a static list and may be out of date."
            if static_docs[provider_name]:
                message += f"\nSee {static_docs[provider_name]} for the latest models."
            typer.echo(message, err=True)


@models_app.command("show")
def models_show(
    provider: str = typer.Option(...),
    model: str = typer.Option(...),
) -> None:
    """Show one model from stored catalog."""
    svc, _ = _catalog_service()
    item = svc.show_model(provider=provider, model=model)
    if item is None:
        typer.secho(
            f"Model not found for provider={provider!r} model={model!r}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    common._emit(
        item.model_dump(mode="json", exclude_none=True, exclude_computed_fields=True)
    )

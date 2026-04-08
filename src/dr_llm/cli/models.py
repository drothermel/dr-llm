from __future__ import annotations

import asyncio
from contextlib import suppress

import typer

from dr_llm.llm.catalog.file_store import FileCatalogStore
from dr_llm.llm.catalog.model_blacklist import blacklisted_models
from dr_llm.llm.catalog.models import ModelCatalogQuery, ModelCatalogSyncResult
from dr_llm.llm.catalog.service import ModelCatalogService
from dr_llm.llm.providers.registry import ProviderRegistry
from dr_llm.llm.providers.registry import build_default_registry

from . import common

models_app = typer.Typer(help="Model catalog commands")


def _catalog_service(
    registry: ProviderRegistry | None = None,
) -> tuple[ModelCatalogService, ProviderRegistry]:
    reg = registry or build_default_registry()
    svc = ModelCatalogService(registry=reg, repository=FileCatalogStore())
    return svc, reg


def _canonical_provider_name(
    *,
    provider: str | None,
    registry: ProviderRegistry,
) -> str | None:
    if provider is None:
        return None
    with suppress(KeyError):
        return registry.get(provider).name
    return provider


def _sync_models(
    *,
    svc: ModelCatalogService,
    provider: str | None,
) -> list[ModelCatalogSyncResult]:
    return asyncio.run(svc.sync_models_detailed(provider=provider))


def _emit_models_list(
    *,
    svc: ModelCatalogService,
    registry: ProviderRegistry,
    provider: str | None,
    supports_reasoning: bool | None,
    model_contains: str | None,
    limit: int,
    offset: int,
    json_output: bool,
) -> None:
    provider = _canonical_provider_name(provider=provider, registry=registry)
    base_query = ModelCatalogQuery(
        provider=provider,
        supports_reasoning=supports_reasoning,
        model_contains=model_contains,
    )
    list_query = base_query.model_copy(update={"limit": limit, "offset": offset})
    items = svc.list_models(list_query)
    grouped_blacklist = blacklisted_models(provider=provider)
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
                ],
                "blacklist": {
                    provider_name: [
                        item.model_dump(
                            mode="json",
                            exclude={"provider"},
                            exclude_computed_fields=True,
                        )
                        for item in blacklisted_items
                    ]
                    for provider_name, blacklisted_items in grouped_blacklist.items()
                },
            }
        )
        return

    total_count = svc.count_models(base_query)
    common._render_models_list(items, provider, total_count)
    common._render_blacklist(grouped_blacklist, provider)
    static_docs: dict[str, str] = {}
    for item in items:
        if item.source_quality == "static" and item.provider not in static_docs:
            static_docs[item.provider] = item.metadata.get("docs_url", "")
    for provider_name in sorted(static_docs):
        message = f"\nNote: {provider_name} models are from a static list and may be out of date."
        if static_docs[provider_name]:
            message += f"\nSee {static_docs[provider_name]} for the latest models."
        typer.echo(message, err=True)


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
    results = _sync_models(svc=svc, provider=provider)
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
    _emit_models_list(
        svc=svc,
        registry=registry,
        provider=provider,
        supports_reasoning=supports_reasoning,
        model_contains=model_contains,
        limit=limit,
        offset=offset,
        json_output=json_output,
    )


@models_app.command("sync-list")
def models_sync_list(
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
    """Sync models first, then list them."""
    svc, registry = _catalog_service()
    results = _sync_models(svc=svc, provider=provider)
    if any(not result.success for result in results):
        common._render_models_sync_summary(results)
        return
    _emit_models_list(
        svc=svc,
        registry=registry,
        provider=provider,
        supports_reasoning=supports_reasoning,
        model_contains=model_contains,
        limit=limit,
        offset=offset,
        json_output=json_output,
    )


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

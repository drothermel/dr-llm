"""FastAPI backend for dr-llm UI tools."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.catalog.service import ModelCatalogService
from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.llm import ControlMode, build_default_registry
from dr_llm.llm.providers.core.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ProviderStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    available: bool
    missing_env_vars: list[str] = Field(default_factory=list)
    missing_executables: list[str] = Field(default_factory=list)
    supports_structured_output: bool = False


class ModelEntryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    control_mode: ControlMode | None = None
    supports_vision: bool | None = None
    source_quality: str = "live"


class ProviderModelsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    models: list[ModelEntryResponse]
    source: str  # "live" | "static" | "error"
    error: str | None = None


class SyncResultResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    success: bool
    model_count: int
    models: list[ModelEntryResponse]
    source: str
    error: str | None = None


def _entry_to_response(entry: ModelCatalogEntry) -> ModelEntryResponse:
    return ModelEntryResponse(
        provider=entry.provider,
        model=entry.model,
        display_name=entry.display_name,
        context_window=entry.context_window,
        max_output_tokens=entry.max_output_tokens,
        control_mode=entry.control_mode,
        supports_vision=entry.supports_vision,
        source_quality=entry.source_quality,
    )


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


def _get_registry(app: FastAPI) -> ProviderRegistry:
    registry = cast(ProviderRegistry | None, getattr(app.state, "registry", None))
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="service unavailable: registry not initialized",
        )
    return registry


def _fallback_model_responses(
    provider: str, service: ModelCatalogService
) -> list[ModelEntryResponse]:
    entries, _raw = service.fallback_provider_models(provider)
    return [_entry_to_response(entry) for entry in entries]


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = build_default_registry()
    app.state.registry = registry
    try:
        yield
    finally:
        registry.close()
        if getattr(app.state, "registry", None) is registry:
            app.state.registry = None


app = FastAPI(title="dr-llm UI API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/providers", response_model=list[ProviderStatusResponse])
def list_providers(request: Request) -> list[ProviderStatusResponse]:
    """List all supported providers with availability status."""
    registry = _get_registry(request.app)
    statuses = registry.availability_statuses()
    return [
        ProviderStatusResponse(
            provider=s.provider,
            available=s.available,
            missing_env_vars=list(s.missing_env_vars),
            missing_executables=list(s.missing_executables),
            supports_structured_output=s.supports_structured_output,
        )
        for s in statuses
    ]


@app.get("/api/providers/{provider}/models", response_model=ProviderModelsResponse)
def get_provider_models(provider: str, request: Request) -> ProviderModelsResponse:
    """Get models for a provider.  Uses static data if the provider is unavailable."""
    registry = _get_registry(request.app)
    try:
        orchestrator = registry.get(provider)
    except KeyError as err:
        raise HTTPException(
            status_code=404, detail=f"Unknown provider: {provider}"
        ) from err

    service = ModelCatalogService(registry=registry)
    is_available = orchestrator.availability_status().available

    if not is_available:
        static = _fallback_model_responses(provider, service)
        return ProviderModelsResponse(
            provider=provider,
            models=static,
            source="static",
        )

    try:
        entries, _raw = service.fetch_provider_models(provider)
    except (ProviderTransportError, ProviderSemanticError) as exc:
        static = _fallback_model_responses(provider, service)
        return ProviderModelsResponse(
            provider=provider,
            models=static,
            source="static",
            error=f"{type(exc).__name__}: {exc}",
        )
    models = [_entry_to_response(e) for e in entries]
    return ProviderModelsResponse(
        provider=provider,
        models=models,
        source="live",
    )


@app.post("/api/providers/{provider}/sync", response_model=SyncResultResponse)
def sync_provider_models(provider: str, request: Request) -> SyncResultResponse:
    """Trigger a live model sync for a provider."""
    registry = _get_registry(request.app)
    try:
        registry.get(provider)
    except KeyError as err:
        raise HTTPException(
            status_code=404, detail=f"Unknown provider: {provider}"
        ) from err

    service = ModelCatalogService(registry=registry)
    try:
        entries, _raw = service.fetch_provider_models(provider)
    except (ProviderTransportError, ProviderSemanticError) as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        static = _fallback_model_responses(provider, service)
        return SyncResultResponse(
            provider=provider,
            success=False,
            model_count=len(static),
            models=static,
            source="static",
            error=error_msg,
        )
    models = [_entry_to_response(e) for e in entries]
    return SyncResultResponse(
        provider=provider,
        success=True,
        model_count=len(models),
        models=models,
        source="live",
    )

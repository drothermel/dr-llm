"""FastAPI backend for dr-llm UI tools."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.catalog.fetchers import fetch_models_for_adapter
from dr_llm.catalog.fetchers.static import (
    CLAUDE_CODE_MODELS,
    CODEX_MODELS,
    KIMI_CODING_MODELS,
    MINIMAX_TEXT_MODELS,
)
from dr_llm.catalog.models import ModelCatalogEntry
from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.providers import (
    build_default_registry,
    supported_provider_statuses,
)
from dr_llm.providers.registry import ProviderRegistry

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
    supports_reasoning: bool | None = None
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


# ---------------------------------------------------------------------------
# Static model fallback data (for providers without API keys)
# ---------------------------------------------------------------------------

# Well-known models for API providers (shown when keys are missing)
_OPENAI_COMMON_MODELS = [
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.4-mini", "GPT-5.4 Mini"),
    ("gpt-5.3", "GPT-5.3"),
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5", "GPT-5"),
    ("o3", "o3"),
    ("o3-mini", "o3-mini"),
    ("o4-mini", "o4-mini"),
]

_ANTHROPIC_COMMON_MODELS = [
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]

_GOOGLE_COMMON_MODELS = [
    ("gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro"),
    ("gemini-2.5-flash-preview-04-17", "Gemini 2.5 Flash"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash"),
    ("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite"),
]

_OPENROUTER_COMMON_MODELS = [
    ("openai/o3-mini", "OpenAI o3-mini"),
    ("openai/gpt-4.1", "OpenAI GPT-4.1"),
    ("anthropic/claude-3.7-sonnet", "Anthropic Claude 3.7 Sonnet"),
]

_GLM_COMMON_MODELS = [
    ("glm-4.5", "GLM 4.5"),
    ("glm-4-air", "GLM 4 Air"),
    ("glm-4-flash", "GLM 4 Flash"),
]

STATIC_MODELS: dict[str, list[tuple[str, str]]] = {
    "codex": CODEX_MODELS,
    "claude-code": CLAUDE_CODE_MODELS,
    "claude-code-minimax": MINIMAX_TEXT_MODELS,
    "claude-code-kimi": KIMI_CODING_MODELS,
    "minimax": MINIMAX_TEXT_MODELS,
    "openrouter": _OPENROUTER_COMMON_MODELS,
    "openai": _OPENAI_COMMON_MODELS,
    "anthropic": _ANTHROPIC_COMMON_MODELS,
    "google": _GOOGLE_COMMON_MODELS,
    "glm": _GLM_COMMON_MODELS,
}


def _static_models_for_provider(provider: str) -> list[ModelEntryResponse]:
    """Return hardcoded model entries when live fetching isn't possible."""
    models = STATIC_MODELS.get(provider, [])
    return [
        ModelEntryResponse(
            provider=provider,
            model=model_id,
            display_name=display_name,
            source_quality="static",
        )
        for model_id, display_name in models
    ]


def _entry_to_response(entry: ModelCatalogEntry) -> ModelEntryResponse:
    return ModelEntryResponse(
        provider=entry.provider,
        model=entry.model,
        display_name=entry.display_name,
        context_window=entry.context_window,
        max_output_tokens=entry.max_output_tokens,
        supports_reasoning=entry.supports_reasoning,
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
    statuses = supported_provider_statuses(registry)
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
        adapter = registry.get(provider)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    # Check if provider is available (has required env vars / executables)
    statuses = supported_provider_statuses(registry)
    status = next((s for s in statuses if s.provider == provider), None)
    is_available = status.available if status else False

    if not is_available:
        # Return static/hardcoded models
        static = _static_models_for_provider(provider)
        return ProviderModelsResponse(
            provider=provider,
            models=static,
            source="static",
        )

    # Try live fetch
    try:
        entries, _raw = fetch_models_for_adapter(adapter)
    except (ProviderTransportError, ProviderSemanticError) as exc:
        # Fall back to static
        static = _static_models_for_provider(provider)
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
        adapter = registry.get(provider)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    try:
        entries, _raw = fetch_models_for_adapter(adapter)
    except (ProviderTransportError, ProviderSemanticError) as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        static = _static_models_for_provider(provider)
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

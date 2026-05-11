"""Shared model selections for demo scripts."""

from __future__ import annotations

from dr_llm.llm import (
    GoogleBudgetConfig,
    LlmConfig,
    OpenAIGpt5Config,
    ProviderName,
    build_default_registry,
)
from dr_llm.llm.catalog.service import ModelCatalogService

_THINKING_SWEEP_OPENAI_MODELS = [
    "gpt-5.4-mini-2026-03-17",
    "gpt-5-mini-2025-08-07",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-nano-2025-04-14",
]
_THINKING_SWEEP_GOOGLE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "gemma-3n-e4b-it",
    "gemma-3n-e2b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]


def _fallback_model_ids(
    provider: ProviderName, preferred: tuple[str, ...] = ()
) -> list[str]:
    registry = build_default_registry()
    try:
        service = ModelCatalogService(registry=registry)
        entries, _raw = service.fallback_provider_models(provider)
        model_ids = [entry.model for entry in entries]
    finally:
        registry.close()
    if not preferred:
        return model_ids
    selected = [model for model in preferred if model in model_ids]
    return selected or model_ids[:1]


DEMO_THINKING_SWEEP_MODELS: dict[ProviderName, list[str]] = {
    ProviderName.CLAUDE_CODE: _fallback_model_ids(ProviderName.CLAUDE_CODE),
    ProviderName.MINIMAX: _fallback_model_ids(ProviderName.MINIMAX),
    ProviderName.KIMI_CODE: _fallback_model_ids(ProviderName.KIMI_CODE),
    ProviderName.OPENROUTER: _fallback_model_ids(ProviderName.OPENROUTER),
    ProviderName.OPENAI: _THINKING_SWEEP_OPENAI_MODELS,
    ProviderName.CODEX: _fallback_model_ids(
        ProviderName.CODEX, preferred=("gpt-5.1-codex-mini",)
    ),
    ProviderName.GOOGLE: _THINKING_SWEEP_GOOGLE_MODELS,
}

DEMO_QUERY_DEFAULT_MODELS: dict[ProviderName, str] = {
    ProviderName.OPENAI: "gpt-4o-mini",
    ProviderName.ANTHROPIC: "claude-sonnet-4-20250514",
    ProviderName.GOOGLE: "gemini-2.5-flash",
    ProviderName.GLM: "glm-4.5",
    ProviderName.MINIMAX: "MiniMax-M2",
    ProviderName.CLAUDE_CODE: "claude-sonnet-4-6",
    ProviderName.CODEX: "gpt-5.4-mini",
    ProviderName.KIMI_CODE: "kimi-for-coding",
}


def demo_pool_fill_llm_configs() -> dict[str, LlmConfig]:
    """Build fresh LLM configs for the pool-fill demo."""
    registry = build_default_registry()
    try:
        return {
            "gpt-5-mini-default": OpenAIGpt5Config(
                model="gpt-5-mini",
                max_tokens=64,
            ).to_llm_config(registry),
            "gemini-flash-default": GoogleBudgetConfig(
                model="gemini-2.5-flash",
                max_tokens=64,
            ).to_llm_config(registry),
        }
    finally:
        registry.close()


__all__ = [
    "DEMO_QUERY_DEFAULT_MODELS",
    "DEMO_THINKING_SWEEP_MODELS",
    "demo_pool_fill_llm_configs",
]

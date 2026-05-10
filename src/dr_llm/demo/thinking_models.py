"""Shared model lineups for the demo_thinking_and_effort sweep."""

from __future__ import annotations

from dr_llm.llm import (
    CLAUDE_CODE_MODELS,
    KIMI_CODING_MODELS,
    MINIMAX_TEXT_MODELS,
    ProviderName,
    openrouter_allowed_models,
)

DEMO_OPENAI_MODELS = [
    "gpt-5.4-mini-2026-03-17",
    "gpt-5-mini-2025-08-07",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-nano-2025-04-14",
]
DEMO_CODEX_MODELS = [
    "gpt-5.1-codex-mini",
]
DEMO_CLAUDE_MODELS = [
    model_id for model_id, _display_name in CLAUDE_CODE_MODELS
]
DEMO_KIMI_CODE_MODELS = [
    model_id for model_id, _display_name in KIMI_CODING_MODELS
]
DEMO_MINIMAX_MODELS = [
    model_id for model_id, _display_name in MINIMAX_TEXT_MODELS
]
DEMO_GOOGLE_MODELS = [
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
DEMO_OPENROUTER_MODELS = list(openrouter_allowed_models())
DEMO_PROVIDER_MODELS: dict[ProviderName, list[str]] = {
    ProviderName.CLAUDE_CODE: DEMO_CLAUDE_MODELS,
    ProviderName.MINIMAX: DEMO_MINIMAX_MODELS,
    ProviderName.KIMI_CODE: DEMO_KIMI_CODE_MODELS,
    ProviderName.OPENROUTER: DEMO_OPENROUTER_MODELS,
    ProviderName.OPENAI: DEMO_OPENAI_MODELS,
    ProviderName.CODEX: DEMO_CODEX_MODELS,
    ProviderName.GOOGLE: DEMO_GOOGLE_MODELS,
}

__all__ = [
    "DEMO_CLAUDE_MODELS",
    "DEMO_CODEX_MODELS",
    "DEMO_GOOGLE_MODELS",
    "DEMO_KIMI_CODE_MODELS",
    "DEMO_MINIMAX_MODELS",
    "DEMO_OPENAI_MODELS",
    "DEMO_OPENROUTER_MODELS",
    "DEMO_PROVIDER_MODELS",
]

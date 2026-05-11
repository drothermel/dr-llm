"""Shared model selections for demo scripts."""

from __future__ import annotations

from dr_llm.llm import (
    ApiLlmConfig,
    CLAUDE_CODE_MODELS,
    KIMI_CODING_MODELS,
    LlmConfig,
    MINIMAX_TEXT_MODELS,
    OpenAILlmConfig,
    ProviderName,
    build_default_registry,
    openrouter_allowed_models,
)

_THINKING_SWEEP_OPENAI_MODELS = [
    "gpt-5.4-mini-2026-03-17",
    "gpt-5-mini-2025-08-07",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-nano-2025-04-14",
]
_THINKING_SWEEP_CODEX_MODELS = [
    "gpt-5.1-codex-mini",
]
_THINKING_SWEEP_CLAUDE_MODELS = [
    model_id for model_id, _display_name in CLAUDE_CODE_MODELS
]
_THINKING_SWEEP_KIMI_CODE_MODELS = [
    model_id for model_id, _display_name in KIMI_CODING_MODELS
]
_THINKING_SWEEP_MINIMAX_MODELS = [
    model_id for model_id, _display_name in MINIMAX_TEXT_MODELS
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
_THINKING_SWEEP_OPENROUTER_MODELS = list(openrouter_allowed_models())

DEMO_THINKING_SWEEP_MODELS: dict[ProviderName, list[str]] = {
    ProviderName.CLAUDE_CODE: _THINKING_SWEEP_CLAUDE_MODELS,
    ProviderName.MINIMAX: _THINKING_SWEEP_MINIMAX_MODELS,
    ProviderName.KIMI_CODE: _THINKING_SWEEP_KIMI_CODE_MODELS,
    ProviderName.OPENROUTER: _THINKING_SWEEP_OPENROUTER_MODELS,
    ProviderName.OPENAI: _THINKING_SWEEP_OPENAI_MODELS,
    ProviderName.CODEX: _THINKING_SWEEP_CODEX_MODELS,
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
            "gpt-5-mini-default": OpenAILlmConfig(
                provider=ProviderName.OPENAI,
                model="gpt-5-mini",
                max_tokens=64,
                reasoning=registry.get(ProviderName.OPENAI)
                .reasoning_controls("gpt-5-mini")
                .default_reasoning,
            ),
            "gemini-flash-default": ApiLlmConfig(
                provider=ProviderName.GOOGLE,
                model="gemini-2.5-flash",
                max_tokens=64,
                reasoning=registry.get(ProviderName.GOOGLE)
                .reasoning_controls("gemini-2.5-flash")
                .default_reasoning,
            ),
        }
    finally:
        registry.close()


__all__ = [
    "DEMO_QUERY_DEFAULT_MODELS",
    "DEMO_THINKING_SWEEP_MODELS",
    "demo_pool_fill_llm_configs",
]

"""Shared model lineups for the demo_thinking_and_effort sweep.

Both ``scripts/demo_thinking_and_effort.py`` and the matching test in
``tests/scripts/test_demo_thinking_and_effort.py`` import from this module so
the test can verify the snapshot without loading the script via ``runpy``.
"""

from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import (
    CLAUDE_CODE_MODELS,
    KIMI_CODING_MODELS,
    MINIMAX_TEXT_MODELS,
)
from dr_llm.llm.providers.openrouter.policy import openrouter_allowed_models

OPENAI_MODELS = [
    "gpt-5.4-mini-2026-03-17",
    "gpt-5-mini-2025-08-07",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-nano-2025-04-14",
]
CODEX_MODELS = [
    "gpt-5.1-codex-mini",
]
CLAUDE_MODELS = [model_id for model_id, _display_name in CLAUDE_CODE_MODELS]
KIMI_CODE_MODELS = [model_id for model_id, _display_name in KIMI_CODING_MODELS]
MINIMAX_MODELS = [model_id for model_id, _display_name in MINIMAX_TEXT_MODELS]
GOOGLE_MODELS = [
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
OPENROUTER_MODELS = list(openrouter_allowed_models())
PROVIDER_MODELS: dict[str, list[str]] = {
    "claude-code": CLAUDE_MODELS,
    "minimax": MINIMAX_MODELS,
    "kimi-code": KIMI_CODE_MODELS,
    "openrouter": OPENROUTER_MODELS,
    "openai": OPENAI_MODELS,
    "codex": CODEX_MODELS,
    "google": GOOGLE_MODELS,
}

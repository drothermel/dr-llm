from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.core.base import Provider
from dr_llm.llm.providers.concepts.capabilities import ReasoningCapabilities

_CODEX_DOCS_URL = "https://developers.openai.com/codex/models"

_CODEX_MODELS = [
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.4-mini", "GPT-5.4 Mini"),
    ("gpt-5.3-codex", "GPT-5.3 Codex"),
    ("gpt-5.3-codex-spark", "GPT-5.3 Codex Spark (Pro only)"),
    ("gpt-5.2-codex", "GPT-5.2 Codex"),
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-5.1-codex-max", "GPT-5.1 Codex Max"),
    ("gpt-5.1-codex", "GPT-5.1 Codex"),
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5-codex", "GPT-5 Codex"),
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
    ("gpt-5", "GPT-5"),
]

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

_CLAUDE_CODE_DOCS_URL = "https://code.claude.com/docs/en/model-config"

_CLAUDE_CODE_MODELS = [
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]

_ANTHROPIC_COMMON_MODELS = [
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]

_KIMI_CODING_MODELS = [
    ("kimi-for-coding", "Kimi For Coding"),
]

_GOOGLE_COMMON_MODELS = [
    ("gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro"),
    ("gemini-2.5-flash-preview-04-17", "Gemini 2.5 Flash"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash"),
    ("gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite"),
]

_GLM_COMMON_MODELS = [
    ("glm-4.5", "GLM 4.5"),
    ("glm-4-air", "GLM 4 Air"),
    ("glm-4-flash", "GLM 4 Flash"),
]

_MINIMAX_DOCS_URL = "https://platform.minimax.io/docs/guides/models-intro"

_MINIMAX_TEXT_MODELS = [
    ("MiniMax-M2.7", "MiniMax M2.7"),
    ("MiniMax-M2.5", "MiniMax M2.5"),
    ("MiniMax-M2.1", "MiniMax M2.1 (legacy)"),
    ("MiniMax-M2", "MiniMax M2 (legacy)"),
]


def build_static_catalog_entries(
    *,
    provider: Provider,
    models: list[tuple[str, str]],
    docs_url: str,
    supports_vision: bool | None,
    capabilities_fn: Callable[[str], ReasoningCapabilities | None],
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    source_meta = {"source": "static", "docs_url": docs_url}
    now = datetime.now(UTC)
    entries = [
        ModelCatalogEntry(
            provider=provider.name,
            model=model_id,
            display_name=display_name,
            reasoning_capabilities=capabilities_fn(model_id),
            supports_vision=supports_vision,
            source_quality="static",
            fetched_at=now,
            metadata=source_meta,
        )
        for model_id, display_name in models
    ]
    return entries, source_meta

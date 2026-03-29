from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dr_llm.providers.headless.claude import ClaudeHeadlessAdapter
from dr_llm.providers.headless.claude_presets import (
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
)
from dr_llm.providers.headless.codex import CodexHeadlessAdapter
from dr_llm.providers.provider_adapter import ProviderAdapter
from dr_llm.catalog.models import ModelCatalogEntry

CODEX_DOCS_URL = "https://developers.openai.com/codex/models"

CODEX_MODELS = [
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
    ("gpt-5-codex-mini", "GPT-5 Codex Mini"),
    ("gpt-5", "GPT-5"),
]

CLAUDE_CODE_DOCS_URL = "https://code.claude.com/docs/en/model-config"

CLAUDE_CODE_MODELS = [
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]

KIMI_CODING_DOCS_URL = "https://www.kimi.com/code/docs/en/more/third-party-agents.html"

KIMI_CODING_MODELS = [
    ("kimi-for-coding", "Kimi For Coding"),
]

MINIMAX_DOCS_URL = "https://platform.minimax.io/docs/guides/models-intro"

MINIMAX_TEXT_MODELS = [
    ("MiniMax-M2.7", "MiniMax M2.7"),
    ("MiniMax-M2.5", "MiniMax M2.5"),
    ("MiniMax-M2.1", "MiniMax M2.1 (legacy)"),
    ("MiniMax-M2", "MiniMax M2 (legacy)"),
]


def fetch_static_headless_models(
    adapter: CodexHeadlessAdapter
    | ClaudeHeadlessAdapter
    | ClaudeHeadlessMiniMaxAdapter
    | ClaudeHeadlessKimiAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    if isinstance(adapter, CodexHeadlessAdapter):
        source_meta = {"source": "static", "docs_url": CODEX_DOCS_URL}
        entries = [
            ModelCatalogEntry(
                provider=adapter.name,
                model=model_id,
                display_name=display_name,
                supports_reasoning=True,
                supports_vision=None,
                source_quality="static",
                fetched_at=now,
                metadata=source_meta,
            )
            for model_id, display_name in CODEX_MODELS
        ]
        return entries, source_meta
    if isinstance(adapter, ClaudeHeadlessMiniMaxAdapter):
        source_meta = {"source": "static", "docs_url": MINIMAX_DOCS_URL}
        entries = [
            ModelCatalogEntry(
                provider=adapter.name,
                model=model_id,
                display_name=display_name,
                supports_reasoning=True,
                supports_vision=None,
                source_quality="static",
                fetched_at=now,
                metadata=source_meta,
            )
            for model_id, display_name in MINIMAX_TEXT_MODELS
        ]
        return entries, source_meta
    if isinstance(adapter, ClaudeHeadlessKimiAdapter):
        source_meta = {"source": "static", "docs_url": KIMI_CODING_DOCS_URL}
        entries = [
            ModelCatalogEntry(
                provider=adapter.name,
                model=model_id,
                display_name=display_name,
                supports_reasoning=True,
                supports_vision=True,
                source_quality="static",
                fetched_at=now,
                metadata=source_meta,
            )
            for model_id, display_name in KIMI_CODING_MODELS
        ]
        return entries, source_meta
    # Default: ClaudeHeadlessAdapter (native Anthropic)
    source_meta = {"source": "static", "docs_url": CLAUDE_CODE_DOCS_URL}
    entries = [
        ModelCatalogEntry(
            provider=adapter.name,
            model=model_id,
            display_name=display_name,
            supports_reasoning=True,
            supports_vision=True,
            source_quality="static",
            fetched_at=now,
            metadata=source_meta,
        )
        for model_id, display_name in CLAUDE_CODE_MODELS
    ]
    return entries, source_meta


def fetch_static_minimax_models(
    adapter: ProviderAdapter,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    source_meta = {"source": "static", "docs_url": MINIMAX_DOCS_URL}
    entries = [
        ModelCatalogEntry(
            provider=adapter.name,
            model=model_id,
            display_name=display_name,
            supports_reasoning=True,
            supports_vision=None,
            source_quality="static",
            fetched_at=now,
            metadata=source_meta,
        )
        for model_id, display_name in MINIMAX_TEXT_MODELS
    ]
    return entries, source_meta

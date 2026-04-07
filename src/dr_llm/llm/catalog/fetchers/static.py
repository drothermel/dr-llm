from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dr_llm.llm.providers.headless.codex import CodexHeadlessProvider
from dr_llm.llm.providers.headless.claude import ClaudeHeadlessProvider
from dr_llm.llm.providers.base import Provider
from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.providers.reasoning_capabilities import reasoning_capabilities_for_model

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
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
    ("gpt-5", "GPT-5"),
]

CLAUDE_CODE_DOCS_URL = "https://code.claude.com/docs/en/model-config"

CLAUDE_CODE_MODELS = [
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
]

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
    provider: CodexHeadlessProvider | ClaudeHeadlessProvider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    now = datetime.now(UTC)
    if isinstance(provider, CodexHeadlessProvider):
        source_meta = {"source": "static", "docs_url": CODEX_DOCS_URL}
        entries = [
            ModelCatalogEntry(
                provider=provider.name,
                model=model_id,
                display_name=display_name,
                reasoning_capabilities=reasoning_capabilities_for_model(
                    provider=provider.name,
                    model=model_id,
                ),
                supports_vision=None,
                source_quality="static",
                fetched_at=now,
                metadata=source_meta,
            )
            for model_id, display_name in CODEX_MODELS
        ]
        return entries, source_meta
    # Default: ClaudeHeadlessProvider (native Anthropic)
    source_meta = {"source": "static", "docs_url": CLAUDE_CODE_DOCS_URL}
    entries = [
        ModelCatalogEntry(
            provider=provider.name,
            model=model_id,
            display_name=display_name,
            reasoning_capabilities=reasoning_capabilities_for_model(
                provider=provider.name,
                model=model_id,
            ),
            supports_vision=True,
            source_quality="static",
            fetched_at=now,
            metadata=source_meta,
        )
        for model_id, display_name in CLAUDE_CODE_MODELS
    ]
    return entries, source_meta


def fetch_static_minimax_models(
    provider: Provider,
) -> tuple[list[ModelCatalogEntry], dict[str, Any]]:
    now = datetime.now(UTC)
    source_meta = {"source": "static", "docs_url": MINIMAX_DOCS_URL}
    entries = [
        ModelCatalogEntry(
            provider=provider.name,
            model=model_id,
            display_name=display_name,
            reasoning_capabilities=reasoning_capabilities_for_model(
                provider=provider.name,
                model=model_id,
            ),
            supports_vision=None,
            source_quality="static",
            fetched_at=now,
            metadata=source_meta,
        )
        for model_id, display_name in MINIMAX_TEXT_MODELS
    ]
    return entries, source_meta

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dr_llm.providers.base import ProviderAdapter
from dr_llm.providers.headless import (
    ClaudeHeadlessAdapter,
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
    CodexHeadlessAdapter,
)
from dr_llm.types import ModelCatalogEntry

MINIMAX_DOCS_URL = "https://platform.minimax.io/docs/guides/models-intro"

MINIMAX_TEXT_MODELS = [
    ("MiniMax-M2.7", "MiniMax M2.7"),
    ("MiniMax-M2.7-highspeed", "MiniMax M2.7 Highspeed"),
    ("MiniMax-M2.5", "MiniMax M2.5"),
    ("MiniMax-M2.5-highspeed", "MiniMax M2.5 Highspeed"),
    ("MiniMax-M2.1", "MiniMax M2.1 (legacy)"),
    ("MiniMax-M2.1-highspeed", "MiniMax M2.1 Highspeed (legacy)"),
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
        return (
            [
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="codex-latest",
                    display_name="Codex CLI Default",
                    supports_reasoning=None,
                    supports_tools=True,
                    supports_vision=None,
                    source_quality="static",
                    fetched_at=now,
                    metadata={"source": "static_default"},
                )
            ],
            {"source": "static_default"},
        )
    if isinstance(adapter, ClaudeHeadlessMiniMaxAdapter):
        return (
            [
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="MiniMax-M2.1",
                    display_name="MiniMax M2.1 (Claude headless preset)",
                    supports_reasoning=True,
                    supports_tools=True,
                    supports_vision=None,
                    source_quality="static",
                    fetched_at=now,
                    metadata={"source": "static_default"},
                )
            ],
            {"source": "static_default"},
        )
    if isinstance(adapter, ClaudeHeadlessKimiAdapter):
        return (
            [
                ModelCatalogEntry(
                    provider=adapter.name,
                    model="kimi-for-coding",
                    display_name="Kimi For Coding (Claude headless preset)",
                    supports_reasoning=True,
                    supports_tools=True,
                    supports_vision=True,
                    source_quality="static",
                    fetched_at=now,
                    metadata={"source": "static_default"},
                )
            ],
            {"source": "static_default"},
        )
    return (
        [
            ModelCatalogEntry(
                provider=adapter.name,
                model="claude-sonnet-4-6",
                display_name="Claude Sonnet 4.6",
                supports_reasoning=True,
                supports_tools=True,
                supports_vision=True,
                source_quality="static",
                fetched_at=now,
                metadata={"source": "static_default"},
            )
        ],
        {"source": "static_default"},
    )


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
            supports_tools=True,
            supports_vision=None,
            source_quality="static",
            fetched_at=now,
            metadata=source_meta,
        )
        for model_id, display_name in MINIMAX_TEXT_MODELS
    ]
    return entries, source_meta

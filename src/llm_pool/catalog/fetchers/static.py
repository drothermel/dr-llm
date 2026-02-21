from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llm_pool.providers.headless import (
    ClaudeHeadlessAdapter,
    ClaudeHeadlessKimiAdapter,
    ClaudeHeadlessMiniMaxAdapter,
    CodexHeadlessAdapter,
)
from llm_pool.types import ModelCatalogEntry


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

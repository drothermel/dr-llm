from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ProviderName(StrEnum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GLM = "glm"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    MINIMAX = "minimax"
    KIMI_CODE = "kimi-code"
    CODEX = "codex"
    CLAUDE_CODE = "claude-code"


class ProviderCategories(BaseModel):
    model_config = ConfigDict(frozen=True)

    openai: tuple[ProviderName, ...] = (ProviderName.OPENAI,)
    sampling_api: tuple[ProviderName, ...] = (
        ProviderName.OPENROUTER,
        ProviderName.GLM,
        ProviderName.GOOGLE,
        ProviderName.ANTHROPIC,
        ProviderName.MINIMAX,
    )
    kimi_code: tuple[ProviderName, ...] = (ProviderName.KIMI_CODE,)
    api_backed: tuple[ProviderName, ...] = (
        ProviderName.OPENAI,
        ProviderName.OPENROUTER,
        ProviderName.GLM,
        ProviderName.GOOGLE,
        ProviderName.ANTHROPIC,
        ProviderName.MINIMAX,
        ProviderName.KIMI_CODE,
    )
    headless: tuple[ProviderName, ...] = (
        ProviderName.CODEX,
        ProviderName.CLAUDE_CODE,
    )


type SamplingApiProviderName = Literal[
    ProviderName.OPENROUTER,
    ProviderName.GLM,
    ProviderName.GOOGLE,
    ProviderName.ANTHROPIC,
    ProviderName.MINIMAX,
]
type OpenAIProviderName = Literal[ProviderName.OPENAI]
type KimiCodeProviderName = Literal[ProviderName.KIMI_CODE]
type ApiBackedProviderName = (
    OpenAIProviderName | SamplingApiProviderName | KimiCodeProviderName
)
type HeadlessProviderName = Literal[
    ProviderName.CODEX, ProviderName.CLAUDE_CODE
]

from __future__ import annotations

from enum import StrEnum
from typing import Literal


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


API_PROVIDER_NAMES = (
    ProviderName.OPENAI,
    ProviderName.OPENROUTER,
    ProviderName.GLM,
    ProviderName.GOOGLE,
    ProviderName.ANTHROPIC,
    ProviderName.MINIMAX,
    ProviderName.KIMI_CODE,
)
SAMPLING_API_PROVIDER_NAMES = (
    ProviderName.OPENROUTER,
    ProviderName.GLM,
    ProviderName.GOOGLE,
    ProviderName.ANTHROPIC,
    ProviderName.MINIMAX,
)
HEADLESS_PROVIDER_NAMES = (ProviderName.CODEX, ProviderName.CLAUDE_CODE)

type ApiProviderName = Literal[
    ProviderName.OPENROUTER,
    ProviderName.GLM,
    ProviderName.GOOGLE,
    ProviderName.ANTHROPIC,
    ProviderName.MINIMAX,
]
type OpenAIProviderName = Literal[ProviderName.OPENAI]
type KimiCodeProviderName = Literal[ProviderName.KIMI_CODE]
type ApiBackedProviderName = (
    OpenAIProviderName | ApiProviderName | KimiCodeProviderName
)
type HeadlessProviderName = Literal[
    ProviderName.CODEX, ProviderName.CLAUDE_CODE
]

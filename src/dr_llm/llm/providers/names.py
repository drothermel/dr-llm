from __future__ import annotations

from enum import StrEnum


class ApiKeyNames(StrEnum):
    OPENAI = "OPENAI_API_KEY"
    OPENROUTER = "OPENROUTER_API_KEY"
    ANTHROPIC = "ANTHROPIC_API_KEY"
    GOOGLE = "GOOGLE_API_KEY"
    GLM = "ZAI_API_KEY"
    MINIMAX = "MINIMAX_API_KEY"
    KIMI = "KIMI_API_KEY"

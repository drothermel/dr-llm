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


class ThinkingLevel(StrEnum):
    NA = "na"
    OFF = "off"
    BUDGET = "budget"
    ADAPTIVE = "adaptive"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class EffortSpec(StrEnum):
    NA = "na"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class ReasoningMode(StrEnum):
    UNSUPPORTED = "unsupported"
    OPENAI_EFFORT = "openai_effort"
    OPENROUTER_TOGGLE = "openrouter_toggle"
    OPENROUTER_EFFORT = "openrouter_effort"
    GLM = "glm"
    GOOGLE_BUDGET = "google_budget"
    GOOGLE_LEVEL = "google_level"
    ANTHROPIC_BUDGET = "anthropic_budget"
    ANTHROPIC_EFFORT = "anthropic_effort"
    ANTHROPIC_EFFORT_AND_BUDGET = "anthropic_effort_and_budget"
    CLAUDE_CLI_EFFORT = "claude_cli_effort"
    CODEX_CLI_EFFORT = "codex_cli_effort"
    KIMI_CODE_EFFORT_AND_BUDGET = "kimi_code_effort_and_budget"
    MINIMAX_EFFORT = "minimax_effort"


class ControlStrategy(StrEnum):
    REASONING = "reasoning"
    EFFORT = "effort"
    NONE = "none"


class ReasoningWarningCode(StrEnum):
    UNSUPPORTED_FOR_PROVIDER = "unsupported_for_provider"
    MAPPED_WITH_HEURISTIC = "mapped_with_heuristic"
    PARTIALLY_SUPPORTED = "partially_supported"


OpenRouterEffortLevel = Literal["low", "medium", "high"]


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

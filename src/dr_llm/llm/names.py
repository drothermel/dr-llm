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


class ReasoningWarningCode(StrEnum):
    UNSUPPORTED_FOR_PROVIDER = "unsupported_for_provider"
    MAPPED_WITH_HEURISTIC = "mapped_with_heuristic"
    PARTIALLY_SUPPORTED = "partially_supported"


OpenRouterEffortLevel = Literal["low", "medium", "high"]

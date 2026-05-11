from __future__ import annotations

from dr_llm.llm.names import ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import ReasoningCapabilities
from dr_llm.llm.providers.concepts.thinking_utils import matches_family
from dr_llm.llm.providers.impls.codex.families import (
    CODEX_MINIMAL_THINKING_SUPPORTED_MODELS,
    CODEX_OFF_THINKING_SUPPORTED_MODELS,
    CODEX_THINKING_SUPPORTED_MODELS,
)

_CODEX_CLI_EFFORT_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.CODEX_CLI_EFFORT
)


def codex_supports_configurable_thinking(model: str) -> bool:
    return matches_family(
        normalized=model, families=CODEX_THINKING_SUPPORTED_MODELS
    )


def codex_supports_minimal_thinking(model: str) -> bool:
    return matches_family(
        normalized=model,
        families=CODEX_MINIMAL_THINKING_SUPPORTED_MODELS,
    )


def codex_supports_off_thinking(model: str) -> bool:
    return matches_family(
        normalized=model,
        families=CODEX_OFF_THINKING_SUPPORTED_MODELS,
    )


def reasoning_capabilities_for_codex(
    model: str,
) -> ReasoningCapabilities | None:
    if codex_supports_configurable_thinking(model):
        return _CODEX_CLI_EFFORT_CAPS
    return None

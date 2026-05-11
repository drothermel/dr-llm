from __future__ import annotations

from dr_llm.llm.names import ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import ReasoningCapabilities
from dr_llm.llm.providers.concepts.model_family import (
    model_matches_any_family,
)
from dr_llm.llm.providers.impls.codex.families import (
    CODEX_MINIMAL_THINKING_SUPPORTED_MODELS,
    CODEX_OFF_THINKING_SUPPORTED_MODELS,
    CODEX_THINKING_SUPPORTED_MODELS,
)


def codex_supports_configurable_thinking(model: str) -> bool:
    return model_matches_any_family(model, CODEX_THINKING_SUPPORTED_MODELS)


def codex_supports_minimal_thinking(model: str) -> bool:
    return model_matches_any_family(
        model, CODEX_MINIMAL_THINKING_SUPPORTED_MODELS
    )


def codex_supports_off_thinking(model: str) -> bool:
    return model_matches_any_family(model, CODEX_OFF_THINKING_SUPPORTED_MODELS)


def reasoning_capabilities_for_codex(
    model: str,
) -> ReasoningCapabilities | None:
    if codex_supports_configurable_thinking(model):
        return ReasoningCapabilities(mode=ReasoningMode.CODEX_CLI_EFFORT)
    return None

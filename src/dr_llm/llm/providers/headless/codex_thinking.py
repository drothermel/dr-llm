from __future__ import annotations

from dr_llm.llm.providers.reasoning_capability_types import ReasoningCapabilities
from dr_llm.llm.providers.thinking_utils import matches_family

_CODEX_CLI_EFFORT_CAPS = ReasoningCapabilities(mode="codex_cli_effort")

CODEX_THINKING_SUPPORTED_MODELS = [
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4-mini",
]

CODEX_MINIMAL_THINKING_SUPPORTED_MODELS = [
    "gpt-5",
]

CODEX_OFF_THINKING_SUPPORTED_MODELS = [
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.4-mini",
]


def codex_supports_configurable_thinking(model: str) -> bool:
    return matches_family(normalized=model, families=CODEX_THINKING_SUPPORTED_MODELS)


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


def reasoning_capabilities_for_codex(model: str) -> ReasoningCapabilities | None:
    if codex_supports_configurable_thinking(model):
        return _CODEX_CLI_EFFORT_CAPS
    return None

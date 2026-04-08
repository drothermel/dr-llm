from __future__ import annotations

from dr_llm.llm.providers.thinking_utils import matches_family

OPENAI_THINKING_SUPPORTED_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-nano",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-5.2-nano",
    "gpt-5.2-codex",
    "gpt-5.3",
    "gpt-5.3-mini",
    "gpt-5.3-nano",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]

OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]

OPENAI_OFF_THINKING_SUPPORTED_MODELS = [
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-nano",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-5.2-nano",
    "gpt-5.2-codex",
    "gpt-5.3",
    "gpt-5.3-mini",
    "gpt-5.3-nano",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]


def normalize_openai_reasoning_model(model: str) -> str:
    if model.startswith("openai/"):
        return model[len("openai/") :]
    return model


def openai_supports_configurable_thinking(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return matches_family(
        normalized=normalized,
        families=OPENAI_THINKING_SUPPORTED_MODELS,
    )


def openai_supports_minimal_thinking(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return matches_family(
        normalized=normalized,
        families=OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS,
    )


def openai_supports_off_thinking(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return matches_family(
        normalized=normalized,
        families=OPENAI_OFF_THINKING_SUPPORTED_MODELS,
    )

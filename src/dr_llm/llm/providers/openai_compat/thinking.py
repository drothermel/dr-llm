from __future__ import annotations

from dr_llm.llm.providers.reasoning import OpenAIReasoning, ReasoningSpec, ThinkingLevel
from dr_llm.llm.providers.reasoning_capability_types import ReasoningCapabilities
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

OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS = [
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-5.2-nano",
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


def openai_uses_max_completion_tokens(model: str) -> bool:
    """OpenAI gpt-5 family models reject ``max_tokens`` and require
    ``max_completion_tokens`` instead. The set of affected models matches
    the configurable-thinking family.
    """
    return openai_supports_configurable_thinking(model)


def openai_is_gpt5_family(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return normalized == "gpt-5" or normalized.startswith(("gpt-5-", "gpt-5."))


def openai_supports_sampling_with_reasoning_off(model: str) -> bool:
    normalized = normalize_openai_reasoning_model(model)
    return matches_family(
        normalized=normalized,
        families=OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS,
    )


def validate_openai_sampling_controls(
    *,
    model: str,
    reasoning: ReasoningSpec | None,
    temperature: float | None,
    top_p: float | None,
) -> None:
    if temperature is None and top_p is None:
        return
    if not openai_is_gpt5_family(model):
        return
    if not openai_supports_sampling_with_reasoning_off(model):
        raise ValueError(
            "OpenAI custom temperature/top_p controls are only supported for "
            "gpt-5.2 and gpt-5.4 families with "
            "OpenAIReasoning(thinking_level='off'); "
            f"model={model!r} does not support them"
        )
    if reasoning != OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
        raise ValueError(
            "OpenAI custom temperature/top_p controls require "
            "OpenAIReasoning(thinking_level='off') "
            f"for model={model!r}"
        )


def reasoning_capabilities_for_openai(model: str) -> ReasoningCapabilities | None:
    if openai_supports_configurable_thinking(model):
        return ReasoningCapabilities(mode="openai_effort")
    return None

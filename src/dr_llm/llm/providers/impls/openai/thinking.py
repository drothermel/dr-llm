from __future__ import annotations

from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import ReasoningCapabilities
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    ReasoningSpec,
)
from dr_llm.llm.providers.concepts.thinking_utils import matches_family
from dr_llm.llm.providers.impls.openai.families import (
    OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS,
    OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS,
    OPENAI_OFF_THINKING_SUPPORTED_MODELS,
    OPENAI_THINKING_SUPPORTED_MODELS,
)

OPENAI_TEMP_TOPP_UNSUPPORTED_MSG = (
    "OpenAI custom temperature/top_p controls are only supported for "
    "gpt-5.2 and gpt-5.4 families with "
    "OpenAIReasoning(thinking_level='off'); "
    "model={model!r} does not support them"
)

OPENAI_TEMP_TOPP_REASONING_REQUIRED_MSG = (
    "OpenAI custom temperature/top_p controls require "
    "OpenAIReasoning(thinking_level='off') "
    "for model={model!r}"
)


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
    sampling: SamplingControls | None,
) -> None:
    if sampling is None or sampling.is_empty():
        return
    if not openai_is_gpt5_family(model):
        return
    if not openai_supports_sampling_with_reasoning_off(model):
        raise ValueError(OPENAI_TEMP_TOPP_UNSUPPORTED_MSG.format(model=model))
    if reasoning != OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
        raise ValueError(
            OPENAI_TEMP_TOPP_REASONING_REQUIRED_MSG.format(model=model)
        )


def reasoning_capabilities_for_openai(
    model: str,
) -> ReasoningCapabilities | None:
    if openai_supports_configurable_thinking(model):
        return ReasoningCapabilities(mode=ReasoningMode.OPENAI_EFFORT)
    return None

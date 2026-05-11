from __future__ import annotations

from typing import Literal

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import ProviderName, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import ReasoningCapabilities
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    OpenAIReasoning,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    unsupported_reasoning_kind_message,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.concepts.model_family import (
    model_matches_any_family,
)
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


def openai_supports_configurable_thinking(model: str) -> bool:
    return model_matches_any_family(model, OPENAI_THINKING_SUPPORTED_MODELS)


def openai_supports_minimal_thinking(model: str) -> bool:
    return model_matches_any_family(
        model, OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS
    )


def openai_supports_off_thinking(model: str) -> bool:
    return model_matches_any_family(
        model, OPENAI_OFF_THINKING_SUPPORTED_MODELS
    )


def openai_is_gpt5_family(model: str) -> bool:
    return model_matches_any_family(model, OPENAI_THINKING_SUPPORTED_MODELS)


def openai_supports_sampling_with_reasoning_off(model: str) -> bool:
    return model_matches_any_family(
        model, OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS
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


def validate_reasoning_for_openai(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    def _validate_native(spec: OpenAIReasoning) -> None:
        if not openai_supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.OPENAI} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.OPENAI,
            model=model,
            thinking_level=spec.thinking_level,
            supports_off=openai_supports_off_thinking(model),
            supports_minimal=openai_supports_minimal_thinking(model),
        )

    def _validate_top_budget(budget: ReasoningBudget) -> None:
        del budget
        raise ValueError(
            f"Top-level reasoning budgets are not supported for provider='{ProviderName.OPENAI}' model={model!r}; use OpenAIReasoning(thinking_level=...)"
        )

    dispatch_reasoning_validation(
        provider=ProviderName.OPENAI,
        model=model,
        reasoning=reasoning,
        native_spec_type=OpenAIReasoning,
        requires_reasoning=openai_supports_configurable_thinking(model),
        validate_native=_validate_native,
        validate_top_budget=_validate_top_budget,
    )


class OpenAIReasoningConfig(BaseProviderReasoningConfig):
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high"] | None
    ) = None

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> OpenAIReasoningConfig:
        if config is None:
            return cls()
        match config:
            case OpenAIReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(reasoning_effort="none")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL):
                return cls(reasoning_effort="minimal")
            case OpenAIReasoning(thinking_level=ThinkingLevel.LOW):
                return cls(reasoning_effort="low")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MEDIUM):
                return cls(reasoning_effort="medium")
            case OpenAIReasoning(thinking_level=ThinkingLevel.HIGH):
                return cls(reasoning_effort="high")
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(ProviderName.OPENAI, config)
        )

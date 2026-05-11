from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.effort import validate_effort
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.providers.impls.openai.families import (
    OPENAI_FAMILIES,
    OpenAIFamilies,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


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


def _validate_openai_sampling_controls(
    *,
    model: str,
    reasoning: ReasoningSpec | None,
    sampling: SamplingControls | None,
    families: OpenAIFamilies | None = None,
) -> None:
    families = families or OPENAI_FAMILIES
    if sampling is None or sampling.is_empty():
        return
    if not families.supports_configurable_thinking(model):
        return
    if not families.supports_sampling_with_reasoning_off(model):
        raise ValueError(OPENAI_TEMP_TOPP_UNSUPPORTED_MSG.format(model=model))
    if reasoning != OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
        raise ValueError(
            OPENAI_TEMP_TOPP_REASONING_REQUIRED_MSG.format(model=model)
        )


def _validate_reasoning_for_openai(
    *,
    model: str,
    reasoning: ReasoningSpec | None,
    families: OpenAIFamilies | None = None,
) -> None:
    families = families or OPENAI_FAMILIES

    def _validate_native(spec: OpenAIReasoning) -> None:
        if not families.supports_configurable_thinking(model):
            raise ValueError(
                f"{ProviderName.OPENAI} thinking is not supported for model={model!r}"
            )
        validate_discrete_thinking_level(
            provider=ProviderName.OPENAI,
            model=model,
            thinking_level=spec.thinking_level,
            supports_off=families.supports_off_thinking(model),
            supports_minimal=families.supports_minimal_thinking(model),
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
        requires_reasoning=families.supports_configurable_thinking(model),
        validate_native=_validate_native,
        validate_top_budget=_validate_top_budget,
    )


class OpenAIControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.OPENAI
    model: str
    mode: CallMode
    families: OpenAIFamilies = Field(
        default_factory=OpenAIFamilies, exclude=True
    )

    @property
    def control_mode(self) -> ControlMode:
        return self.families.control_mode(self.model)

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        return self.families.supported_thinking_levels(self.model)

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        return self.families.default_thinking_level(self.model)

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return self.families.supported_effort_levels(self.model)

    @property
    def default_effort(self) -> EffortSpec:
        return self.families.default_effort(self.model)

    @property
    def default_reasoning(self) -> ReasoningSpec | None:
        return self.reasoning_for_thinking_level(
            thinking_level=self.default_thinking_level
        )

    @property
    def catalog_metadata(self) -> dict[str, Any]:
        return {
            "control_mode": self.control_mode,
            "supported_thinking_levels": self.supported_thinking_levels,
            "default_thinking_level": self.default_thinking_level,
            "supported_effort_levels": self.supported_effort_levels,
            "default_effort": self.default_effort,
        }

    def request_defaults(self) -> ProviderRequestDefaults:
        return ProviderRequestDefaults(
            provider=self.provider,
            model=self.model,
            mode=self.mode,
            effort=self.default_effort,
            reasoning=self.default_reasoning,
            sampling_supported=True,
        )

    def resolve_reasoning(
        self,
        *,
        reasoning: ReasoningSpec | None,
        thinking_level: ThinkingLevel | None,
        budget_tokens: int | None,
    ) -> ReasoningSpec | None:
        if reasoning is not None:
            return reasoning
        if thinking_level is not None:
            return self.reasoning_for_thinking_level(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
            )
        return self.default_reasoning

    def resolve_effort(self, effort: EffortSpec | None) -> EffortSpec:
        if effort is None:
            return self.default_effort
        return effort

    def resolve_sampling(
        self, sampling: SamplingControls | None
    ) -> SamplingControls | None:
        if sampling is None or sampling.is_empty():
            return None
        return sampling

    def reasoning_for_thinking_level(
        self,
        *,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del budget_tokens
        if thinking_level == ThinkingLevel.NA:
            return None
        return OpenAIReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list:
        validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        _validate_reasoning_for_openai(
            model=request.model,
            reasoning=request.reasoning,
            families=self.families,
        )

        _validate_openai_sampling_controls(
            model=request.model,
            reasoning=request.reasoning,
            sampling=request.sampling,
            families=self.families,
        )
        return []

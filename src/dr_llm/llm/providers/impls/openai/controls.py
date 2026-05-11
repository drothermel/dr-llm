from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.model_family import (
    is_snapshot_of_family,
    model_matches_any_family,
)
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    OpenAIReasoning,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    unsupported_reasoning_kind_message,
    validate_discrete_thinking_level,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class OpenAIModelFamily(StrEnum):
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"
    GPT51 = "gpt-5.1"
    GPT51_MINI = "gpt-5.1-mini"
    GPT51_NANO = "gpt-5.1-nano"
    GPT51_CODEX = "gpt-5.1-codex"
    GPT51_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT51_CODEX_MAX = "gpt-5.1-codex-max"
    GPT52 = "gpt-5.2"
    GPT52_MINI = "gpt-5.2-mini"
    GPT52_NANO = "gpt-5.2-nano"
    GPT52_CODEX = "gpt-5.2-codex"
    GPT53 = "gpt-5.3"
    GPT53_MINI = "gpt-5.3-mini"
    GPT53_NANO = "gpt-5.3-nano"
    GPT53_CODEX = "gpt-5.3-codex"
    GPT54 = "gpt-5.4"
    GPT54_MINI = "gpt-5.4-mini"
    GPT54_NANO = "gpt-5.4-nano"

    def in_family(self, model: str) -> bool:
        normalized = _normalize_openai_model(model)
        return normalized == self or is_snapshot_of_family(
            model=normalized, family=str(self)
        )


def _normalize_openai_model(model: str) -> str:
    if model.startswith("openai/"):
        return model[len("openai/") :]
    return model


OPENAI_GPT5_FAMILIES = (
    OpenAIModelFamily.GPT5,
    OpenAIModelFamily.GPT5_MINI,
    OpenAIModelFamily.GPT5_NANO,
)
OPENAI_GPT51_FAMILIES = (
    OpenAIModelFamily.GPT51,
    OpenAIModelFamily.GPT51_MINI,
    OpenAIModelFamily.GPT51_NANO,
    OpenAIModelFamily.GPT51_CODEX,
    OpenAIModelFamily.GPT51_CODEX_MINI,
    OpenAIModelFamily.GPT51_CODEX_MAX,
)
OPENAI_GPT52_FAMILIES = (
    OpenAIModelFamily.GPT52,
    OpenAIModelFamily.GPT52_MINI,
    OpenAIModelFamily.GPT52_NANO,
    OpenAIModelFamily.GPT52_CODEX,
)
OPENAI_GPT53_FAMILIES = (
    OpenAIModelFamily.GPT53,
    OpenAIModelFamily.GPT53_MINI,
    OpenAIModelFamily.GPT53_NANO,
    OpenAIModelFamily.GPT53_CODEX,
)
OPENAI_GPT54_FAMILIES = (
    OpenAIModelFamily.GPT54,
    OpenAIModelFamily.GPT54_MINI,
    OpenAIModelFamily.GPT54_NANO,
)
OPENAI_THINKING_SUPPORTED_MODELS = (
    *OPENAI_GPT5_FAMILIES,
    *OPENAI_GPT51_FAMILIES,
    *OPENAI_GPT52_FAMILIES,
    *OPENAI_GPT53_FAMILIES,
    *OPENAI_GPT54_FAMILIES,
)
OPENAI_MINIMAL_THINKING_SUPPORTED_MODELS = OPENAI_GPT5_FAMILIES
OPENAI_OFF_THINKING_SUPPORTED_MODELS = (
    *OPENAI_GPT51_FAMILIES,
    *OPENAI_GPT52_FAMILIES,
    *OPENAI_GPT53_FAMILIES,
    *OPENAI_GPT54_FAMILIES,
)
OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS = (
    OpenAIModelFamily.GPT52,
    OpenAIModelFamily.GPT52_MINI,
    OpenAIModelFamily.GPT52_NANO,
    OpenAIModelFamily.GPT54,
    OpenAIModelFamily.GPT54_MINI,
    OpenAIModelFamily.GPT54_NANO,
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


def reasoning_mode_for_openai(model: str) -> ReasoningMode:
    if openai_supports_configurable_thinking(model):
        return ReasoningMode.OPENAI_EFFORT
    return ReasoningMode.UNSUPPORTED


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


class OpenAIControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.OPENAI
    model: str
    mode: CallMode

    @property
    def supports_reasoning(self) -> bool:
        return self.reasoning_mode != ReasoningMode.UNSUPPORTED

    @property
    def reasoning_mode(self) -> ReasoningMode:
        return reasoning_mode_for_openai(self.model)

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        if not openai_supports_configurable_thinking(self.model):
            return (ThinkingLevel.NA,)
        levels: list[ThinkingLevel] = []
        if openai_supports_off_thinking(self.model):
            levels.append(ThinkingLevel.OFF)
        elif openai_supports_minimal_thinking(self.model):
            levels.append(ThinkingLevel.MINIMAL)
        levels.extend(
            [ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH]
        )
        return tuple(levels)

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        levels = self.supported_thinking_levels
        for level in (
            ThinkingLevel.OFF,
            ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW,
        ):
            if level in levels:
                return level
        return ThinkingLevel.NA

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return ()

    @property
    def default_effort(self) -> EffortSpec:
        return EffortSpec.NA

    @property
    def default_reasoning(self) -> ReasoningSpec | None:
        return self.reasoning_for_thinking_level(
            thinking_level=self.default_thinking_level
        )

    @property
    def catalog_metadata(self) -> dict[str, Any]:
        return {
            "reasoning_mode": self.reasoning_mode,
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
        _validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        validate_reasoning_for_openai(
            model=request.model, reasoning=request.reasoning
        )
        validate_openai_sampling_controls(
            model=request.model,
            reasoning=request.reasoning,
            sampling=request.sampling,
        )
        return []


def _validate_effort(
    *,
    provider: str,
    model: str,
    effort: EffortSpec,
    supported_effort_levels: tuple[EffortSpec, ...],
) -> None:
    if not supported_effort_levels:
        if effort != EffortSpec.NA:
            raise ValueError(
                f"effort is not supported for provider={provider!r} "
                f"model={model!r}"
            )
        return
    if effort == EffortSpec.NA:
        raise ValueError(
            f"effort is required for provider={provider!r} model={model!r}"
        )
    if effort not in supported_effort_levels:
        allowed = ", ".join(str(level) for level in supported_effort_levels)
        raise ValueError(
            f"effort={effort!r} is not supported for provider={provider!r} "
            f"model={model!r}; allowed levels: {allowed}"
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

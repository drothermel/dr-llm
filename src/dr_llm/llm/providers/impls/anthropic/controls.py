from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.effort import (
    validate_effort,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    BaseProviderControlMapping,
    ReasoningBudget,
    ReasoningSpec,
    dispatch_reasoning_validation,
    is_control_unsupported,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
    validate_budget_range,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.providers.impls.anthropic.families import (
    ANTHROPIC_FAMILIES,
    AnthropicFamilies,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class AnthropicThinkingType(StrEnum):
    ENABLED = "enabled"
    ADAPTIVE = "adaptive"


ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
ANTHROPIC_DEFAULT_SAMPLING = SamplingControls(temperature=1.0, top_p=0.95)


def _validate_budget_for_anthropic(
    *,
    model: str,
    budget_tokens: int | None,
    min_budget_tokens: int | None,
    max_budget_tokens: int | None,
    families: AnthropicFamilies | None = None,
) -> None:
    families = families or ANTHROPIC_FAMILIES
    if budget_tokens is None:
        raise ValueError(
            f"{ProviderName.ANTHROPIC} budget thinking requires budget_tokens when "
            "thinking_level is 'budget'"
        )
    if (
        not families.supports_budget_thinking(model)
        or min_budget_tokens is None
        or max_budget_tokens is None
    ):
        raise ValueError(
            f"{ProviderName.ANTHROPIC} budget thinking is not supported for "
            f"model={model!r}"
        )
    validate_budget_range(
        provider=ProviderName.ANTHROPIC,
        model=model,
        label=f"{ProviderName.ANTHROPIC} budget_tokens",
        tokens=budget_tokens,
        min_budget_tokens=min_budget_tokens,
        max_budget_tokens=max_budget_tokens,
    )


def _validate_reasoning_for_anthropic(
    *,
    model: str,
    reasoning: ReasoningSpec | None,
    families: AnthropicFamilies | None = None,
) -> None:
    families = families or ANTHROPIC_FAMILIES
    controls = AnthropicControls(
        model=model, mode=CallMode.api, families=families
    )
    dispatch_reasoning_validation(
        provider=ProviderName.ANTHROPIC,
        model=model,
        reasoning=reasoning,
        native_spec_type=AnthropicReasoning,
        requires_reasoning=not is_control_unsupported(controls.control_mode),
        validate_native=lambda spec: _validate_anthropic_reasoning_shape(
            model=model, reasoning=spec, controls=controls
        ),
        validate_top_budget=lambda budget: _validate_budget_for_anthropic(
            model=model,
            budget_tokens=budget.tokens,
            min_budget_tokens=controls.min_budget_tokens,
            max_budget_tokens=controls.max_budget_tokens,
            families=controls.families,
        ),
    )


def _validate_anthropic_reasoning_shape(
    *,
    model: str,
    reasoning: AnthropicReasoning,
    controls: "AnthropicControls",
) -> None:
    thinking_level = reasoning.thinking_level
    if (
        thinking_level != ThinkingLevel.BUDGET
        and reasoning.budget_tokens is not None
    ):
        raise ValueError(
            "anthropic budget_tokens are only allowed with thinking_level='budget'"
        )
    if thinking_level == ThinkingLevel.NA:
        if not is_control_unsupported(controls.control_mode):
            raise ValueError(
                f"thinking_level='na' is not supported for provider='{ProviderName.ANTHROPIC}' model={model!r}"
            )
        return
    if thinking_level == ThinkingLevel.OFF:
        return
    if thinking_level == ThinkingLevel.ADAPTIVE:
        if not controls.families.supports_adaptive_thinking(model):
            raise ValueError(
                f"anthropic adaptive thinking is not supported for model={model!r}"
            )
        return
    if thinking_level == ThinkingLevel.BUDGET:
        _validate_budget_for_anthropic(
            model=model,
            budget_tokens=reasoning.budget_tokens,
            min_budget_tokens=controls.min_budget_tokens,
            max_budget_tokens=controls.max_budget_tokens,
            families=controls.families,
        )
        return
    raise ValueError(
        f"Unsupported anthropic thinking level {thinking_level!r} for model={model!r}"
    )


class AnthropicControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.ANTHROPIC
    model: str
    mode: CallMode
    families: AnthropicFamilies = Field(
        default_factory=AnthropicFamilies, exclude=True
    )

    @property
    def control_mode(self) -> ControlMode:
        return self.families.control_mode(self.model)

    @property
    def min_budget_tokens(self) -> int | None:
        return self.families.budget_min_for_model(self.model)

    @property
    def max_budget_tokens(self) -> int | None:
        return self.families.budget_max_for_model(self.model)

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
            thinking_level=self.default_thinking_level,
            budget_tokens=self.min_budget_tokens,
        )

    @property
    def catalog_metadata(self) -> dict[str, Any]:
        return {
            "control_mode": self.control_mode,
            "min_budget_tokens": self.min_budget_tokens,
            "max_budget_tokens": self.max_budget_tokens,
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
            max_tokens=ANTHROPIC_DEFAULT_MAX_TOKENS,
            max_tokens_required=True,
            effort=self.default_effort,
            reasoning=self.default_reasoning,
            sampling_supported=True,
            sampling=ANTHROPIC_DEFAULT_SAMPLING,
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
        if effort == EffortSpec.NA and self.default_effort != EffortSpec.NA:
            return self.default_effort
        return effort

    def resolve_sampling(
        self, sampling: SamplingControls | None
    ) -> SamplingControls | None:
        if sampling is not None:
            if sampling.is_empty():
                return None
            return sampling
        return ANTHROPIC_DEFAULT_SAMPLING

    def reasoning_for_thinking_level(
        self,
        *,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        if thinking_level == ThinkingLevel.NA:
            return None
        if thinking_level == ThinkingLevel.BUDGET:
            return AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=_require_budget_tokens(
                    provider=self.provider,
                    budget_tokens=budget_tokens,
                ),
            )
        return AnthropicReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list:
        self._validate_max_tokens_required(request)
        validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        _validate_reasoning_for_anthropic(
            model=request.model,
            reasoning=request.reasoning,
            families=self.families,
        )
        return []

    def _validate_max_tokens_required(self, request: LlmRequest) -> None:
        if request.max_tokens is None:
            raise ValueError(
                f"max_tokens is required for provider={self.provider!r}"
            )


def _require_budget_tokens(*, provider: str, budget_tokens: int | None) -> int:
    if budget_tokens is None:
        raise ValueError(f"{provider} budget thinking requires budget_tokens")
    return budget_tokens


class AnthropicControlMapping(BaseProviderControlMapping):
    thinking: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> AnthropicControlMapping:
        if config is None:
            return cls()
        match config:
            case ReasoningBudget(tokens=tokens):
                return cls(
                    thinking={
                        "type": AnthropicThinkingType.ENABLED,
                        "budget_tokens": tokens,
                    }
                )
            case AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                display=display,
            ):
                thinking: dict[str, Any] = {}
                if thinking_level == ThinkingLevel.BUDGET:
                    tokens = require_budget_tokens(
                        budget_tokens,
                        label=ProviderName.ANTHROPIC,
                        min_value=1,
                    )
                    thinking = {
                        "type": AnthropicThinkingType.ENABLED,
                        "budget_tokens": tokens,
                    }
                elif thinking_level == ThinkingLevel.ADAPTIVE:
                    thinking = {"type": AnthropicThinkingType.ADAPTIVE}
                if display is not None:
                    thinking["display"] = display
                return cls(thinking=thinking)
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.ANTHROPIC, config
                    )
                )

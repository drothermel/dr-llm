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
from dr_llm.llm.providers.concepts.effort import FULL_EFFORT
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    BaseProviderControlMapping,
    ReasoningBudget,
    ReasoningSpec,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
)
from dr_llm.llm.providers.impls.anthropic.controls import (
    validate_anthropic_budget_for_provider,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class KimiCodeModelFamily(StrEnum):
    KIMI_FOR_CODING = "kimi-for-coding"

    def in_family(self, model: str) -> bool:
        return model == self


KIMI_CODE_SUPPORTED_MODELS = (KimiCodeModelFamily.KIMI_FOR_CODING,)


def kimi_code_control_mode(model: str) -> ControlMode:
    if KimiCodeModelFamily.KIMI_FOR_CODING.in_family(model):
        return ControlMode.KIMI_CODE_EFFORT_AND_BUDGET
    return ControlMode.UNSUPPORTED


def supported_effort_levels_for_kimi_code(
    model: str,
) -> tuple[EffortSpec, ...]:
    if kimi_code_control_mode(model) == ControlMode.UNSUPPORTED:
        return ()
    return FULL_EFFORT


def validate_reasoning_for_kimi_code(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if reasoning is None:
        return
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            "kimi-code requires anthropic reasoning configs; "
            "use AnthropicReasoning(thinking_level='budget', budget_tokens=...)"
        )
    if not isinstance(reasoning, AnthropicReasoning):
        raise ValueError(
            f"kimi-code reasoning is not supported for kind={reasoning.kind!r}"
        )
    if reasoning.display is not None:
        raise ValueError(
            "kimi-code does not support anthropic display controls"
        )
    if reasoning.thinking_level not in {
        ThinkingLevel.NA,
        ThinkingLevel.OFF,
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.BUDGET,
    }:
        raise ValueError(
            "kimi-code supports only anthropic thinking levels "
            "'na', 'off', 'adaptive', and 'budget'"
        )
    if reasoning.thinking_level == ThinkingLevel.BUDGET:
        validate_anthropic_budget_for_provider(
            provider=ProviderName.KIMI_CODE,
            model=model,
            budget_tokens=reasoning.budget_tokens,
            min_budget_tokens=1024,
            max_budget_tokens=128000,
        )
        return
    if reasoning.budget_tokens is not None:
        raise ValueError(
            "kimi-code budget_tokens are only valid when "
            "thinking_level is 'budget'"
        )


class KimiCodeControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.KIMI_CODE
    model: str
    mode: CallMode

    @property
    def control_mode(self) -> ControlMode:
        return kimi_code_control_mode(self.model)

    @property
    def min_budget_tokens(self) -> int | None:
        if self.control_mode != ControlMode.UNSUPPORTED:
            return 1024
        return None

    @property
    def max_budget_tokens(self) -> int | None:
        if self.control_mode != ControlMode.UNSUPPORTED:
            return 128000
        return None

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        if self.control_mode == ControlMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if self.control_mode == ControlMode.KIMI_CODE_EFFORT_AND_BUDGET:
            return (
                ThinkingLevel.OFF,
                ThinkingLevel.ADAPTIVE,
                ThinkingLevel.BUDGET,
            )
        raise ValueError(
            f"unexpected control mode for provider={self.provider!r} "
            f"model={self.model!r}: {self.control_mode!r}"
        )

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        if ThinkingLevel.OFF in self.supported_thinking_levels:
            return ThinkingLevel.OFF
        return ThinkingLevel.NA

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return supported_effort_levels_for_kimi_code(self.model)

    @property
    def default_effort(self) -> EffortSpec:
        if self.supported_effort_levels:
            return self.supported_effort_levels[0]
        return EffortSpec.NA

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
            max_tokens=16384,
            max_tokens_required=True,
            effort=self.default_effort,
            reasoning=self.default_reasoning,
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
        if sampling is not None and not sampling.is_empty():
            raise ValueError(
                f"sampling is not supported for provider={self.provider!r}"
            )
        return None

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
        _validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        validate_reasoning_for_kimi_code(
            model=request.model, reasoning=request.reasoning
        )
        if request.has_sampling_controls:
            raise ValueError(
                f"sampling is not supported for provider={self.provider!r}"
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


class KimiCodeControlMapping(BaseProviderControlMapping):
    thinking: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> KimiCodeControlMapping:
        if config is None:
            return cls()
        match config:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                budget_tokens=None,
                display=None,
            ):
                return cls()
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.OFF,
                budget_tokens=None,
                display=None,
            ):
                return cls(thinking={"type": "disabled"})
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=None,
                display=None,
            ):
                return cls(thinking={"type": "adaptive"})
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=budget_tokens,
                display=None,
            ):
                tokens = require_budget_tokens(
                    budget_tokens, label=ProviderName.KIMI_CODE, min_value=1
                )
                return cls(
                    thinking={"type": "enabled", "budget_tokens": tokens}
                )
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.KIMI_CODE, config
                    )
                )

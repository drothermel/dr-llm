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
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class AnthropicModelFamily(StrEnum):
    CLAUDE_HAIKU_45 = "claude-haiku-4-5"
    CLAUDE_HAIKU_45_20251001 = "claude-haiku-4-5-20251001"
    CLAUDE_OPUS_46 = "claude-opus-4-6"
    CLAUDE_OPUS_46_20250514 = "claude-opus-4-6-20250514"
    CLAUDE_OPUS_45 = "claude-opus-4-5"
    CLAUDE_OPUS_45_20251101 = "claude-opus-4-5-20251101"
    CLAUDE_OPUS_41 = "claude-opus-4-1"
    CLAUDE_OPUS_41_20250805 = "claude-opus-4-1-20250805"
    CLAUDE_OPUS_4 = "claude-opus-4-"
    CLAUDE_OPUS_4_20250514 = "claude-opus-4-20250514"
    CLAUDE_SONNET_46 = "claude-sonnet-4-6"
    CLAUDE_SONNET_46_20250514 = "claude-sonnet-4-6-20250514"
    CLAUDE_SONNET_45 = "claude-sonnet-4-5"
    CLAUDE_SONNET_45_20250929 = "claude-sonnet-4-5-20250929"
    CLAUDE_SONNET_4 = "claude-sonnet-4-"
    CLAUDE_SONNET_4_20250514 = "claude-sonnet-4-20250514"
    CLAUDE_37_SONNET = "claude-3-7-sonnet"
    CLAUDE_37_SONNET_20250219 = "claude-3-7-sonnet-20250219"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


ANTHROPIC_BUDGET_CAPABILITY_FAMILIES = (
    AnthropicModelFamily.CLAUDE_OPUS_41,
    AnthropicModelFamily.CLAUDE_OPUS_4,
    AnthropicModelFamily.CLAUDE_SONNET_45,
    AnthropicModelFamily.CLAUDE_SONNET_4,
    AnthropicModelFamily.CLAUDE_37_SONNET,
    AnthropicModelFamily.CLAUDE_HAIKU_45,
)
ANTHROPIC_BUDGET_THINKING_SUPPORTED = (
    AnthropicModelFamily.CLAUDE_HAIKU_45_20251001,
    AnthropicModelFamily.CLAUDE_OPUS_45,
    AnthropicModelFamily.CLAUDE_OPUS_45_20251101,
    AnthropicModelFamily.CLAUDE_SONNET_45,
    AnthropicModelFamily.CLAUDE_SONNET_45_20250929,
    AnthropicModelFamily.CLAUDE_OPUS_41,
    AnthropicModelFamily.CLAUDE_OPUS_41_20250805,
    AnthropicModelFamily.CLAUDE_OPUS_4_20250514,
    AnthropicModelFamily.CLAUDE_SONNET_4_20250514,
    AnthropicModelFamily.CLAUDE_37_SONNET,
    AnthropicModelFamily.CLAUDE_37_SONNET_20250219,
)
ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED = (
    AnthropicModelFamily.CLAUDE_SONNET_46,
    AnthropicModelFamily.CLAUDE_SONNET_46_20250514,
    AnthropicModelFamily.CLAUDE_OPUS_46,
    AnthropicModelFamily.CLAUDE_OPUS_46_20250514,
)

ANTHROPIC_BUDGET_MIN_TOKENS = 1024
ANTHROPIC_BUDGET_MAX_TOKENS = 128000


def anthropic_control_mode(model: str) -> ControlMode:
    if AnthropicModelFamily.CLAUDE_OPUS_46.in_family(model):
        return ControlMode.ANTHROPIC_EFFORT
    if AnthropicModelFamily.CLAUDE_SONNET_46.in_family(model):
        return ControlMode.ANTHROPIC_EFFORT
    if AnthropicModelFamily.CLAUDE_OPUS_45.in_family(model):
        return ControlMode.ANTHROPIC_EFFORT_AND_BUDGET
    if any(
        family.in_family(model)
        for family in ANTHROPIC_BUDGET_CAPABILITY_FAMILIES
    ):
        return ControlMode.ANTHROPIC_BUDGET
    return ControlMode.UNSUPPORTED


def anthropic_supports_adaptive_thinking(model: str) -> bool:
    return anthropic_control_mode(model) == ControlMode.ANTHROPIC_EFFORT


def anthropic_supports_budget_thinking(model: str) -> bool:
    return anthropic_control_mode(model) in {
        ControlMode.ANTHROPIC_BUDGET,
        ControlMode.ANTHROPIC_EFFORT_AND_BUDGET,
    }


def anthropic_supports_effort(model: str) -> bool:
    return anthropic_control_mode(model) in {
        ControlMode.ANTHROPIC_EFFORT,
        ControlMode.ANTHROPIC_EFFORT_AND_BUDGET,
    }


def supported_effort_levels_for_anthropic(
    model: str,
) -> tuple[EffortSpec, ...]:
    if not anthropic_supports_effort(model):
        return ()
    levels = [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
    if AnthropicModelFamily.CLAUDE_OPUS_46.in_family(model):
        levels.append(EffortSpec.MAX)
    return tuple(levels)


def validate_anthropic_budget_for_provider(
    *,
    provider: str,
    model: str,
    budget_tokens: int | None,
    min_budget_tokens: int | None,
    max_budget_tokens: int | None,
) -> None:
    if budget_tokens is None:
        raise ValueError(
            f"{provider} budget thinking requires budget_tokens when "
            "thinking_level is 'budget'"
        )
    unsupported_anthropic_model = (
        provider == ProviderName.ANTHROPIC
        and not anthropic_supports_budget_thinking(model)
    )
    if (
        unsupported_anthropic_model
        or min_budget_tokens is None
        or max_budget_tokens is None
    ):
        raise ValueError(
            f"{provider} budget thinking is not supported for model={model!r}"
        )
    validate_budget_range(
        provider=provider,
        model=model,
        label=f"{provider} budget_tokens",
        tokens=budget_tokens,
        min_budget_tokens=min_budget_tokens,
        max_budget_tokens=max_budget_tokens,
    )


def validate_reasoning_for_anthropic(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    controls = AnthropicControls(model=model, mode=CallMode.api)
    dispatch_reasoning_validation(
        provider=ProviderName.ANTHROPIC,
        model=model,
        reasoning=reasoning,
        native_spec_type=AnthropicReasoning,
        requires_reasoning=not is_control_unsupported(controls.control_mode),
        validate_native=lambda spec: _validate_anthropic_reasoning_shape(
            model=model, reasoning=spec, controls=controls
        ),
        validate_top_budget=lambda budget: (
            validate_anthropic_budget_for_provider(
                provider=ProviderName.ANTHROPIC,
                model=model,
                budget_tokens=budget.tokens,
                min_budget_tokens=controls.min_budget_tokens,
                max_budget_tokens=controls.max_budget_tokens,
            )
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
        _validate_anthropic_na(model=model)
        return
    if thinking_level == ThinkingLevel.OFF:
        return
    if thinking_level == ThinkingLevel.ADAPTIVE:
        _validate_anthropic_adaptive(model=model)
        return
    if thinking_level == ThinkingLevel.BUDGET:
        validate_anthropic_budget_for_provider(
            provider=ProviderName.ANTHROPIC,
            model=model,
            budget_tokens=reasoning.budget_tokens,
            min_budget_tokens=controls.min_budget_tokens,
            max_budget_tokens=controls.max_budget_tokens,
        )
        return
    raise ValueError(
        f"Unsupported anthropic thinking level {thinking_level!r} for model={model!r}"
    )


def _validate_anthropic_na(*, model: str) -> None:
    if not is_control_unsupported(anthropic_control_mode(model)):
        raise ValueError(
            f"thinking_level='na' is not supported for provider='{ProviderName.ANTHROPIC}' model={model!r}"
        )


def _validate_anthropic_adaptive(*, model: str) -> None:
    if not anthropic_supports_adaptive_thinking(model):
        raise ValueError(
            f"anthropic adaptive thinking is not supported for model={model!r}"
        )


class AnthropicControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.ANTHROPIC
    model: str
    mode: CallMode

    @property
    def control_mode(self) -> ControlMode:
        return anthropic_control_mode(self.model)

    @property
    def min_budget_tokens(self) -> int | None:
        if self.control_mode in {
            ControlMode.ANTHROPIC_BUDGET,
            ControlMode.ANTHROPIC_EFFORT_AND_BUDGET,
        }:
            return ANTHROPIC_BUDGET_MIN_TOKENS
        return None

    @property
    def max_budget_tokens(self) -> int | None:
        if self.control_mode in {
            ControlMode.ANTHROPIC_BUDGET,
            ControlMode.ANTHROPIC_EFFORT_AND_BUDGET,
        }:
            return ANTHROPIC_BUDGET_MAX_TOKENS
        return None

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        if self.control_mode == ControlMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if self.control_mode == ControlMode.ANTHROPIC_BUDGET:
            return (ThinkingLevel.OFF, ThinkingLevel.BUDGET)
        if self.control_mode == ControlMode.ANTHROPIC_EFFORT:
            return self._supported_effort_thinking_levels()
        if self.control_mode == ControlMode.ANTHROPIC_EFFORT_AND_BUDGET:
            return (
                *self._supported_effort_thinking_levels(),
                ThinkingLevel.BUDGET,
            )
        raise ValueError(
            f"unexpected control mode for provider={self.provider!r} "
            f"model={self.model!r}: {self.control_mode!r}"
        )

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        levels = self.supported_thinking_levels
        for level in (
            ThinkingLevel.OFF,
            ThinkingLevel.ADAPTIVE,
            ThinkingLevel.BUDGET,
        ):
            if level in levels:
                return level
        return ThinkingLevel.NA

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return supported_effort_levels_for_anthropic(self.model)

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
            max_tokens=4096,
            max_tokens_required=True,
            effort=self.default_effort,
            reasoning=self.default_reasoning,
            sampling_supported=True,
            sampling=SamplingControls(temperature=1.0, top_p=0.95),
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
        return SamplingControls(temperature=1.0, top_p=0.95)

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
        validate_reasoning_for_anthropic(
            model=request.model, reasoning=request.reasoning
        )
        return []

    def _supported_effort_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        if anthropic_supports_adaptive_thinking(self.model):
            return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
        return (ThinkingLevel.OFF,)

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
                    thinking={"type": "enabled", "budget_tokens": tokens}
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
                    thinking = {"type": "enabled", "budget_tokens": tokens}
                elif thinking_level == ThinkingLevel.ADAPTIVE:
                    thinking = {"type": "adaptive"}
                if display is not None:
                    thinking["display"] = display
                return cls(thinking=thinking)
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.ANTHROPIC, config
                    )
                )


ANTHROPIC_THINKING_MIN_BUDGET_TOKENS = ANTHROPIC_BUDGET_MIN_TOKENS
ANTHROPIC_THINKING_MAX_BUDGET_TOKENS = ANTHROPIC_BUDGET_MAX_TOKENS

__all__ = [
    "ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED",
    "ANTHROPIC_BUDGET_THINKING_SUPPORTED",
    "ANTHROPIC_THINKING_MAX_BUDGET_TOKENS",
    "ANTHROPIC_THINKING_MIN_BUDGET_TOKENS",
]

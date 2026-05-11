from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.core.authoring import (
    build_provider_config,
    validate_budget_range,
)
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.anthropic.controls import (
    ANTHROPIC_BUDGET_MAX_TOKENS,
    ANTHROPIC_BUDGET_MIN_TOKENS,
    anthropic_control_mode,
    supported_effort_levels_for_anthropic,
)

type _AnthropicBudgetThinkingLevel = Literal[
    ThinkingLevel.OFF,
    ThinkingLevel.BUDGET,
]
type _AnthropicEffortThinkingLevel = Literal[
    ThinkingLevel.OFF,
    ThinkingLevel.ADAPTIVE,
]
type _AnthropicEffort = Literal[
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
    EffortSpec.MAX,
]


class _AnthropicBaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.ANTHROPIC] = ProviderName.ANTHROPIC
    model: str
    max_tokens: int | None = None
    sampling: SamplingControls | None = None

    def _expected_control_mode(self) -> ControlMode:
        raise NotImplementedError

    @model_validator(mode="after")
    def _validate_model_family(self) -> _AnthropicBaseConfig:
        mode = anthropic_control_mode(self.model)
        expected_mode = self._expected_control_mode()
        if mode != expected_mode:
            raise ValueError(
                f"{type(self).__name__} requires "
                f"provider={self.provider!r} control mode "
                f"{expected_mode.value!r}; got {mode.value!r} "
                f"for model={self.model!r}"
            )
        return self

    def _effort(self) -> EffortSpec | None:
        return None

    def _thinking_level(self) -> ThinkingLevel | None:
        return None

    def _budget_tokens(self) -> int | None:
        return None

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self._effort(),
            thinking_level=self._thinking_level(),
            budget_tokens=self._budget_tokens(),
            sampling=self.sampling,
            registry=registry,
        )


class AnthropicLegacyConfig(_AnthropicBaseConfig):
    def _expected_control_mode(self) -> ControlMode:
        return ControlMode.UNSUPPORTED


class AnthropicBudgetConfig(_AnthropicBaseConfig):
    thinking_level: _AnthropicBudgetThinkingLevel | None = None
    budget_tokens: int | None = None

    def _expected_control_mode(self) -> ControlMode:
        return ControlMode.ANTHROPIC_BUDGET

    @model_validator(mode="after")
    def _validate_budget(self) -> AnthropicBudgetConfig:
        _validate_anthropic_budget(
            config_name=type(self).__name__,
            provider=self.provider,
            model=self.model,
            thinking_level=self.thinking_level,
            budget_tokens=self.budget_tokens,
        )
        return self

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level

    def _budget_tokens(self) -> int | None:
        return self.budget_tokens


class AnthropicEffortConfig(_AnthropicBaseConfig):
    effort: _AnthropicEffort | None = None
    thinking_level: _AnthropicEffortThinkingLevel | None = None

    def _expected_control_mode(self) -> ControlMode:
        return ControlMode.ANTHROPIC_EFFORT

    @model_validator(mode="after")
    def _validate_effort(self) -> AnthropicEffortConfig:
        _validate_anthropic_effort(
            config_name=type(self).__name__,
            provider=self.provider,
            model=self.model,
            effort=self.effort,
        )
        return self

    def _effort(self) -> EffortSpec | None:
        return self.effort

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class AnthropicEffortAndBudgetConfig(_AnthropicBaseConfig):
    effort: _AnthropicEffort | None = None
    thinking_level: _AnthropicBudgetThinkingLevel | None = None
    budget_tokens: int | None = None

    def _expected_control_mode(self) -> ControlMode:
        return ControlMode.ANTHROPIC_EFFORT_AND_BUDGET

    @model_validator(mode="after")
    def _validate_controls(self) -> AnthropicEffortAndBudgetConfig:
        _validate_anthropic_effort(
            config_name=type(self).__name__,
            provider=self.provider,
            model=self.model,
            effort=self.effort,
        )
        _validate_anthropic_budget(
            config_name=type(self).__name__,
            provider=self.provider,
            model=self.model,
            thinking_level=self.thinking_level,
            budget_tokens=self.budget_tokens,
        )
        return self

    def _effort(self) -> EffortSpec | None:
        return self.effort

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level

    def _budget_tokens(self) -> int | None:
        return self.budget_tokens


def _validate_anthropic_effort(
    *,
    config_name: str,
    provider: str,
    model: str,
    effort: EffortSpec | None,
) -> None:
    if effort is None:
        return
    allowed = supported_effort_levels_for_anthropic(model)
    if effort in allowed:
        return
    allowed_values = ", ".join(level.value for level in allowed)
    raise ValueError(
        f"{config_name} effort={effort.value!r} is not supported for "
        f"provider={provider!r} model={model!r}; allowed levels: "
        f"{allowed_values}"
    )


def _validate_anthropic_budget(
    *,
    config_name: str,
    provider: str,
    model: str,
    thinking_level: ThinkingLevel | None,
    budget_tokens: int | None,
) -> None:
    if thinking_level == ThinkingLevel.BUDGET and budget_tokens is None:
        raise ValueError(
            f"{config_name} budget thinking requires budget_tokens"
        )
    if thinking_level != ThinkingLevel.BUDGET and budget_tokens is not None:
        raise ValueError("budget_tokens requires thinking_level='budget'")
    if budget_tokens is None:
        return
    validate_budget_range(
        provider=provider,
        model=model,
        budget_tokens=budget_tokens,
        min_tokens=ANTHROPIC_BUDGET_MIN_TOKENS,
        max_tokens=ANTHROPIC_BUDGET_MAX_TOKENS,
    )

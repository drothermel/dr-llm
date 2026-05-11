from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ReasoningMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.effort import FULL_EFFORT
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    BaseProviderReasoningConfig,
    ReasoningBudget,
    ReasoningSpec,
    unsupported_reasoning_kind_message,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


class MiniMaxModelFamily(StrEnum):
    MINIMAX = "MiniMax-"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


MINIMAX_SUPPORTED_MODEL_FAMILIES = (MiniMaxModelFamily.MINIMAX,)


def minimax_reasoning_mode(model: str) -> ReasoningMode:
    if any(
        family.in_family(model) for family in MINIMAX_SUPPORTED_MODEL_FAMILIES
    ):
        return ReasoningMode.MINIMAX_EFFORT
    return ReasoningMode.UNSUPPORTED


def supported_effort_levels_for_minimax(model: str) -> tuple[EffortSpec, ...]:
    if minimax_reasoning_mode(model) == ReasoningMode.UNSUPPORTED:
        return ()
    return FULL_EFFORT


def validate_reasoning_for_minimax(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if reasoning is None:
        raise ValueError(
            f"reasoning is required for provider='{ProviderName.MINIMAX}' model={model!r}"
        )
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            "minimax requires anthropic reasoning configs; "
            "use AnthropicReasoning(thinking_level='na')"
        )
    if not isinstance(reasoning, AnthropicReasoning):
        raise ValueError(
            f"minimax reasoning is not supported for kind={reasoning.kind!r}"
        )
    if reasoning.display is not None:
        raise ValueError("minimax does not support anthropic display controls")
    if reasoning.budget_tokens is not None:
        raise ValueError("minimax does not support anthropic budget_tokens")
    if reasoning.thinking_level != ThinkingLevel.NA:
        raise ValueError(
            "minimax does not support explicit anthropic thinking; use thinking_level='na'"
        )


class MiniMaxControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.MINIMAX
    model: str
    mode: CallMode

    @property
    def reasoning_mode(self) -> ReasoningMode:
        return minimax_reasoning_mode(self.model)

    @property
    def supports_reasoning(self) -> bool:
        return self.reasoning_mode != ReasoningMode.UNSUPPORTED

    @property
    def supported_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        return (ThinkingLevel.NA,)

    @property
    def default_thinking_level(self) -> ThinkingLevel:
        return ThinkingLevel.NA

    @property
    def supported_effort_levels(self) -> tuple[EffortSpec, ...]:
        return supported_effort_levels_for_minimax(self.model)

    @property
    def default_effort(self) -> EffortSpec:
        if self.supported_effort_levels:
            return self.supported_effort_levels[0]
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
            sampling=SamplingControls(temperature=1.0, top_p=0.95),
        )

    def resolve_reasoning(
        self,
        *,
        reasoning: ReasoningSpec | None,
        thinking_level: ThinkingLevel | None,
        budget_tokens: int | None,
    ) -> ReasoningSpec | None:
        del budget_tokens
        if reasoning is not None:
            return reasoning
        if thinking_level is not None:
            return self.reasoning_for_thinking_level(
                thinking_level=thinking_level
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
        del budget_tokens
        if thinking_level == ThinkingLevel.NA:
            return AnthropicReasoning(thinking_level=ThinkingLevel.NA)
        raise ValueError(
            f"{self.provider} does not support thinking_level={thinking_level!r}"
        )

    def validate_request(self, request: LlmRequest) -> list:
        _validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        validate_reasoning_for_minimax(
            model=request.model, reasoning=request.reasoning
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


class MiniMaxReasoningConfig(BaseProviderReasoningConfig):
    thinking: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> MiniMaxReasoningConfig:
        if config is None:
            raise ProviderSemanticError(
                f"{ProviderName.MINIMAX} requires explicit AnthropicReasoning(thinking_level='na')"
            )
        match config:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                budget_tokens=None,
                display=None,
            ):
                return cls()
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.MINIMAX, config
                    )
                )

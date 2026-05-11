from __future__ import annotations

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
from dr_llm.llm.providers.concepts.effort import validate_effort
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    BaseProviderControlMapping,
    ReasoningBudget,
    ReasoningSpec,
    unsupported_reasoning_kind_message,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.providers.impls.minimax.families import (
    MiniMaxFamilies,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


MINIMAX_DEFAULT_SAMPLING = SamplingControls(temperature=1.0, top_p=0.95)


def _validate_reasoning_for_minimax(
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
    families: MiniMaxFamilies = Field(
        default_factory=MiniMaxFamilies, exclude=True
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
            sampling=MINIMAX_DEFAULT_SAMPLING,
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
        return MINIMAX_DEFAULT_SAMPLING

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
        validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        _validate_reasoning_for_minimax(
            model=request.model, reasoning=request.reasoning
        )
        return []


class MiniMaxControlMapping(BaseProviderControlMapping):
    thinking: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> MiniMaxControlMapping:
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

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import EffortSpec, ControlMode
from dr_llm.llm.names import ProviderName, ThinkingLevel
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
from dr_llm.llm.providers.impls.claude_code.families import (
    CLAUDE_CODE_FAMILIES,
    ClaudeCodeFamilies,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import CallMode


def _validate_reasoning_for_claude_code(
    *,
    model: str,
    reasoning: ReasoningSpec | None,
    families: ClaudeCodeFamilies | None = None,
) -> None:
    families = families or CLAUDE_CODE_FAMILIES
    if reasoning is None:
        if families.supports_adaptive_thinking(model):
            msg = (
                "reasoning is required for "
                f"provider='{ProviderName.CLAUDE_CODE}' model={model!r}"
            )
            raise ValueError(msg)
        return
    if isinstance(reasoning, ReasoningBudget):
        msg = f"{ProviderName.CLAUDE_CODE} does not support budget thinking for model={model!r}"
        raise TypeError(msg)
    if not isinstance(reasoning, AnthropicReasoning):
        msg = f"{ProviderName.CLAUDE_CODE} reasoning is not supported for kind={reasoning.kind!r}"
        raise TypeError(msg)
    if reasoning.display is not None:
        msg = (
            f"{ProviderName.CLAUDE_CODE} does not support anthropic display "
            "controls"
        )
        raise ValueError(msg)
    if reasoning.budget_tokens is not None:
        msg = f"{ProviderName.CLAUDE_CODE} does not support budget_tokens"
        raise ValueError(msg)
    if families.supports_adaptive_thinking(model):
        if reasoning.thinking_level != ThinkingLevel.ADAPTIVE:
            msg = f"{ProviderName.CLAUDE_CODE} model {model!r} only supports anthropic thinking_level='adaptive'"
            raise ValueError(msg)
        return
    if reasoning.thinking_level != ThinkingLevel.NA:
        msg = f"{ProviderName.CLAUDE_CODE} model {model!r} does not support explicit anthropic thinking; use thinking_level='na'"
        raise ValueError(msg)


class ClaudeCodeControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ProviderName = ProviderName.CLAUDE_CODE
    model: str
    mode: CallMode
    families: ClaudeCodeFamilies = Field(
        default_factory=ClaudeCodeFamilies, exclude=True
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
        del budget_tokens
        if thinking_level == ThinkingLevel.ADAPTIVE:
            return AnthropicReasoning(thinking_level=ThinkingLevel.ADAPTIVE)
        if thinking_level == ThinkingLevel.NA:
            return None
        raise ValueError(
            f"unsupported {self.provider} thinking level for "
            f"model={self.model!r}: {thinking_level!r}"
        )

    def validate_request(self, request: LlmRequest) -> list:
        validate_effort(
            provider=self.provider,
            model=self.model,
            effort=request.effort,
            supported_effort_levels=self.supported_effort_levels,
        )
        _validate_reasoning_for_claude_code(
            model=request.model,
            reasoning=request.reasoning,
            families=self.families,
        )
        if request.has_sampling_controls:
            raise ValueError(
                f"sampling is not supported for provider={self.provider!r}"
            )
        return []


class ClaudeHeadlessControlMapping(BaseProviderControlMapping):
    cli_args: list[str] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> ClaudeHeadlessControlMapping:
        if config is None:
            return cls()
        match config:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA, display=None
            ):
                return cls()
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=None,
                display=None,
            ):
                return cls()
        raise HeadlessExecutionError(
            unsupported_reasoning_kind_message("claude headless", config)
        )

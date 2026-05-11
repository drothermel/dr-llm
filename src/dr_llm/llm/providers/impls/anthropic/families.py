from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.effort import default_effort
from dr_llm.llm.providers.concepts.model_family import (
    model_matches_any_family,
)


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


ANTHROPIC_THINKING_MIN_BUDGET_TOKENS = 1024
ANTHROPIC_THINKING_MAX_BUDGET_TOKENS = 128000


class AnthropicFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    budget_families: tuple[AnthropicModelFamily, ...] = (
        AnthropicModelFamily.CLAUDE_OPUS_41,
        AnthropicModelFamily.CLAUDE_OPUS_4,
        AnthropicModelFamily.CLAUDE_SONNET_45,
        AnthropicModelFamily.CLAUDE_SONNET_4,
        AnthropicModelFamily.CLAUDE_37_SONNET,
        AnthropicModelFamily.CLAUDE_HAIKU_45,
    )
    effort_families: tuple[AnthropicModelFamily, ...] = (
        AnthropicModelFamily.CLAUDE_OPUS_46,
        AnthropicModelFamily.CLAUDE_SONNET_46,
    )
    effort_and_budget_families: tuple[AnthropicModelFamily, ...] = (
        AnthropicModelFamily.CLAUDE_OPUS_45,
    )
    max_effort_families: tuple[AnthropicModelFamily, ...] = (
        AnthropicModelFamily.CLAUDE_OPUS_46,
    )
    min_budget_tokens: int = ANTHROPIC_THINKING_MIN_BUDGET_TOKENS
    max_budget_tokens: int = ANTHROPIC_THINKING_MAX_BUDGET_TOKENS

    def matches_any(
        self, model: str, families: tuple[AnthropicModelFamily, ...]
    ) -> bool:
        return model_matches_any_family(model, families)

    def control_mode(self, model: str) -> ControlMode:
        if self.matches_any(model, self.effort_families):
            return ControlMode.ANTHROPIC_EFFORT
        if self.matches_any(model, self.effort_and_budget_families):
            return ControlMode.ANTHROPIC_EFFORT_AND_BUDGET
        if self.matches_any(model, self.budget_families):
            return ControlMode.ANTHROPIC_BUDGET
        return ControlMode.UNSUPPORTED

    def supports_adaptive_thinking(self, model: str) -> bool:
        return self.control_mode(model) == ControlMode.ANTHROPIC_EFFORT

    def supports_budget_thinking(self, model: str) -> bool:
        return self.control_mode(model) in {
            ControlMode.ANTHROPIC_BUDGET,
            ControlMode.ANTHROPIC_EFFORT_AND_BUDGET,
        }

    def supports_effort(self, model: str) -> bool:
        return self.control_mode(model) in {
            ControlMode.ANTHROPIC_EFFORT,
            ControlMode.ANTHROPIC_EFFORT_AND_BUDGET,
        }

    def budget_min_for_model(self, model: str) -> int | None:
        if self.supports_budget_thinking(model):
            return self.min_budget_tokens
        return None

    def budget_max_for_model(self, model: str) -> int | None:
        if self.supports_budget_thinking(model):
            return self.max_budget_tokens
        return None

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        control_mode = self.control_mode(model)
        if control_mode == ControlMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if control_mode == ControlMode.ANTHROPIC_BUDGET:
            return (ThinkingLevel.OFF, ThinkingLevel.BUDGET)
        if control_mode == ControlMode.ANTHROPIC_EFFORT:
            return self._supported_effort_thinking_levels(model)
        if control_mode == ControlMode.ANTHROPIC_EFFORT_AND_BUDGET:
            return (
                *self._supported_effort_thinking_levels(model),
                ThinkingLevel.BUDGET,
            )
        raise ValueError(
            f"unexpected control mode for provider={ProviderName.ANTHROPIC!r} "
            f"model={model!r}: {control_mode!r}"
        )

    def default_thinking_level(self, model: str) -> ThinkingLevel:
        levels = self.supported_thinking_levels(model)
        for level in (
            ThinkingLevel.OFF,
            ThinkingLevel.ADAPTIVE,
            ThinkingLevel.BUDGET,
        ):
            if level in levels:
                return level
        return ThinkingLevel.NA

    def supported_effort_levels(self, model: str) -> tuple[EffortSpec, ...]:
        if not self.supports_effort(model):
            return ()
        levels = [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
        if self.matches_any(model, self.max_effort_families):
            levels.append(EffortSpec.MAX)
        return tuple(levels)

    def default_effort(self, model: str) -> EffortSpec:
        return default_effort(self.supported_effort_levels(model))

    def _supported_effort_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if self.supports_adaptive_thinking(model):
            return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
        return (ThinkingLevel.OFF,)


ANTHROPIC_FAMILIES = AnthropicFamilies()

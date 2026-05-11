from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm.names import EffortSpec, ControlMode, ThinkingLevel
from dr_llm.llm.providers.concepts.model_family import (
    model_matches_any_family,
)
from dr_llm.llm.providers.impls.anthropic.families import AnthropicFamilies


class ClaudeCodeModelFamily(StrEnum):
    CLAUDE = "claude-"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


class ClaudeCodeFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    supported: tuple[ClaudeCodeModelFamily, ...] = (
        ClaudeCodeModelFamily.CLAUDE,
    )
    anthropic_families: AnthropicFamilies = Field(
        default_factory=AnthropicFamilies
    )

    def is_supported_model(self, model: str) -> bool:
        return model_matches_any_family(model, self.supported)

    def control_mode(self, model: str) -> ControlMode:
        if self.is_supported_model(model):
            return ControlMode.CLAUDE_CLI_EFFORT
        return ControlMode.UNSUPPORTED

    def supports_adaptive_thinking(self, model: str) -> bool:
        return self.anthropic_families.supports_adaptive_thinking(model)

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if self.supports_adaptive_thinking(model):
            return (ThinkingLevel.ADAPTIVE,)
        return (ThinkingLevel.NA,)

    def default_thinking_level(self, model: str) -> ThinkingLevel:
        return self.supported_thinking_levels(model)[0]

    def supported_effort_levels(self, model: str) -> tuple[EffortSpec, ...]:
        return self.anthropic_families.supported_effort_levels(model)

    def default_effort(self, model: str) -> EffortSpec:
        return self.anthropic_families.default_effort(model)


CLAUDE_CODE_FAMILIES = ClaudeCodeFamilies()

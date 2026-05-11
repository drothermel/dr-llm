from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec, ControlMode, ThinkingLevel
from dr_llm.llm.providers.concepts.effort import FULL_EFFORT, default_effort
from dr_llm.llm.providers.concepts.model_family import (
    model_matches_any_family,
)


class KimiCodeModelFamily(StrEnum):
    KIMI_FOR_CODING = "kimi-for-coding"

    def in_family(self, model: str) -> bool:
        return model == self


KIMI_CODE_THINKING_MIN_BUDGET_TOKENS = 1024
KIMI_CODE_THINKING_MAX_BUDGET_TOKENS = 128000


class KimiCodeFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    supported: tuple[KimiCodeModelFamily, ...] = (
        KimiCodeModelFamily.KIMI_FOR_CODING,
    )
    min_budget_tokens: int = KIMI_CODE_THINKING_MIN_BUDGET_TOKENS
    max_budget_tokens: int = KIMI_CODE_THINKING_MAX_BUDGET_TOKENS

    def is_supported_model(self, model: str) -> bool:
        return model_matches_any_family(model, self.supported)

    def control_mode(self, model: str) -> ControlMode:
        if self.is_supported_model(model):
            return ControlMode.KIMI_CODE_EFFORT_AND_BUDGET
        return ControlMode.UNSUPPORTED

    def supports_budget_thinking(self, model: str) -> bool:
        return self.control_mode(model) != ControlMode.UNSUPPORTED

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
        if self.control_mode(model) == ControlMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        return (
            ThinkingLevel.OFF,
            ThinkingLevel.ADAPTIVE,
            ThinkingLevel.BUDGET,
        )

    def default_thinking_level(self, model: str) -> ThinkingLevel:
        if ThinkingLevel.OFF in self.supported_thinking_levels(model):
            return ThinkingLevel.OFF
        return ThinkingLevel.NA

    def supported_effort_levels(self, model: str) -> tuple[EffortSpec, ...]:
        if self.control_mode(model) == ControlMode.UNSUPPORTED:
            return ()
        return FULL_EFFORT

    def default_effort(self, model: str) -> EffortSpec:
        return default_effort(self.supported_effort_levels(model))


KIMI_CODE_FAMILIES = KimiCodeFamilies()

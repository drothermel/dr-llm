from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec, ControlMode, ThinkingLevel
from dr_llm.llm.providers.concepts.effort import FULL_EFFORT, default_effort
from dr_llm.llm.providers.concepts.model_family import (
    model_matches_any_family,
)


class MiniMaxModelFamily(StrEnum):
    MINIMAX = "MiniMax-"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


class MiniMaxFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    supported: tuple[MiniMaxModelFamily, ...] = (MiniMaxModelFamily.MINIMAX,)

    def is_supported_model(self, model: str) -> bool:
        return model_matches_any_family(model, self.supported)

    def control_mode(self, model: str) -> ControlMode:
        if self.is_supported_model(model):
            return ControlMode.MINIMAX_EFFORT
        return ControlMode.UNSUPPORTED

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        del model
        return (ThinkingLevel.NA,)

    def default_thinking_level(self, model: str) -> ThinkingLevel:
        del model
        return ThinkingLevel.NA

    def supported_effort_levels(self, model: str) -> tuple[EffortSpec, ...]:
        if self.control_mode(model) == ControlMode.UNSUPPORTED:
            return ()
        return FULL_EFFORT

    def default_effort(self, model: str) -> EffortSpec:
        return default_effort(self.supported_effort_levels(model))


MINIMAX_FAMILIES = MiniMaxFamilies()

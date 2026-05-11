from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec, ControlMode, ThinkingLevel
from dr_llm.llm.providers.concepts.model_family import (
    model_matches_any_family,
)


class GlmModelFamily(StrEnum):
    GLM5 = "glm-5"
    GLM47 = "glm-4.7"
    GLM46 = "glm-4.6"
    GLM45 = "glm-4.5"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


class GlmFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    thinking_supported: tuple[GlmModelFamily, ...] = (
        GlmModelFamily.GLM5,
        GlmModelFamily.GLM47,
        GlmModelFamily.GLM46,
        GlmModelFamily.GLM45,
    )

    def supports_thinking(self, model: str) -> bool:
        return model_matches_any_family(model, self.thinking_supported)

    def control_mode(self, model: str) -> ControlMode:
        if self.supports_thinking(model):
            return ControlMode.GLM
        return ControlMode.UNSUPPORTED

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if self.control_mode(model) == ControlMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)

    def default_thinking_level(self, model: str) -> ThinkingLevel:
        if ThinkingLevel.OFF in self.supported_thinking_levels(model):
            return ThinkingLevel.OFF
        return ThinkingLevel.NA

    def supported_effort_levels(self, model: str) -> tuple[EffortSpec, ...]:
        del model
        return ()

    def default_effort(self, model: str) -> EffortSpec:
        del model
        return EffortSpec.NA


GLM_FAMILIES = GlmFamilies()

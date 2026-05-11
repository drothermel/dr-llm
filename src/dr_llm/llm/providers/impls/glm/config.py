from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import ProviderName, ControlMode, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import GlmReasoning
from dr_llm.llm.providers.core.authoring import build_provider_config
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.glm.families import (
    GLM_FAMILIES,
)

type _GlmThinkingLevel = Literal[
    ThinkingLevel.OFF,
    ThinkingLevel.ADAPTIVE,
]


class _GlmBaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.GLM] = ProviderName.GLM
    model: str
    max_tokens: int | None = None
    sampling: SamplingControls | None = None

    def _expected_control_mode(self) -> ControlMode:
        raise NotImplementedError

    @model_validator(mode="after")
    def _validate_model_family(self) -> _GlmBaseConfig:
        mode = GLM_FAMILIES.control_mode(self.model)
        expected_mode = self._expected_control_mode()
        if mode != expected_mode:
            raise ValueError(
                f"{type(self).__name__} requires "
                f"provider={self.provider!r} control mode "
                f"{expected_mode!r}; got {mode!r} "
                f"for model={self.model!r}"
            )
        return self

    def _reasoning(self) -> GlmReasoning | None:
        return None

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            reasoning=self._reasoning(),
            sampling=self.sampling,
            registry=registry,
        )


class GlmLegacyConfig(_GlmBaseConfig):
    def _expected_control_mode(self) -> ControlMode:
        return ControlMode.UNSUPPORTED


class GlmThinkingConfig(_GlmBaseConfig):
    thinking_level: _GlmThinkingLevel | None = None

    def _expected_control_mode(self) -> ControlMode:
        return ControlMode.GLM

    def _reasoning(self) -> GlmReasoning | None:
        if self.thinking_level is None:
            return None
        return GlmReasoning(thinking_level=self.thinking_level)

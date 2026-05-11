from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import EffortSpec, ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import AnthropicReasoning
from dr_llm.llm.providers.core.authoring import build_provider_config
from dr_llm.llm.providers.core.registry import ProviderRegistry


class MiniMaxConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.MINIMAX] = ProviderName.MINIMAX
    model: str
    max_tokens: int | None = None
    effort: EffortSpec | None = None
    reasoning: AnthropicReasoning | None = None
    thinking_level: Literal[ThinkingLevel.NA] | None = None
    sampling: SamplingControls | None = None

    @model_validator(mode="after")
    def _validate_reasoning_choice(self) -> MiniMaxConfig:
        if self.reasoning is not None and self.thinking_level is not None:
            raise ValueError(
                "reasoning and thinking_level are mutually exclusive"
            )
        return self

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
            thinking_level=self.thinking_level,
            sampling=self.sampling,
            registry=registry,
        )

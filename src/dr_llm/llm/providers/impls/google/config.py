from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import GoogleReasoning
from dr_llm.llm.providers.core.authoring import build_provider_config
from dr_llm.llm.providers.core.registry import ProviderRegistry


class GoogleConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.GOOGLE] = ProviderName.GOOGLE
    model: str
    max_tokens: int | None = None
    reasoning: GoogleReasoning | None = None
    thinking_level: Literal[
        ThinkingLevel.NA,
        ThinkingLevel.OFF,
        ThinkingLevel.ADAPTIVE,
        ThinkingLevel.BUDGET,
    ] | None = None
    budget_tokens: int | None = None
    sampling: SamplingControls | None = None

    @model_validator(mode="after")
    def _validate_reasoning_choice(self) -> GoogleConfig:
        if self.reasoning is not None and self.thinking_level is not None:
            raise ValueError(
                "reasoning and thinking_level are mutually exclusive"
            )
        if (
            self.budget_tokens is not None
            and self.thinking_level != ThinkingLevel.BUDGET
        ):
            raise ValueError("budget_tokens requires thinking_level='budget'")
        return self

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
            thinking_level=self.thinking_level,
            budget_tokens=self.budget_tokens,
            sampling=self.sampling,
            registry=registry,
        )

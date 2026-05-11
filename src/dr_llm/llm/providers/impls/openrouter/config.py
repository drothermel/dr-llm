from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    OpenRouterReasoning,
)
from dr_llm.llm.providers.core.authoring import build_provider_config
from dr_llm.llm.providers.core.registry import ProviderRegistry


class OpenRouterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.OPENROUTER] = ProviderName.OPENROUTER
    model: str
    max_tokens: int | None = None
    reasoning: OpenRouterReasoning | OpenAIReasoning | None = None
    sampling: SamplingControls | None = None

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
            sampling=self.sampling,
            registry=registry,
        )

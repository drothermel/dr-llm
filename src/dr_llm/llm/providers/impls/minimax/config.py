from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import EffortSpec, ProviderName
from dr_llm.llm.providers.core.authoring import build_provider_config
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.minimax.capabilities import (
    supported_effort_levels_for_minimax,
)
from dr_llm.llm.providers.impls.minimax.families import MiniMaxModelFamily

type _MiniMaxEffort = Literal[
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
    EffortSpec.MAX,
]


class MiniMaxConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.MINIMAX] = ProviderName.MINIMAX
    model: str
    max_tokens: int | None = None
    effort: _MiniMaxEffort | None = None
    sampling: SamplingControls | None = None

    @model_validator(mode="after")
    def _validate_controls(self) -> MiniMaxConfig:
        if not self.model.startswith(MiniMaxModelFamily.MINIMAX):
            raise ValueError(
                f"MiniMaxConfig only supports provider={self.provider!r} "
                f"model family={MiniMaxModelFamily.MINIMAX.value!r}; "
                f"got model={self.model!r}"
            )
        if self.effort is None:
            return self
        allowed = supported_effort_levels_for_minimax(self.model)
        if self.effort in allowed:
            return self
        allowed_values = ", ".join(level.value for level in allowed)
        raise ValueError(
            f"MiniMaxConfig effort={self.effort.value!r} is not supported "
            f"for provider={self.provider!r} model={self.model!r}; "
            f"allowed levels: {allowed_values}"
        )

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            sampling=self.sampling,
            registry=registry,
        )

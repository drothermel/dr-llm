from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import EffortSpec
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec
from dr_llm.llm.response import CallMode


class ProviderRequestDefaults(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    mode: CallMode
    max_tokens: int | None = None
    max_tokens_required: bool = False
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    sampling_supported: bool = False
    sampling: SamplingControls | None = None

    @model_validator(mode="after")
    def _validate_consistency(self) -> ProviderRequestDefaults:
        if self.max_tokens_required and self.max_tokens is None:
            raise ValueError(
                "max_tokens_required=True requires a max_tokens default"
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        if not self.sampling_supported and self.sampling is not None:
            raise ValueError(
                "sampling default requires sampling_supported=True"
            )
        return self

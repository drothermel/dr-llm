from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

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
    supports_temperature: bool = False
    temperature: float | None = None
    supports_top_p: bool = False
    top_p: float | None = None

    @model_validator(mode="after")
    def _validate_consistency(self) -> ProviderRequestDefaults:
        if self.max_tokens_required and self.max_tokens is None:
            raise ValueError(
                "max_tokens_required=True requires a max_tokens default"
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        if not self.supports_temperature and self.temperature is not None:
            raise ValueError(
                "temperature default requires supports_temperature=True"
            )
        if not self.supports_top_p and self.top_p is not None:
            raise ValueError("top_p default requires supports_top_p=True")
        return self

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec


class ProviderRequestDefaults(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    mode: str
    max_tokens: int | None = None
    max_tokens_required: bool = False
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    supports_temperature: bool = False
    temperature: float | None = None
    supports_top_p: bool = False
    top_p: float | None = None

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
)

from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import EffortSpec, MessageRole, ProviderName
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec
from dr_llm.llm.response import CallMode


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: str


class LlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName
    model: str
    mode: CallMode
    messages: list[Message]
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    sampling: SamplingControls | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_sampling_controls(self) -> bool:
        return self.sampling is not None and not self.sampling.is_empty()

    @property
    def sampling_temperature(self) -> float | None:
        return self.sampling.temperature if self.sampling is not None else None

    @property
    def sampling_top_p(self) -> float | None:
        return self.sampling.top_p if self.sampling is not None else None


LLM_REQUEST_ADAPTER = TypeAdapter(LlmRequest)


def parse_llm_request(payload: object) -> LlmRequest:
    return LLM_REQUEST_ADAPTER.validate_python(payload)

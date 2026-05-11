from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from dr_llm.llm.config import SamplingControls
from dr_llm.llm.names import EffortSpec, ProviderName
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec
from dr_llm.llm.response import CallMode


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
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


LLM_REQUEST_ADAPTER = TypeAdapter(LlmRequest)


def parse_llm_request(payload: object) -> LlmRequest:
    return LLM_REQUEST_ADAPTER.validate_python(payload)

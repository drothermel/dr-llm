from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from dr_llm.llm.names import (
    ApiBackedProviderName,
    EffortSpec,
    HeadlessProviderName,
    KimiCodeProviderName,
    OpenAIProviderName,
    SamplingApiProviderName,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ApiBackedLlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ApiBackedProviderName
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApiLlmRequest(ApiBackedLlmRequest):
    provider: SamplingApiProviderName
    temperature: float | None = 1.0
    top_p: float | None = 0.95


class OpenAILlmRequest(ApiBackedLlmRequest):
    provider: OpenAIProviderName
    temperature: float | None = None
    top_p: float | None = None


class KimiCodeLlmRequest(ApiBackedLlmRequest):
    provider: KimiCodeProviderName


class HeadlessLlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: HeadlessProviderName
    model: str
    messages: list[Message]
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


type LlmRequest = (
    OpenAILlmRequest | ApiLlmRequest | KimiCodeLlmRequest | HeadlessLlmRequest
)
LlmRequestSpec = Annotated[LlmRequest, Field(discriminator="provider")]
LLM_REQUEST_ADAPTER = TypeAdapter(LlmRequestSpec)


def parse_llm_request(payload: object) -> LlmRequest:
    return LLM_REQUEST_ADAPTER.validate_python(payload)

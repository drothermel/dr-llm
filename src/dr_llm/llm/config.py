from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from dr_llm.llm.names import (
    EffortSpec,
    HeadlessProviderName,
    KimiCodeProviderName,
    OpenAIProviderName,
    SamplingApiProviderName,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec
from dr_llm.llm.request import (
    ApiBackedLlmRequest,
    ApiLlmRequest,
    HeadlessLlmRequest,
    KimiCodeLlmRequest,
    OpenAILlmRequest,
    Message,
)


class ApiBackedLlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: (
        OpenAIProviderName | SamplingApiProviderName | KimiCodeProviderName
    )
    model: str
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None

    def to_request(self, messages: list[Message]) -> ApiBackedLlmRequest:
        raise NotImplementedError


class ApiLlmConfig(ApiBackedLlmConfig):
    provider: SamplingApiProviderName
    temperature: float | None = 1.0
    top_p: float | None = 0.95

    def to_request(self, messages: list[Message]) -> ApiLlmRequest:
        return ApiLlmRequest(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
        )


class OpenAILlmConfig(ApiBackedLlmConfig):
    provider: OpenAIProviderName
    temperature: float | None = None
    top_p: float | None = None

    def to_request(self, messages: list[Message]) -> OpenAILlmRequest:
        return OpenAILlmRequest(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
        )


class KimiCodeLlmConfig(ApiBackedLlmConfig):
    provider: KimiCodeProviderName

    def to_request(self, messages: list[Message]) -> KimiCodeLlmRequest:
        return KimiCodeLlmRequest(
            provider=self.provider,
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
        )


class HeadlessLlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: HeadlessProviderName
    model: str
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None

    def to_request(self, messages: list[Message]) -> HeadlessLlmRequest:
        return HeadlessLlmRequest(
            provider=self.provider,
            model=self.model,
            messages=messages,
            effort=self.effort,
            reasoning=self.reasoning,
        )


type LlmConfig = (
    OpenAILlmConfig | ApiLlmConfig | KimiCodeLlmConfig | HeadlessLlmConfig
)
LlmConfigSpec = Annotated[LlmConfig, Field(discriminator="provider")]
LLM_CONFIG_ADAPTER = TypeAdapter(LlmConfigSpec)


def parse_llm_config(payload: object) -> LlmConfig:
    return LLM_CONFIG_ADAPTER.validate_python(payload)

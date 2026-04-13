from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from dr_llm.llm.messages import Message
from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.providers.reasoning import ReasoningSpec
from dr_llm.llm.request import (
    ApiBackedLlmRequest,
    ApiLlmRequest,
    ApiProviderName,
    HeadlessLlmRequest,
    HeadlessProviderName,
    KimiCodeLlmRequest,
    KimiCodeProviderName,
    OpenAILlmRequest,
    OpenAIProviderName,
    validate_llm_constraints,
)
from dr_llm.llm.providers.openai_compat.thinking import (
    validate_openai_sampling_controls,
)


class ApiBackedLlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: OpenAIProviderName | ApiProviderName | KimiCodeProviderName
    model: str
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None

    @model_validator(mode="after")
    def _validate_generation_params(self) -> ApiBackedLlmConfig:
        validate_llm_constraints(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
        )
        return self

    def to_request(self, messages: list[Message]) -> ApiBackedLlmRequest:
        raise NotImplementedError


class ApiLlmConfig(ApiBackedLlmConfig):
    provider: ApiProviderName
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

    @model_validator(mode="after")
    def _validate_openai_sampling_controls(self) -> OpenAILlmConfig:
        validate_openai_sampling_controls(
            model=self.model,
            reasoning=self.reasoning,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return self

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

    @model_validator(mode="after")
    def _validate_generation_params(self) -> HeadlessLlmConfig:
        validate_llm_constraints(
            provider=self.provider,
            model=self.model,
            max_tokens=None,
            effort=self.effort,
            reasoning=self.reasoning,
        )
        return self

    def to_request(self, messages: list[Message]) -> HeadlessLlmRequest:
        return HeadlessLlmRequest(
            provider=self.provider,
            model=self.model,
            messages=messages,
            effort=self.effort,
            reasoning=self.reasoning,
        )


type LlmConfig = OpenAILlmConfig | ApiLlmConfig | KimiCodeLlmConfig | HeadlessLlmConfig
LlmConfigSpec = Annotated[LlmConfig, Field(discriminator="provider")]
LLM_CONFIG_ADAPTER = TypeAdapter(LlmConfigSpec)


def parse_llm_config(payload: object) -> LlmConfig:
    return LLM_CONFIG_ADAPTER.validate_python(payload)

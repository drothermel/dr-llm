from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from dr_llm.llm.names import (
    ApiBackedProviderName,
    EffortSpec,
    HeadlessProviderName,
    KimiCodeProviderName,
    OpenAIProviderName,
    ProviderName,
    SamplingApiProviderName,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec
from dr_llm.llm.providers.effort import validate_effort
from dr_llm.llm.providers.openai_compat.thinking import (
    validate_openai_sampling_controls,
)
from dr_llm.llm.providers.reasoning_validation import validate_reasoning


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


def validate_max_tokens(*, provider: str, max_tokens: int | None) -> None:
    if (
        provider in {ProviderName.ANTHROPIC, ProviderName.KIMI_CODE}
        and max_tokens is None
    ):
        raise ValueError(f"max_tokens is required for provider={provider!r}")


def validate_llm_constraints(
    *,
    provider: str,
    model: str,
    max_tokens: int | None,
    effort: EffortSpec,
    reasoning: ReasoningSpec | None,
) -> None:
    validate_max_tokens(provider=provider, max_tokens=max_tokens)
    validate_effort(provider=provider, model=model, effort=effort)
    validate_reasoning(provider=provider, model=model, reasoning=reasoning)


class ApiBackedLlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ApiBackedProviderName
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_request_constraints(self) -> ApiBackedLlmRequest:
        validate_llm_constraints(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
        )
        return self


class ApiLlmRequest(ApiBackedLlmRequest):
    provider: SamplingApiProviderName
    temperature: float | None = 1.0
    top_p: float | None = 0.95


class OpenAILlmRequest(ApiBackedLlmRequest):
    provider: OpenAIProviderName
    temperature: float | None = None
    top_p: float | None = None

    @model_validator(mode="after")
    def _validate_openai_sampling_controls(self) -> OpenAILlmRequest:
        validate_openai_sampling_controls(
            model=self.model,
            reasoning=self.reasoning,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return self


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

    @model_validator(mode="after")
    def _validate_request_constraints(self) -> HeadlessLlmRequest:
        validate_llm_constraints(
            provider=self.provider,
            model=self.model,
            max_tokens=None,
            effort=self.effort,
            reasoning=self.reasoning,
        )
        return self


type LlmRequest = (
    OpenAILlmRequest | ApiLlmRequest | KimiCodeLlmRequest | HeadlessLlmRequest
)
LlmRequestSpec = Annotated[LlmRequest, Field(discriminator="provider")]
LLM_REQUEST_ADAPTER = TypeAdapter(LlmRequestSpec)


def parse_llm_request(payload: object) -> LlmRequest:
    return LLM_REQUEST_ADAPTER.validate_python(payload)

from __future__ import annotations

from typing import Annotated, Any

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
    LlmRequest,
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


class ApiLlmConfig(ApiBackedLlmConfig):
    provider: SamplingApiProviderName
    temperature: float | None = 1.0
    top_p: float | None = 0.95


class OpenAILlmConfig(ApiBackedLlmConfig):
    provider: OpenAIProviderName
    temperature: float | None = None
    top_p: float | None = None


class KimiCodeLlmConfig(ApiBackedLlmConfig):
    provider: KimiCodeProviderName


class HeadlessLlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: HeadlessProviderName
    model: str
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None


type LlmConfig = (
    OpenAILlmConfig | ApiLlmConfig | KimiCodeLlmConfig | HeadlessLlmConfig
)
LlmConfigSpec = Annotated[LlmConfig, Field(discriminator="provider")]
LLM_CONFIG_ADAPTER = TypeAdapter(LlmConfigSpec)


def parse_llm_config(payload: object) -> LlmConfig:
    return LLM_CONFIG_ADAPTER.validate_python(payload)


def build_request_from_config(
    orchestrator: Any,
    config: LlmConfig,
    messages: list[Message],
    *,
    metadata: dict[str, Any] | None = None,
) -> LlmRequest:
    if orchestrator.name != config.provider:
        raise ValueError(
            f"config provider {config.provider!r} does not match "
            f"orchestrator provider {orchestrator.name!r}"
        )
    return orchestrator.build_request(
        model=config.model,
        messages=messages,
        max_tokens=getattr(config, "max_tokens", None),
        effort=config.effort,
        reasoning=config.reasoning,
        temperature=getattr(config, "temperature", None),
        top_p=getattr(config, "top_p", None),
        metadata=metadata,
    )

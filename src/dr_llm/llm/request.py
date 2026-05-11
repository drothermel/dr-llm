from __future__ import annotations

from collections.abc import Callable
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
from dr_llm.llm.providers.anthropic.effort import (
    supported_effort_levels_for_anthropic,
)
from dr_llm.llm.providers.anthropic.reasoning import (
    validate_reasoning_for_anthropic,
)
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec
from dr_llm.llm.providers.google.reasoning import validate_reasoning_for_google
from dr_llm.llm.providers.headless.claude.capabilities import (
    supported_effort_levels_for_claude_code,
)
from dr_llm.llm.providers.headless.claude.reasoning import (
    validate_reasoning_for_claude_code,
)
from dr_llm.llm.providers.headless.codex.reasoning import (
    validate_reasoning_for_codex,
)
from dr_llm.llm.providers.kimi_code.capabilities import (
    supported_effort_levels_for_kimi_code,
)
from dr_llm.llm.providers.kimi_code.reasoning import (
    validate_reasoning_for_kimi_code,
)
from dr_llm.llm.providers.minimax.capabilities import (
    supported_effort_levels_for_minimax,
)
from dr_llm.llm.providers.minimax.reasoning import (
    validate_reasoning_for_minimax,
)
from dr_llm.llm.providers.openai_compat.reasoning import (
    validate_reasoning_for_glm,
    validate_reasoning_for_openai,
    validate_reasoning_for_openrouter,
)
from dr_llm.llm.providers.openai_compat.thinking import (
    validate_openai_sampling_controls,
)

_EFFORT_RESOLVERS: dict[str, Callable[[str], tuple[EffortSpec, ...]]] = {
    ProviderName.ANTHROPIC: supported_effort_levels_for_anthropic,
    ProviderName.CLAUDE_CODE: supported_effort_levels_for_claude_code,
    ProviderName.KIMI_CODE: supported_effort_levels_for_kimi_code,
    ProviderName.MINIMAX: supported_effort_levels_for_minimax,
}

_REASONING_VALIDATORS: dict[str, Callable[..., None]] = {
    ProviderName.ANTHROPIC: validate_reasoning_for_anthropic,
    ProviderName.CLAUDE_CODE: validate_reasoning_for_claude_code,
    ProviderName.CODEX: validate_reasoning_for_codex,
    ProviderName.GLM: validate_reasoning_for_glm,
    ProviderName.GOOGLE: validate_reasoning_for_google,
    ProviderName.KIMI_CODE: validate_reasoning_for_kimi_code,
    ProviderName.MINIMAX: validate_reasoning_for_minimax,
    ProviderName.OPENAI: validate_reasoning_for_openai,
    ProviderName.OPENROUTER: validate_reasoning_for_openrouter,
}


def validate_effort(*, provider: str, model: str, effort: EffortSpec) -> None:
    resolver = _EFFORT_RESOLVERS.get(provider)
    allowed_levels = resolver(model) if resolver else ()
    if not allowed_levels:
        if effort != EffortSpec.NA:
            raise ValueError(
                f"effort is not supported for provider={provider!r} model={model!r}"
            )
        return
    if effort == EffortSpec.NA:
        raise ValueError(
            f"effort is required for provider={provider!r} model={model!r}"
        )
    if effort not in allowed_levels:
        allowed = ", ".join(level.value for level in allowed_levels)
        raise ValueError(
            f"effort={effort.value!r} is not supported for provider={provider!r} "
            f"model={model!r}; allowed levels: {allowed}"
        )


def validate_reasoning(
    *, provider: str, model: str, reasoning: ReasoningSpec | None
) -> None:
    validator = _REASONING_VALIDATORS.get(provider)
    if validator is None:
        if reasoning is not None:
            raise ValueError(
                f"reasoning is not supported for provider={provider!r}"
            )
        return
    validator(model=model, reasoning=reasoning)


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

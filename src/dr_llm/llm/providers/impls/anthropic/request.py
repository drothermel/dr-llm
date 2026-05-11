from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import EffortSpec
from dr_llm.llm.providers.impls.anthropic.config import AnthropicConfig
from dr_llm.llm.providers.impls.anthropic.reasoning import (
    AnthropicReasoningConfig,
)
from dr_llm.llm.providers.transports.api_config import resolve_api_key
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.request import ApiBackedLlmRequest, Message


class AnthropicReasoningPayload(Protocol):
    thinking: dict[str, Any]
    warnings: list[ReasoningWarning]


class _AnthropicRequestTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class _AnthropicRequestMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[_AnthropicRequestTextBlock] = Field(default_factory=list)


class AnthropicRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(exclude=True)
    model: str
    messages: list[_AnthropicRequestMessage]
    max_tokens: int | None = None
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    base_url: str = Field(exclude=True)
    api_key: str = Field(exclude=True, repr=False)
    anthropic_version: str = Field(exclude=True)
    warnings: list[ReasoningWarning] = Field(
        default_factory=list, exclude=True
    )

    @classmethod
    def from_llm_request(
        cls,
        request: ApiBackedLlmRequest,
        config: AnthropicConfig,
        *,
        reasoning_mapping: AnthropicReasoningPayload | None = None,
        require_max_tokens: bool = True,
    ) -> AnthropicRequest:
        if require_max_tokens and request.max_tokens is None:
            raise cls._missing_max_tokens_error(request.provider)
        resolved_reasoning_mapping = (
            reasoning_mapping
            or AnthropicReasoningConfig.from_base(request.reasoning)
        )
        output_config = (
            {"effort": request.effort}
            if request.effort != EffortSpec.NA
            else None
        )
        system = "\n".join(
            message.content
            for message in request.messages
            if message.role == "system"
        )
        return cls(
            provider=request.provider,
            model=request.model,
            messages=cls._to_anthropic_messages(request.messages),
            max_tokens=request.max_tokens,
            system=system or None,
            temperature=getattr(request, "temperature", None),
            top_p=getattr(request, "top_p", None),
            thinking=resolved_reasoning_mapping.thinking or None,
            output_config=output_config,
            base_url=config.base_url,
            api_key=resolve_api_key(config, label="Anthropic"),
            anthropic_version=config.anthropic_version,
            warnings=resolved_reasoning_mapping.warnings,
        )

    @staticmethod
    def _missing_max_tokens_error(provider: str) -> ProviderSemanticError:
        return ProviderSemanticError(f"{provider} requests require max_tokens")

    @staticmethod
    def _to_anthropic_messages(
        messages: list[Message],
    ) -> list[_AnthropicRequestMessage]:
        payload: list[_AnthropicRequestMessage] = []
        for message in messages:
            if message.role == "system":
                continue

            if message.role == "assistant":
                if message.content:
                    payload.append(
                        _AnthropicRequestMessage(
                            role="assistant",
                            content=[
                                _AnthropicRequestTextBlock(
                                    text=message.content
                                )
                            ],
                        )
                    )
                continue

            if message.role == "user" and message.content:
                payload.append(
                    _AnthropicRequestMessage(
                        role="user",
                        content=[
                            _AnthropicRequestTextBlock(text=message.content)
                        ],
                    )
                )

        return payload

    def endpoint(self) -> str:
        return self.base_url

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }

    def json_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

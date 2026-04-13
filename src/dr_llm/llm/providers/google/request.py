from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.api_config import APIProviderConfig, resolve_api_key
from dr_llm.llm.providers.google.reasoning import GoogleReasoningConfig
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.request import ApiBackedLlmRequest, ApiLlmRequest


class _GoogleGenerationConfig(BaseModel):
    temperature: float | None = None
    topP: float | None = None
    maxOutputTokens: int | None = None
    thinkingConfig: _GoogleThinkingConfig | None = None


class _GoogleThinkingConfig(BaseModel):
    thinkingBudget: int | None = None
    thinkingLevel: str | None = None
    includeThoughts: bool | None = None


class _GoogleRequestPart(BaseModel):
    text: str


class _GoogleRequestContent(BaseModel):
    role: Literal["user", "model"]
    parts: list[_GoogleRequestPart] = Field(default_factory=list)


class _GoogleSystemInstruction(BaseModel):
    parts: list[_GoogleRequestPart] = Field(default_factory=list)


class GoogleRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(exclude=True)
    model: str = Field(exclude=True)
    contents: list[_GoogleRequestContent]
    systemInstruction: _GoogleSystemInstruction | None = None
    generationConfig: _GoogleGenerationConfig | None = None
    base_url: str = Field(exclude=True)
    api_key_env: str = Field(exclude=True)
    api_key: str = Field(exclude=True, repr=False)
    warnings: list[ReasoningWarning] = Field(default_factory=list, exclude=True)

    @classmethod
    def from_llm_request(
        cls,
        request: ApiBackedLlmRequest,
        config: APIProviderConfig,
    ) -> GoogleRequest:
        if not isinstance(request, ApiLlmRequest):
            raise ProviderSemanticError(
                "google requires a sampling-capable API request shape"
            )
        reasoning_mapping = GoogleReasoningConfig.from_base(request.reasoning)
        system = "\n".join(
            message.content for message in request.messages if message.role == "system"
        )
        return cls(
            provider=request.provider,
            model=request.model,
            contents=cls._to_google_contents(request.messages),
            systemInstruction=(
                _GoogleSystemInstruction(parts=[_GoogleRequestPart(text=system)])
                if system
                else None
            ),
            generationConfig=cls._generation_config(
                request=request,
                reasoning_payload=reasoning_mapping.payload,
            ),
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            api_key=resolve_api_key(config, label="Google"),
            warnings=reasoning_mapping.warnings,
        )

    @staticmethod
    def _to_google_contents(messages: list[Message]) -> list[_GoogleRequestContent]:
        contents: list[_GoogleRequestContent] = []
        for message in messages:
            if message.role == "system" or not message.content:
                continue
            if message.role == "assistant":
                contents.append(
                    _GoogleRequestContent(
                        role="model",
                        parts=[_GoogleRequestPart(text=message.content)],
                    )
                )
                continue
            if message.role == "user":
                contents.append(
                    _GoogleRequestContent(
                        role="user",
                        parts=[_GoogleRequestPart(text=message.content)],
                    )
                )
                continue
            # e.g. Message.model_construct(role="tool", ...) bypasses validation
            role = message.role
            content_length = len(message.content)
            raise ValueError(
                "Unsupported Message.role for Google generateContent: "
                f"{role!r}. Supported roles are 'system', 'user', and 'assistant'. "
                f"Content length: {content_length}"
            )
        return contents

    @staticmethod
    def _generation_config(
        *,
        request: ApiLlmRequest,
        reasoning_payload: dict[str, Any],
    ) -> _GoogleGenerationConfig | None:
        thinking_config = (
            _GoogleThinkingConfig(**reasoning_payload) if reasoning_payload else None
        )
        if (
            request.temperature is not None
            or request.top_p is not None
            or request.max_tokens is not None
            or thinking_config is not None
        ):
            return _GoogleGenerationConfig(
                temperature=request.temperature,
                topP=request.top_p,
                maxOutputTokens=request.max_tokens,
                thinkingConfig=thinking_config,
            )
        return None

    def endpoint(self) -> str:
        return f"{self.base_url}/models/{self.model}:generateContent"

    def headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key}

    def json_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

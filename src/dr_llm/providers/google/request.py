from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.api_provider_config import APIProviderConfig
from dr_llm.providers.google.reasoning import GoogleReasoningConfig
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.models import Message
from dr_llm.providers.reasoning import ReasoningWarning


class _GoogleGenerationConfig(BaseModel):
    temperature: float | None = None
    topP: float | None = None
    maxOutputTokens: int | None = None


class _GoogleThinkingConfig(BaseModel):
    thinkingBudget: int | None = None
    thinkingLevel: str | None = None


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
    thinkingConfig: _GoogleThinkingConfig | None = None
    base_url: str = Field(exclude=True)
    api_key_env: str = Field(exclude=True)
    api_key: str = Field(exclude=True, repr=False)
    warnings: list[ReasoningWarning] = Field(default_factory=list, exclude=True)

    @classmethod
    def from_llm_request(
        cls,
        request: LlmRequest,
        config: APIProviderConfig,
    ) -> GoogleRequest:
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
            generationConfig=cls._generation_config(request=request),
            thinkingConfig=cls._thinking_config(
                reasoning_payload=reasoning_mapping.to_payload()
            ),
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            api_key=cls._resolve_api_key(config=config),
            warnings=reasoning_mapping.warnings,
        )

    @staticmethod
    def _resolve_api_key(*, config: APIProviderConfig) -> str:
        key = config.api_key or os.getenv(config.api_key_env)
        if not key:
            raise ProviderSemanticError(
                f"Missing Google API key. Set {config.api_key_env} or pass config.api_key"
            )
        return key

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
        return contents

    @staticmethod
    def _generation_config(
        *,
        request: LlmRequest,
    ) -> _GoogleGenerationConfig | None:
        generation_config = _GoogleGenerationConfig()
        has_generation_config = False
        if (
            request.temperature is not None
            or request.top_p is not None
            or request.max_tokens is not None
        ):
            generation_config = _GoogleGenerationConfig(
                temperature=request.temperature,
                topP=request.top_p,
                maxOutputTokens=request.max_tokens,
            )
            has_generation_config = True
        if not has_generation_config:
            return None
        return generation_config

    @staticmethod
    def _thinking_config(
        *,
        reasoning_payload: dict[str, Any],
    ) -> _GoogleThinkingConfig | None:
        if not reasoning_payload:
            return None
        return _GoogleThinkingConfig(
            thinkingBudget=(
                int(reasoning_payload["thinkingBudget"])
                if "thinkingBudget" in reasoning_payload
                else None
            ),
            thinkingLevel=(
                str(reasoning_payload["thinkingLevel"])
                if "thinkingLevel" in reasoning_payload
                else None
            ),
        )

    def endpoint(self) -> str:
        return f"{self.base_url}/models/{self.model}:generateContent"

    def headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key}

    def json_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=True)
        thinking_config = payload.pop("thinkingConfig", None)
        if thinking_config is not None:
            generation_config = payload.setdefault("generationConfig", {})
            generation_config["thinkingConfig"] = thinking_config
        return payload

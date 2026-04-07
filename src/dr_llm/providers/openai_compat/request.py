from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.models import Message
from dr_llm.providers.reasoning import ReasoningWarning
from dr_llm.providers.openai_compat.reasoning import OpenAICompatReasoningConfig

if TYPE_CHECKING:
    from dr_llm.providers.openai_compat.config import OpenAICompatConfig


class OpenAICompatRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(exclude=True)
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    base_url: str = Field(exclude=True)
    chat_path: str = Field(exclude=True)
    api_key_env: str = Field(exclude=True)
    api_key: str = Field(exclude=True, repr=False)
    idempotency_key: str = Field(exclude=True)
    warnings: list[ReasoningWarning] = Field(default_factory=list, exclude=True)

    @classmethod
    def from_llm_request(
        cls,
        request: LlmRequest,
        config: OpenAICompatConfig,
    ) -> OpenAICompatRequest:
        reasoning_mapping = OpenAICompatReasoningConfig.from_base(
            request.reasoning,
            provider=request.provider,
            model=request.model,
        )
        reasoning_effort = reasoning_mapping.to_reasoning_effort()
        extra_body = reasoning_mapping.to_extra_body()
        return cls(
            provider=request.provider,
            model=request.model,
            messages=cls._to_openai_messages(request.messages),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            base_url=config.base_url,
            chat_path=config.chat_path,
            api_key_env=config.api_key_env,
            api_key=cls._resolve_api_key(config=config, provider=request.provider),
            idempotency_key=cls._resolve_idempotency_key(request=request),
            warnings=reasoning_mapping.warnings,
        )

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content} for message in messages
        ]

    @staticmethod
    def _resolve_api_key(
        *,
        config: OpenAICompatConfig,
        provider: str,
    ) -> str:
        key = config.api_key or os.getenv(config.api_key_env)
        if not key:
            raise ProviderSemanticError(
                f"Missing API key for {provider}. Set {config.api_key_env} or pass config.api_key"
            )
        return key

    @staticmethod
    def _resolve_idempotency_key(*, request: LlmRequest) -> str:
        raw_idempotency_key = request.metadata.get("idempotency_key")
        if isinstance(raw_idempotency_key, str) and raw_idempotency_key:
            return raw_idempotency_key
        return uuid4().hex

    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + self.chat_path

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": self.idempotency_key,
        }

    def json_payload(self) -> dict[str, Any]:
        payload = self.model_dump(
            mode="json", exclude_none=True, exclude={"extra_body"}
        )
        overlapping_keys = sorted(set(self.extra_body) & set(payload))
        if overlapping_keys:
            conflicts = ", ".join(overlapping_keys)
            raise ValueError(
                f"extra_body conflicts with validated payload keys: {conflicts}"
            )
        payload.update(self.extra_body)
        return payload

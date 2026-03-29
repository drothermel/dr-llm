from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.generation.models import (
    CallMode,
    LlmRequest,
    LlmResponse,
    ReasoningWarning,
    TokenUsage,
)
from dr_llm.providers.openai_compat_request import OpenAICompatRequest
from dr_llm.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
)


class _OpenAICompatUsageDetails(BaseModel):
    reasoning_tokens: int | None = None


class _OpenAICompatUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: _OpenAICompatUsageDetails | None = None
    output_tokens_details: _OpenAICompatUsageDetails | None = None


class _OpenAICompatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: str | None = None
    reasoning: str | int | float | None = None
    reasoning_content: str | int | float | None = None
    reasoning_details: list[dict[str, Any]] | None = None


class _OpenAICompatChoice(BaseModel):
    finish_reason: str | None = None
    message: _OpenAICompatMessage


class _OpenAICompatResponseBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: list[_OpenAICompatChoice] = Field(default_factory=list)
    usage: _OpenAICompatUsage | None = None


class OpenAICompatResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status_code: int
    response_text_preview: str = Field(exclude=True)
    raw_json: dict[str, Any] | None = Field(default=None, exclude=True, repr=False)
    choices: list[_OpenAICompatChoice] = Field(default_factory=list)
    usage: _OpenAICompatUsage | None = None
    json_error: str | None = Field(default=None, exclude=True, repr=False)
    response_shape_error: str | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def from_http_response(cls, response: httpx.Response) -> OpenAICompatResponse:
        raw_json: dict[str, Any] | None = None
        choices: list[_OpenAICompatChoice] = []
        usage: _OpenAICompatUsage | None = None
        json_error: str | None = None
        response_shape_error: str | None = None

        try:
            body_raw = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            json_error = str(exc)
        else:
            if not isinstance(body_raw, dict):
                response_shape_error = "expected JSON object"
            else:
                raw_json = body_raw
                try:
                    parsed = _OpenAICompatResponseBody(**body_raw)
                except ValidationError as exc:
                    response_shape_error = str(exc)
                else:
                    choices = parsed.choices
                    usage = parsed.usage

        return cls(
            status_code=response.status_code,
            response_text_preview=response.text[:500],
            raw_json=raw_json,
            choices=choices,
            usage=usage,
            json_error=json_error,
            response_shape_error=response_shape_error,
        )

    def event_payload(self, request: OpenAICompatRequest) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "endpoint": request.endpoint(),
            "idempotency_key": request.idempotency_key,
            "response_text_preview": self.response_text_preview,
            "request_shape": {
                "model": request.model,
                "message_count": len(request.messages),
            },
        }

    def _validated_choice(self, *, provider_name: str) -> _OpenAICompatChoice:
        if self.status_code >= 500 or self.status_code == 429:
            raise ProviderTransportError(
                f"{provider_name} transient error status={self.status_code} body={self.response_text_preview}"
            )
        if self.status_code >= 400:
            raise ProviderSemanticError(
                f"{provider_name} request rejected status={self.status_code} body={self.response_text_preview}"
            )
        if self.json_error is not None:
            raise ProviderTransportError(
                f"{provider_name} invalid JSON response: {self.json_error}"
            )
        if self.response_shape_error is not None:
            raise ProviderSemanticError(
                f"{provider_name} response shape invalid: {self.response_shape_error}"
            )
        if not self.choices:
            raise ProviderSemanticError(f"{provider_name} response missing choices")
        return self.choices[0]

    def to_llm_response(
        self,
        request: LlmRequest,
        *,
        latency_ms: int,
        warnings: list[ReasoningWarning],
    ) -> LlmResponse:
        choice = self._validated_choice(provider_name=request.provider)
        message = choice.message
        usage_raw = (
            self.usage.model_dump(mode="json", exclude_none=True) if self.usage else {}
        )
        reasoning_tokens = parse_reasoning_tokens(usage_raw)
        usage = TokenUsage.from_raw(
            prompt_tokens=self.usage.prompt_tokens if self.usage else None,
            completion_tokens=self.usage.completion_tokens if self.usage else None,
            total_tokens=self.usage.total_tokens if self.usage else None,
            reasoning_tokens=reasoning_tokens,
        )
        reasoning, reasoning_details = parse_reasoning(
            message.model_dump(mode="json", exclude_none=True)
        )
        raw_json = self.raw_json or {}
        return LlmResponse(
            text=message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            cost=parse_cost_info(raw_json),
            raw_json=raw_json,
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
            warnings=warnings,
        )

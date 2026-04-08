from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse
from dr_llm.llm.messages import CallMode
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.providers.openai_compat.request import OpenAICompatRequest
from dr_llm.llm.providers.response_validation import (
    parse_http_response_body,
    validate_http_response,
)
from dr_llm.llm.providers.usage import CostInfo, build_usage_and_reasoning


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
    raw_json: Any | None = Field(default=None, exclude=True, repr=False)
    choices: list[_OpenAICompatChoice] = Field(default_factory=list)
    usage: _OpenAICompatUsage | None = None
    json_error: str | None = Field(default=None, exclude=True, repr=False)
    response_shape_error: str | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def from_http_response(cls, response: httpx.Response) -> OpenAICompatResponse:
        raw_json, parsed, json_error, shape_error = parse_http_response_body(
            response, _OpenAICompatResponseBody
        )
        return cls(
            status_code=response.status_code,
            response_text_preview=response.text[:500],
            raw_json=raw_json,
            choices=parsed.choices if parsed else [],
            usage=parsed.usage if parsed else None,
            json_error=json_error,
            response_shape_error=shape_error,
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
        validate_http_response(
            provider_label=provider_name,
            status_code=self.status_code,
            response_text_preview=self.response_text_preview,
            json_error=self.json_error,
            response_shape_error=self.response_shape_error,
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
        usage_dump = (
            self.usage.model_dump(mode="json", exclude_none=True)
            if self.usage
            else None
        )
        usage, reasoning, reasoning_details = build_usage_and_reasoning(
            usage_dump=usage_dump,
            prompt_tokens=self.usage.prompt_tokens if self.usage else None,
            completion_tokens=self.usage.completion_tokens if self.usage else None,
            total_tokens=self.usage.total_tokens if self.usage else None,
            reasoning_source=message.model_dump(mode="json", exclude_none=True),
        )
        raw_json = self.raw_json if isinstance(self.raw_json, dict) else {}
        return LlmResponse(
            text=message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            cost=CostInfo.from_raw(raw_json),
            raw_json=raw_json,
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
            warnings=warnings,
        )

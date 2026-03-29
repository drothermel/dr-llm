from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.providers.anthropic.request import AnthropicRequest
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.models import CallMode, ReasoningWarning
from dr_llm.providers.usage import CostInfo, TokenUsage, parse_reasoning


class _AnthropicUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_details: dict[str, Any] | None = None


class _AnthropicContentItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None


class _AnthropicResponseBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: list[_AnthropicContentItem] = Field(default_factory=list)
    usage: _AnthropicUsage | None = None
    stop_reason: str | None = None


class AnthropicResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status_code: int
    response_text: str = Field(exclude=True)
    response_text_preview: str = Field(exclude=True)
    raw_json: dict[str, Any] | None = Field(default=None, exclude=True, repr=False)
    content: list[_AnthropicContentItem] = Field(default_factory=list)
    usage: _AnthropicUsage | None = None
    stop_reason: str | None = None
    json_error: str | None = Field(default=None, exclude=True, repr=False)
    response_shape_error: str | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def from_http_response(cls, response: httpx.Response) -> AnthropicResponse:
        raw_json: dict[str, Any] | None = None
        content: list[_AnthropicContentItem] = []
        usage: _AnthropicUsage | None = None
        stop_reason: str | None = None
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
                    parsed = _AnthropicResponseBody(**body_raw)
                except ValidationError as exc:
                    response_shape_error = str(exc)
                else:
                    content = parsed.content
                    usage = parsed.usage
                    stop_reason = parsed.stop_reason
        return cls(
            status_code=response.status_code,
            response_text=response.text,
            response_text_preview=response.text[:500],
            raw_json=raw_json,
            content=content,
            usage=usage,
            stop_reason=stop_reason,
            json_error=json_error,
            response_shape_error=response_shape_error,
        )

    def event_payload(self, request: AnthropicRequest) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "endpoint": request.endpoint(),
            "response_text": self.response_text,
            "request_payload": request.json_payload(),
        }

    def _validate(self) -> None:
        if self.status_code >= 500 or self.status_code == 429:
            raise ProviderTransportError(
                f"anthropic transient error status={self.status_code} body={self.response_text_preview}"
            )
        if self.status_code >= 400:
            raise ProviderSemanticError(
                f"anthropic rejected request status={self.status_code} body={self.response_text_preview}"
            )
        if self.json_error is not None:
            raise ProviderTransportError(
                f"anthropic invalid JSON response: {self.json_error}"
            )
        if self.response_shape_error is not None:
            raise ProviderSemanticError(
                f"anthropic response shape invalid: {self.response_shape_error}"
            )

    def to_llm_response(
        self,
        request: LlmRequest,
        *,
        latency_ms: int,
        warnings: list[ReasoningWarning],
    ) -> LlmResponse:
        self._validate()
        text_chunks = [item.text or "" for item in self.content if item.type == "text"]
        usage_raw = (
            self.usage.model_dump(mode="json", exclude_none=True) if self.usage else {}
        )
        reasoning_tokens = TokenUsage.extract_reasoning_tokens(usage_raw)
        usage = TokenUsage.from_raw(
            prompt_tokens=self.usage.input_tokens if self.usage else None,
            completion_tokens=self.usage.output_tokens if self.usage else None,
            total_tokens=(
                (self.usage.input_tokens or 0) + (self.usage.output_tokens or 0)
                if self.usage
                else None
            ),
            reasoning_tokens=reasoning_tokens,
        )
        raw_json = self.raw_json or {}
        reasoning, reasoning_details = parse_reasoning(raw_json)
        return LlmResponse(
            text="\n".join(chunk for chunk in text_chunks if chunk),
            finish_reason=self.stop_reason,
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

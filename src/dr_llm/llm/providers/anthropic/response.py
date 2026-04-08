from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm.providers.anthropic.request import AnthropicRequest
from dr_llm.llm.providers.response_validation import (
    parse_http_response_body,
    validate_http_response,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse
from dr_llm.llm.messages import CallMode
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.providers.usage import CostInfo, build_usage_and_reasoning


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
    raw_json: Any | None = Field(default=None, exclude=True, repr=False)
    content: list[_AnthropicContentItem] = Field(default_factory=list)
    usage: _AnthropicUsage | None = None
    stop_reason: str | None = None
    json_error: str | None = Field(default=None, exclude=True, repr=False)
    response_shape_error: str | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def from_http_response(cls, response: httpx.Response) -> AnthropicResponse:
        raw_json, parsed, json_error, shape_error = parse_http_response_body(
            response, _AnthropicResponseBody
        )
        return cls(
            status_code=response.status_code,
            response_text=response.text,
            response_text_preview=response.text[:500],
            raw_json=raw_json,
            content=parsed.content if parsed else [],
            usage=parsed.usage if parsed else None,
            stop_reason=parsed.stop_reason if parsed else None,
            json_error=json_error,
            response_shape_error=shape_error,
        )

    def event_payload(self, request: AnthropicRequest) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "endpoint": request.endpoint(),
            "response_text": self.response_text,
            "request_payload": request.json_payload(),
        }

    def _validate(self) -> None:
        validate_http_response(
            provider_label="anthropic",
            status_code=self.status_code,
            response_text_preview=self.response_text_preview,
            json_error=self.json_error,
            response_shape_error=self.response_shape_error,
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
        raw_json = self.raw_json if isinstance(self.raw_json, dict) else {}
        usage_dump = (
            self.usage.model_dump(mode="json", exclude_none=True)
            if self.usage
            else None
        )
        prompt_tokens = self.usage.input_tokens if self.usage else None
        completion_tokens = self.usage.output_tokens if self.usage else None
        total_tokens = (
            (prompt_tokens or 0) + (completion_tokens or 0) if self.usage else None
        )
        usage, reasoning, reasoning_details = build_usage_and_reasoning(
            usage_dump=usage_dump,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_source=raw_json,
        )
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

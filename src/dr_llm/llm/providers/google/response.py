from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.google.request import GoogleRequest
from dr_llm.llm.providers.response_validation import (
    parse_http_response_body,
    validate_http_response,
)
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse
from dr_llm.llm.messages import CallMode
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.providers.usage import CostInfo, build_usage_and_reasoning


class _GooglePart(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str | None = None
    thought: bool | None = None


class _GoogleContent(BaseModel):
    role: str | None = None
    parts: list[_GooglePart] = Field(default_factory=list)


class _GoogleCandidate(BaseModel):
    content: _GoogleContent | None = None
    finishReason: str | None = None


class _GoogleUsageMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    promptTokenCount: int | None = None
    candidatesTokenCount: int | None = None
    totalTokenCount: int | None = None
    output_tokens_details: dict[str, Any] | None = Field(
        default=None,
        alias="candidatesTokensDetails",
    )


class _GoogleResponseBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidates: list[_GoogleCandidate] = Field(default_factory=list)
    usageMetadata: _GoogleUsageMetadata | None = None


class GoogleResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status_code: int
    response_text: str = Field(exclude=True)
    response_text_preview: str = Field(exclude=True)
    raw_json: Any | None = Field(default=None, exclude=True, repr=False)
    candidates: list[_GoogleCandidate] = Field(default_factory=list)
    usageMetadata: _GoogleUsageMetadata | None = None
    json_error: str | None = Field(default=None, exclude=True, repr=False)
    response_shape_error: str | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def from_http_response(cls, response: httpx.Response) -> GoogleResponse:
        raw_json, parsed, json_error, shape_error = parse_http_response_body(
            response, _GoogleResponseBody
        )
        return cls(
            status_code=response.status_code,
            response_text=response.text,
            response_text_preview=response.text[:500],
            raw_json=raw_json,
            candidates=parsed.candidates if parsed else [],
            usageMetadata=parsed.usageMetadata if parsed else None,
            json_error=json_error,
            response_shape_error=shape_error,
        )

    def event_payload(self, request: GoogleRequest) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "endpoint": request.endpoint(),
            "response_preview": self.response_text_preview,
            "response_length": len(self.response_text),
        }

    def _validated_candidate(self) -> _GoogleCandidate:
        validate_http_response(
            provider_label="google",
            status_code=self.status_code,
            response_text_preview=self.response_text_preview,
            json_error=self.json_error,
            response_shape_error=self.response_shape_error,
        )
        if not self.candidates:
            raise ProviderSemanticError("google response missing candidates")
        return self.candidates[0]

    def to_llm_response(
        self,
        request: LlmRequest,
        *,
        latency_ms: int,
        warnings: list[ReasoningWarning],
    ) -> LlmResponse:
        candidate = self._validated_candidate()
        parts = candidate.content.parts if candidate.content else []
        text_chunks = [part.text for part in parts if part.text and not part.thought]
        thought_chunks = [part.text for part in parts if part.text and part.thought]
        thought_details = [
            part.model_dump(mode="json", exclude_none=True)
            for part in parts
            if part.thought
        ]
        usage_dump = (
            self.usageMetadata.model_dump(mode="json", exclude_none=True)
            if self.usageMetadata
            else None
        )
        raw_json = self.raw_json if isinstance(self.raw_json, dict) else {}
        usage, fallback_reasoning, fallback_reasoning_details = (
            build_usage_and_reasoning(
                usage_dump=usage_dump,
                prompt_tokens=(
                    self.usageMetadata.promptTokenCount if self.usageMetadata else None
                ),
                completion_tokens=(
                    self.usageMetadata.candidatesTokenCount
                    if self.usageMetadata
                    else None
                ),
                total_tokens=(
                    self.usageMetadata.totalTokenCount if self.usageMetadata else None
                ),
                reasoning_source=raw_json,
            )
        )
        reasoning = "\n".join(thought_chunks) if thought_chunks else fallback_reasoning
        reasoning_details = thought_details or fallback_reasoning_details
        return LlmResponse(
            text="\n".join(text_chunks),
            finish_reason=candidate.finishReason,
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

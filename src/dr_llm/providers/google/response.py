from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.providers.google.request import GoogleRequest
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.llm_response import LlmResponse
from dr_llm.providers.models import CallMode
from dr_llm.providers.reasoning import ReasoningWarning
from dr_llm.providers.usage import CostInfo, TokenUsage, parse_reasoning


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
    raw_json: dict[str, Any] | None = Field(default=None, exclude=True, repr=False)
    candidates: list[_GoogleCandidate] = Field(default_factory=list)
    usageMetadata: _GoogleUsageMetadata | None = None
    json_error: str | None = Field(default=None, exclude=True, repr=False)
    response_shape_error: str | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def from_http_response(cls, response: httpx.Response) -> GoogleResponse:
        raw_json: dict[str, Any] | None = None
        candidates: list[_GoogleCandidate] = []
        usage_metadata: _GoogleUsageMetadata | None = None
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
                    parsed = _GoogleResponseBody(**body_raw)
                except ValidationError as exc:
                    response_shape_error = str(exc)
                else:
                    candidates = parsed.candidates
                    usage_metadata = parsed.usageMetadata
        return cls(
            status_code=response.status_code,
            response_text=response.text,
            response_text_preview=response.text[:500],
            raw_json=raw_json,
            candidates=candidates,
            usageMetadata=usage_metadata,
            json_error=json_error,
            response_shape_error=response_shape_error,
        )

    def event_payload(self, request: GoogleRequest) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "endpoint": request.endpoint(),
            "response_preview": self.response_text_preview,
            "response_length": len(self.response_text),
        }

    def _validated_candidate(self) -> _GoogleCandidate:
        if self.status_code >= 500 or self.status_code == 429:
            raise ProviderTransportError(
                f"google transient error status={self.status_code} body={self.response_text_preview}"
            )
        if self.status_code >= 400:
            raise ProviderSemanticError(
                f"google rejected request status={self.status_code} body={self.response_text_preview}"
            )
        if self.json_error is not None:
            raise ProviderTransportError(
                f"google invalid JSON response: {self.json_error}"
            )
        if self.response_shape_error is not None:
            raise ProviderSemanticError(
                f"google response shape invalid: {self.response_shape_error}"
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
        usage_raw = (
            self.usageMetadata.model_dump(mode="json", exclude_none=True)
            if self.usageMetadata
            else {}
        )
        reasoning_tokens = TokenUsage.extract_reasoning_tokens(usage_raw)
        usage = TokenUsage.from_raw(
            prompt_tokens=(
                self.usageMetadata.promptTokenCount if self.usageMetadata else None
            ),
            completion_tokens=(
                self.usageMetadata.candidatesTokenCount if self.usageMetadata else None
            ),
            total_tokens=(
                self.usageMetadata.totalTokenCount if self.usageMetadata else None
            ),
            reasoning_tokens=reasoning_tokens,
        )
        raw_json = self.raw_json or {}
        fallback_reasoning, fallback_reasoning_details = parse_reasoning(raw_json)
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

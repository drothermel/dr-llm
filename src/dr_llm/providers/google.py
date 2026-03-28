from __future__ import annotations

import json
import os
import time
from typing import Any, Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dr_llm.errors import ProviderSemanticError, ProviderTransportError
from dr_llm.logging import emit_generation_event
from dr_llm.providers.base import (
    ProviderAdapter,
    ProviderCapabilities,
    ProviderRuntimeRequirements,
)
from dr_llm.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
)
from dr_llm.reasoning import map_reasoning_for_google
from dr_llm.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    ModelToolCall,
    ProviderToolSpec,
    TokenUsage,
)


class _GoogleGenerationConfig(BaseModel):
    temperature: float | None = None
    topP: float | None = None
    maxOutputTokens: int | None = None
    thinkingBudget: int | None = None
    thinkingLevel: str | None = None


class _GoogleFunctionDeclaration(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class _GoogleToolBlock(BaseModel):
    functionDeclarations: list[_GoogleFunctionDeclaration]


class _GoogleFunctionCall(BaseModel):
    name: str | None = None
    args: dict[str, Any] | None = None


class _GoogleRequestFunctionResponse(BaseModel):
    name: str
    response: dict[str, Any]


class _GoogleRequestPart(BaseModel):
    text: str | None = None
    functionCall: _GoogleFunctionCall | None = None
    functionResponse: _GoogleRequestFunctionResponse | None = None

    @model_validator(mode="after")
    def _validate_exactly_one_content_field(self) -> _GoogleRequestPart:
        populated_fields = sum(
            value is not None
            for value in (self.text, self.functionCall, self.functionResponse)
        )
        if populated_fields != 1:
            raise ValueError(
                "google request part must include exactly one of text, functionCall, functionResponse"
            )
        return self


class _GoogleRequestContent(BaseModel):
    role: Literal["user", "model"]
    parts: list[_GoogleRequestPart] = Field(default_factory=list)


class _GoogleSystemInstruction(BaseModel):
    parts: list[_GoogleRequestPart] = Field(default_factory=list)


class _GoogleRequestPayload(BaseModel):
    contents: list[_GoogleRequestContent]
    systemInstruction: _GoogleSystemInstruction | None = None
    generationConfig: _GoogleGenerationConfig | None = None
    tools: list[_GoogleToolBlock] | None = None


class _GooglePart(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str | None = None
    functionCall: _GoogleFunctionCall | None = None


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


class _GoogleResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidates: list[_GoogleCandidate] = Field(default_factory=list)
    usageMetadata: _GoogleUsageMetadata | None = None


class GoogleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GOOGLE_API_KEY"
    api_key: str | None = None
    timeout_seconds: float = 120.0


class GoogleAdapter(ProviderAdapter):
    name = "google"
    mode = "api"

    def __init__(
        self, config: GoogleConfig | None = None, client: httpx.Client | None = None
    ) -> None:
        self._config = config or GoogleConfig()
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)

    @property
    def config(self) -> GoogleConfig:
        return self._config

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_native_tools=True, supports_structured_output=True
        )

    @property
    def runtime_requirements(self) -> ProviderRuntimeRequirements:
        required_env_vars = [] if self._config.api_key else [self._config.api_key_env]
        return ProviderRuntimeRequirements(
            required_env_vars=required_env_vars,
        )

    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.TransportError, ProviderTransportError)
        ),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, request: LlmRequest) -> LlmResponse:
        reasoning_mapping = map_reasoning_for_google(
            request.reasoning,
            provider=request.provider,
            mode=CallMode.api,
        )
        key = self._config.api_key or os.getenv(self._config.api_key_env)
        if not key:
            raise ProviderSemanticError(
                f"Missing Google API key. Set {self._config.api_key_env} or pass config.api_key"
            )
        endpoint = f"{self._config.base_url}/models/{request.model}:generateContent"
        system = "\n".join(
            msg.content for msg in request.messages if msg.role == "system"
        )
        contents = _to_google_contents(request.messages)
        payload = _GoogleRequestPayload(contents=contents)
        generation_cfg = _GoogleGenerationConfig()
        has_generation_cfg = False
        if system:
            payload.systemInstruction = _GoogleSystemInstruction(
                parts=[_GoogleRequestPart(text=system)]
            )
        if (
            request.temperature is not None
            or request.top_p is not None
            or request.max_tokens is not None
        ):
            generation_cfg = _GoogleGenerationConfig(
                temperature=request.temperature,
                topP=request.top_p,
                maxOutputTokens=request.max_tokens,
            )
            has_generation_cfg = True
        if reasoning_mapping.payload:
            generation_cfg = generation_cfg.model_copy(
                update={
                    "thinkingBudget": (
                        int(reasoning_mapping.payload["thinkingBudget"])
                        if "thinkingBudget" in reasoning_mapping.payload
                        else generation_cfg.thinkingBudget
                    ),
                    "thinkingLevel": (
                        str(reasoning_mapping.payload["thinkingLevel"])
                        if "thinkingLevel" in reasoning_mapping.payload
                        else generation_cfg.thinkingLevel
                    ),
                }
            )
            has_generation_cfg = True
        if has_generation_cfg:
            payload.generationConfig = generation_cfg
        normalized_tools = _to_google_tools(request.tools)
        if normalized_tools:
            payload.tools = normalized_tools

        started = time.perf_counter()
        resp = self._client.post(
            endpoint,
            headers={"x-goog-api-key": key},
            json=payload.model_dump(mode="json", exclude_none=True),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        emit_generation_event(
            event_type="provider.raw_response",
            stage="google.http_response",
            payload={
                "status_code": resp.status_code,
                "endpoint": endpoint,
                "response_text": resp.text,
                "request_payload": payload.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_computed_fields=True,
                ),
            },
        )
        if resp.status_code >= 500 or resp.status_code == 429:
            raise ProviderTransportError(
                f"google transient error status={resp.status_code} body={resp.text[:500]}"
            )
        if resp.status_code >= 400:
            raise ProviderSemanticError(
                f"google rejected request status={resp.status_code} body={resp.text[:500]}"
            )
        try:
            body_raw = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderTransportError(
                f"google invalid JSON response: {exc}"
            ) from exc
        if not isinstance(body_raw, dict):
            raise ProviderSemanticError(
                "google response shape invalid: expected JSON object"
            )
        try:
            body = _GoogleResponse(**body_raw)
        except ValidationError as exc:
            raise ProviderSemanticError(
                f"google response shape invalid: {exc}"
            ) from exc
        if not body.candidates:
            raise ProviderSemanticError("google response missing candidates")
        candidate = body.candidates[0]
        parts = candidate.content.parts if candidate.content else []
        text_chunks: list[str] = []
        tool_calls: list[ModelToolCall] = []
        request_id = uuid4().hex
        tool_call_ordinal = 0
        for part in parts:
            if part.text:
                text_chunks.append(part.text)
            fc = part.functionCall
            if fc is not None:
                name = str(fc.name or "")
                if not name:
                    continue
                tool_call_ordinal += 1
                tool_calls.append(
                    ModelToolCall(
                        tool_call_id=f"google_{request_id}_call_{tool_call_ordinal}",
                        name=name,
                        arguments=fc.args if isinstance(fc.args, dict) else {},
                    )
                )
        usage_raw = (
            body.usageMetadata.model_dump(mode="json", exclude_none=True)
            if body.usageMetadata
            else {}
        )
        reasoning_tokens = parse_reasoning_tokens(usage_raw)
        usage = TokenUsage.from_raw(
            prompt_tokens=body.usageMetadata.promptTokenCount
            if body.usageMetadata
            else None,
            completion_tokens=body.usageMetadata.candidatesTokenCount
            if body.usageMetadata
            else None,
            total_tokens=body.usageMetadata.totalTokenCount
            if body.usageMetadata
            else None,
            reasoning_tokens=reasoning_tokens,
        )
        reasoning, reasoning_details = parse_reasoning(
            body_raw if isinstance(body_raw, dict) else None
        )
        cost = parse_cost_info(body_raw if isinstance(body_raw, dict) else {})
        return LlmResponse(
            text="\n".join(chunk for chunk in text_chunks if chunk),
            finish_reason=candidate.finishReason,
            usage=usage,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            cost=cost,
            raw_json=body_raw if isinstance(body_raw, dict) else {"body": body_raw},
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
            tool_calls=tool_calls,
            warnings=reasoning_mapping.warnings,
        )


def _to_google_tools(
    tools: list[ProviderToolSpec] | None,
) -> list[_GoogleToolBlock] | None:
    if not tools:
        return None
    declarations: list[_GoogleFunctionDeclaration] = []
    for tool in tools:
        declaration = _GoogleFunctionDeclaration(
            name=tool.function.name,
            description=tool.function.description or None,
            parameters=tool.function.parameters or None,
        )
        declarations.append(declaration)
    return (
        [_GoogleToolBlock(functionDeclarations=declarations)] if declarations else None
    )


def _parse_tool_response_content(content: str) -> dict[str, Any]:
    if not content:
        return {"content": ""}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {"content": content}
    if isinstance(parsed, dict):
        return parsed
    return {"content": parsed}


def _to_google_contents(messages: list[Message]) -> list[_GoogleRequestContent]:
    contents: list[_GoogleRequestContent] = []
    for msg in messages:
        if msg.role == "system":
            continue

        if msg.role == "tool":
            if msg.name:
                contents.append(
                    _GoogleRequestContent(
                        role="user",
                        parts=[
                            _GoogleRequestPart(
                                functionResponse=_GoogleRequestFunctionResponse(
                                    name=msg.name,
                                    response=_parse_tool_response_content(msg.content),
                                )
                            )
                        ],
                    )
                )
            else:
                raise ValueError(
                    f"google tool message missing name; cannot encode functionResponse: {msg!r}"
                )
            continue

        if msg.role == "assistant":
            parts: list[_GoogleRequestPart] = []
            if msg.content:
                parts.append(_GoogleRequestPart(text=msg.content))
            if msg.tool_calls:
                for call in msg.tool_calls:
                    parts.append(
                        _GoogleRequestPart(
                            functionCall=_GoogleFunctionCall(
                                name=call.name,
                                args=call.arguments,
                            )
                        )
                    )
            if parts:
                contents.append(_GoogleRequestContent(role="model", parts=parts))
            continue

        if msg.role == "user" and msg.content:
            contents.append(
                _GoogleRequestContent(
                    role="user",
                    parts=[_GoogleRequestPart(text=msg.content)],
                )
            )
    return contents

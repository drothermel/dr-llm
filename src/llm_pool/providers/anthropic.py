from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_pool.errors import ProviderSemanticError, ProviderTransportError
from llm_pool.logging import emit_generation_event
from llm_pool.providers.base import ProviderAdapter, ProviderCapabilities
from llm_pool.providers.utils import (
    parse_cost_info,
    parse_reasoning,
    parse_reasoning_tokens,
)
from llm_pool.reasoning import map_reasoning_for_anthropic
from llm_pool.types import (
    CallMode,
    LlmRequest,
    LlmResponse,
    Message,
    ModelToolCall,
    ProviderToolSpec,
    TokenUsage,
)


class _AnthropicToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class _AnthropicRequestTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class _AnthropicRequestToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class _AnthropicRequestToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool | None = None


class _AnthropicRequestMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[
        _AnthropicRequestTextBlock
        | _AnthropicRequestToolUseBlock
        | _AnthropicRequestToolResultBlock
    ] = Field(default_factory=list)


class _AnthropicRequestPayload(BaseModel):
    model: str
    messages: list[_AnthropicRequestMessage]
    max_tokens: int
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[_AnthropicToolSpec] | None = None
    thinking: dict[str, Any] | None = None


class _AnthropicUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_details: dict[str, Any] | None = None


class _AnthropicContentItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


class _AnthropicResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: list[_AnthropicContentItem] = Field(default_factory=list)
    usage: _AnthropicUsage | None = None
    stop_reason: str | None = None


class AnthropicConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str = "https://api.anthropic.com/v1/messages"
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key: str | None = None
    anthropic_version: str = "2023-06-01"
    timeout_seconds: float = 120.0


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"
    mode = "api"

    def __init__(
        self, config: AnthropicConfig | None = None, client: httpx.Client | None = None
    ) -> None:
        self._config = config or AnthropicConfig()
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=self._config.timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> AnthropicAdapter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def config(self) -> AnthropicConfig:
        return self._config

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_native_tools=True, supports_structured_output=True
        )

    def _headers(self) -> dict[str, str]:
        key = self._config.api_key or os.getenv(self._config.api_key_env)
        if not key:
            raise ProviderSemanticError(
                f"Missing Anthropic API key. Set {self._config.api_key_env} or pass config.api_key"
            )
        return {
            "x-api-key": key,
            "anthropic-version": self._config.anthropic_version,
            "content-type": "application/json",
        }

    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.TransportError, ProviderTransportError)
        ),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, request: LlmRequest) -> LlmResponse:
        reasoning_mapping = map_reasoning_for_anthropic(
            request.reasoning,
            provider=request.provider,
            mode=CallMode.api,
            request_max_tokens=request.max_tokens,
        )
        system = "\n".join(
            msg.content for msg in request.messages if msg.role == "system"
        )
        messages = _to_anthropic_messages(request.messages)
        payload = _AnthropicRequestPayload(
            model=request.model,
            messages=messages,
            max_tokens=request.max_tokens or 1024,
            system=system or None,
            temperature=request.temperature,
            top_p=request.top_p,
            thinking=reasoning_mapping.payload or None,
        )
        normalized_tools = _to_anthropic_tools(request.tools)
        if normalized_tools:
            payload.tools = normalized_tools
        started = time.perf_counter()
        resp = self._client.post(
            self._config.base_url,
            headers=self._headers(),
            json=payload.model_dump(mode="json", exclude_none=True),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        emit_generation_event(
            event_type="provider.raw_response",
            stage="anthropic.http_response",
            payload={
                "status_code": resp.status_code,
                "endpoint": self._config.base_url,
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
                f"anthropic transient error status={resp.status_code} body={resp.text[:500]}"
            )
        if resp.status_code >= 400:
            raise ProviderSemanticError(
                f"anthropic rejected request status={resp.status_code} body={resp.text[:500]}"
            )

        try:
            body_raw = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderTransportError(
                f"anthropic invalid JSON response: {exc}"
            ) from exc
        if not isinstance(body_raw, dict):
            raise ProviderSemanticError(
                "anthropic response shape invalid: expected JSON object"
            )
        try:
            body = _AnthropicResponse(**body_raw)
        except ValidationError as exc:
            raise ProviderSemanticError(
                f"anthropic response shape invalid: {exc}"
            ) from exc

        text_chunks: list[str] = []
        tool_calls: list[ModelToolCall] = []
        for idx, item in enumerate(body.content):
            item_type = item.type
            if item_type == "text":
                text_chunks.append(item.text or "")
            elif item_type == "tool_use":
                tool_calls.append(
                    ModelToolCall(
                        tool_call_id=str(item.id or f"call_{idx + 1}"),
                        name=str(item.name or ""),
                        arguments=item.input if isinstance(item.input, dict) else {},
                    )
                )

        usage_raw = (
            body.usage.model_dump(mode="json", exclude_none=True) if body.usage else {}
        )
        reasoning_tokens = parse_reasoning_tokens(usage_raw)
        usage = TokenUsage.from_raw(
            prompt_tokens=body.usage.input_tokens if body.usage else None,
            completion_tokens=body.usage.output_tokens if body.usage else None,
            total_tokens=(
                (body.usage.input_tokens or 0) + (body.usage.output_tokens or 0)
                if body.usage
                else None
            ),
            reasoning_tokens=reasoning_tokens,
        )
        reasoning, reasoning_details = parse_reasoning(
            body_raw if isinstance(body_raw, dict) else None
        )
        cost = parse_cost_info(body_raw if isinstance(body_raw, dict) else {})
        return LlmResponse(
            text="\n".join(chunk for chunk in text_chunks if chunk),
            finish_reason=body.stop_reason,
            usage=usage,
            reasoning=reasoning,
            reasoning_details=reasoning_details,
            cost=cost,
            raw_json=body_raw if isinstance(body_raw, dict) else {"body": body_raw},
            latency_ms=latency_ms,
            provider=request.provider,
            model=request.model,
            mode=CallMode.api,
            tool_calls=[tc for tc in tool_calls if tc.name],
            warnings=reasoning_mapping.warnings,
        )


def _to_anthropic_tools(
    tools: list[ProviderToolSpec] | None,
) -> list[_AnthropicToolSpec] | None:
    if not tools:
        return None
    return [
        _AnthropicToolSpec(
            name=tool.function.name,
            description=tool.function.description,
            input_schema=tool.function.parameters,
        )
        for tool in tools
    ] or None


def _to_anthropic_messages(messages: list[Message]) -> list[_AnthropicRequestMessage]:
    payload: list[_AnthropicRequestMessage] = []
    for msg in messages:
        if msg.role == "system":
            continue

        if msg.role == "tool":
            if msg.tool_call_id:
                content = msg.content or "{}"
                block = _AnthropicRequestToolResultBlock(
                    tool_use_id=msg.tool_call_id,
                    content=content,
                )
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and "error" in parsed:
                    block = _AnthropicRequestToolResultBlock(
                        tool_use_id=msg.tool_call_id,
                        content=content,
                        is_error=True,
                    )
                payload.append(_AnthropicRequestMessage(role="user", content=[block]))
            elif msg.content:
                payload.append(
                    _AnthropicRequestMessage(
                        role="user",
                        content=[_AnthropicRequestTextBlock(text=msg.content)],
                    )
                )
            continue

        if msg.role == "assistant":
            blocks: list[
                _AnthropicRequestTextBlock
                | _AnthropicRequestToolUseBlock
                | _AnthropicRequestToolResultBlock
            ] = []
            if msg.content:
                blocks.append(_AnthropicRequestTextBlock(text=msg.content))
            if msg.tool_calls:
                for call in msg.tool_calls:
                    blocks.append(
                        _AnthropicRequestToolUseBlock(
                            id=call.tool_call_id,
                            name=call.name,
                            input=call.arguments,
                        )
                    )
            if blocks:
                payload.append(
                    _AnthropicRequestMessage(role="assistant", content=blocks)
                )
            continue

        if msg.role == "user" and msg.content:
            payload.append(
                _AnthropicRequestMessage(
                    role="user",
                    content=[_AnthropicRequestTextBlock(text=msg.content)],
                )
            )

    return payload

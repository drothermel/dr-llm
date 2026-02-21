from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm_pool.types import Message, ModelToolCall, ReasoningConfig, ToolError


class SessionEventType(StrEnum):
    session_started = "session_started"
    message = "message"
    session_canceled = "session_canceled"
    model_response = "model_response"
    model_requested_tool = "model_requested_tool"
    tool_started = "tool_started"
    tool_queued = "tool_queued"
    tool_succeeded = "tool_succeeded"
    tool_failed = "tool_failed"
    tool_result_message = "tool_result_message"
    session_waiting_for_tools = "session_waiting_for_tools"
    model_completed = "model_completed"
    session_step_failed = "session_step_failed"


class SessionMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    model: str
    run_id: str | None = None
    reasoning: ReasoningConfig | None = None

    @field_validator("provider", "model")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider/model must be non-empty")
        return normalized


class SessionStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[Message]


class SessionMessagePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: Message


class ModelResponseData(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    finish_reason: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)


class ModelResponsePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    response: ModelResponseData


class ModelRequestedToolPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call: ModelToolCall


class ToolLifecyclePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str


class ToolExecutionPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    result: dict[str, Any] | None = None
    error: ToolError | None = None


class SessionWaitingForToolsPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_ids: list[str]
    message: str


class SessionStepFailedPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    error_type: str
    message: str


class SessionCanceledPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str


class BrokeredToolCallCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool call name must be non-empty")
        return normalized


class ToolProcessingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    final_output: Message
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    waiting_for_tools: bool = False


def payload_dict(payload: BaseModel) -> dict[str, Any]:
    return payload.model_dump(mode="json", exclude_computed_fields=True)


def parse_brokered_tool_calls(text: str) -> list[ModelToolCall]:
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []

    parsed: list[ModelToolCall] = []
    for idx, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            continue
        try:
            candidate = BrokeredToolCallCandidate(**item)
        except ValidationError:
            continue
        parsed.append(
            ModelToolCall(
                tool_call_id=candidate.tool_call_id or f"brokered_call_{idx + 1}",
                name=candidate.name,
                arguments=candidate.arguments,
            )
        )
    return parsed


def error_payload(error: ToolError | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return error.model_dump(
        mode="json",
        exclude_none=True,
        exclude_computed_fields=True,
    )

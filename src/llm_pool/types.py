from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class CallMode(str, Enum):
    api = "api"
    headless = "headless"


class RunStatus(str, Enum):
    running = "running"
    success = "success"
    failed = "failed"
    canceled = "canceled"


class SessionStatus(str, Enum):
    active = "active"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


class SessionTurnStatus(str, Enum):
    active = "active"
    completed = "completed"
    failed = "failed"


class ToolPolicy(str, Enum):
    native_preferred = "native_preferred"
    brokered_only = "brokered_only"
    native_only = "native_only"


class ToolCallStatus(str, Enum):
    pending = "pending"
    claimed = "claimed"
    succeeded = "succeeded"
    failed = "failed"
    dead_letter = "dead_letter"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ModelToolCall] | None = None

    @model_validator(mode="after")
    def _validate_role_fields(self) -> Message:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages must include tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("tool_calls are only valid for assistant messages")
        return self


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    effort: Literal["xhigh", "high", "medium", "low", "minimal", "none"] | None = None
    max_tokens: int | None = None
    exclude: bool | None = None
    enabled: bool | None = None

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("reasoning.max_tokens must be > 0")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> ReasoningConfig:
        if self.effort is not None and self.max_tokens is not None:
            raise ValueError(
                "reasoning.effort and reasoning.max_tokens are mutually exclusive"
            )
        if self.enabled is False and (
            self.effort is not None or self.max_tokens is not None
        ):
            raise ValueError(
                "reasoning.enabled=false cannot be combined with effort or max_tokens"
            )
        return self

    @computed_field
    @property
    def effective_enabled(self) -> bool:
        if self.enabled is not None:
            return self.enabled
        return self.effort is not None or self.max_tokens is not None


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    @field_validator(
        "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"
    )
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("token counts must be non-negative")
        return value

    @computed_field
    @property
    def computed_total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_cost_usd: float | None = None
    prompt_cost_usd: float | None = None
    completion_cost_usd: float | None = None
    reasoning_cost_usd: float | None = None
    currency: str | None = "USD"
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None
    tool_policy: ToolPolicy = ToolPolicy.native_preferred


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    cost: CostInfo | None = None
    raw_json: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    provider: str
    model: str
    mode: CallMode
    tool_calls: list[ModelToolCall] = Field(default_factory=list)


class CallError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error_type: str
    message: str
    retryable: bool = False
    raw_json: dict[str, Any] | None = None


class SessionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    session_id: str
    turn_id: str | None = None
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


class SessionState(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    status: SessionStatus
    version: int
    strategy_mode: ToolPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    last_error_text: str | None = None


class SessionHandle(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    status: SessionStatus
    version: int
    strategy_mode: ToolPolicy


class SessionStartInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    messages: list[Message]
    reasoning: ReasoningConfig | None = None
    strategy_mode: ToolPolicy = ToolPolicy.native_preferred
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None


class SessionStepInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    messages: list[Message]
    expected_version: int | None = None
    inline_tool_execution: bool = False
    reasoning: ReasoningConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStepResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    turn_id: str
    status: SessionTurnStatus
    version: int
    output: Message | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)


class ToolInvocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    session_id: str
    turn_id: str | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    ok: bool
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    session_id: str
    turn_id: str | None
    idempotency_key: str
    tool_name: str
    status: ToolCallStatus
    args: dict[str, Any]
    attempt_count: int
    worker_id: str | None
    created_at: datetime
    claimed_at: datetime | None
    lease_expires_at: datetime | None


class RecordedCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    run_id: str | None
    provider: str
    model: str
    mode: CallMode
    status: str
    created_at: datetime
    latency_ms: int
    error_text: str | None
    reasoning_tokens: int = 0
    reasoning_text: str | None = None
    cost_total_usd: float | None = None
    cost_prompt_usd: float | None = None
    cost_completion_usd: float | None = None
    cost_reasoning_usd: float | None = None
    request: dict[str, Any]
    response: dict[str, Any] | None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

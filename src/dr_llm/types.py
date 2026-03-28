from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class CallMode(StrEnum):
    api = "api"
    headless = "headless"


class RunStatus(StrEnum):
    running = "running"
    success = "success"
    failed = "failed"
    canceled = "canceled"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


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

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    reasoning_tokens: int = Field(default=0)

    @classmethod
    def _coerce_token_count(cls, value: Any, *, field_name: str) -> int:
        if value is None:
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
        if parsed < 0:
            raise ValueError("token counts must be non-negative")
        return parsed

    @model_validator(mode="before")
    @classmethod
    def _normalize_counts(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        prompt_tokens = cls._coerce_token_count(
            data.get("prompt_tokens"), field_name="prompt_tokens"
        )
        completion_tokens = cls._coerce_token_count(
            data.get("completion_tokens"), field_name="completion_tokens"
        )
        reasoning_tokens = cls._coerce_token_count(
            data.get("reasoning_tokens"), field_name="reasoning_tokens"
        )
        total_raw = data.get("total_tokens")
        total_tokens = (
            prompt_tokens + completion_tokens
            if total_raw is None
            else cls._coerce_token_count(total_raw, field_name="total_tokens")
        )
        return {
            **data,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    @classmethod
    def from_raw(
        cls,
        *,
        prompt_tokens: Any = None,
        completion_tokens: Any = None,
        total_tokens: Any = None,
        reasoning_tokens: Any = None,
    ) -> TokenUsage:
        return cls(
            **{
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
            }
        )

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


class ReasoningWarningCode(StrEnum):
    unsupported_for_provider = "unsupported_for_provider"
    mapped_with_heuristic = "mapped_with_heuristic"
    partially_supported = "partially_supported"


class ReasoningWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ReasoningWarningCode
    message: str
    provider: str | None = None
    mode: CallMode | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ModelCatalogPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str | None = "USD"
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    reasoning_cost_per_1m: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelCatalogRateLimit(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    concurrency_limit: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelCatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_reasoning: bool | None = None
    supports_vision: bool | None = None
    pricing: ModelCatalogPricing | None = None
    rate_limits: ModelCatalogRateLimit | None = None
    source_quality: Literal["live", "static"] = "live"
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime | None = None


class ModelCatalogQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str | None = None
    supports_reasoning: bool | None = None
    model_contains: str | None = None
    limit: int = 200
    offset: int = 0


class LlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning: ReasoningConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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
    warnings: list[ReasoningWarning] = Field(default_factory=list)


class CallError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error_type: str
    message: str
    retryable: bool = False
    raw_json: dict[str, Any] | None = None


class RecordedCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    run_id: str | None
    provider: str
    model: str
    mode: CallMode
    status: str
    created_at: datetime
    latency_ms: int | None
    error_text: str | None
    reasoning_tokens: int = 0
    reasoning_text: str | None = None
    cost_total_usd: float | None = None
    cost_prompt_usd: float | None = None
    cost_completion_usd: float | None = None
    cost_reasoning_usd: float | None = None
    warnings: list[ReasoningWarning] = Field(default_factory=list)
    request: dict[str, Any]
    response: dict[str, Any] | None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

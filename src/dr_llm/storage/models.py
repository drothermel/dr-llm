from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.generation.models import CallMode, ReasoningWarning


class RunStatus(StrEnum):
    running = "running"
    success = "success"
    failed = "failed"
    canceled = "canceled"


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

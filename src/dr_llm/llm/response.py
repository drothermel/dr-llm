from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm.messages import CallMode
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.providers.usage import CostInfo, TokenUsage


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

from __future__ import annotations

import sys
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.llm.providers.usage import CostInfo, TokenUsage

if TYPE_CHECKING:
    from dr_llm.llm.providers.reasoning import ReasoningWarning


class CallMode(StrEnum):
    api = "api"
    headless = "headless"


class CallError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error_type: str
    message: str
    retryable: bool = False
    raw_json: dict[str, Any] | None = None


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


def _rebuild_with_reasoning_warning() -> None:
    # CallMode lives here while ReasoningWarning keeps its provider-specific
    # model; rebuild the forward ref once both modules are available.
    reasoning_module = sys.modules.get("dr_llm.llm.providers.reasoning")
    if reasoning_module is None:
        from dr_llm.llm.providers import reasoning as reasoning_module

    reasoning_warning = getattr(reasoning_module, "ReasoningWarning", None)
    if reasoning_warning is None:
        return
    LlmResponse.model_rebuild(
        _types_namespace={"ReasoningWarning": reasoning_warning}
    )


_rebuild_with_reasoning_warning()

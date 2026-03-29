from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.providers.models import CallMode, ReasoningWarning, ReasoningWarningCode
from dr_llm.providers.reasoning import ReasoningConfig


_EFFORT_RATIO = {
    "xhigh": 0.95,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2,
    "minimal": 0.1,
    "none": 0.0,
}


class AnthropicReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningConfig | None,
        *,
        provider: str,
        mode: CallMode,
        request_max_tokens: int | None,
    ) -> AnthropicReasoningConfig:
        if config is None:
            return cls()
        warnings: list[ReasoningWarning] = []
        budget: int | None = None
        if config.max_tokens is not None:
            budget = max(1024, min(int(config.max_tokens), 128000))
        elif config.effort is not None:
            max_tokens = int(request_max_tokens or 2048)
            ratio = _EFFORT_RATIO.get(config.effort, 0.5)
            budget = max(1024, min(int(max_tokens * ratio), 128000))
        if budget is None:
            return cls()
        if request_max_tokens is not None and request_max_tokens <= 1024:
            warnings.append(
                ReasoningWarning(
                    code=ReasoningWarningCode.mapped_with_heuristic,
                    message=(
                        "anthropic reasoning budget cannot be strictly less than max_tokens when max_tokens <= 1024; reasoning payload was omitted"
                    ),
                    provider=provider,
                    mode=mode,
                    details={"requested_max_tokens": request_max_tokens},
                )
            )
            return cls(warnings=warnings)
        if request_max_tokens is not None and budget >= request_max_tokens:
            warnings.append(
                ReasoningWarning(
                    code=ReasoningWarningCode.mapped_with_heuristic,
                    message=(
                        "anthropic reasoning budget was clamped below max_tokens to leave room for final output"
                    ),
                    provider=provider,
                    mode=mode,
                    details={
                        "budget_tokens": max(1024, request_max_tokens - 1),
                        "requested_max_tokens": request_max_tokens,
                    },
                )
            )
            budget = max(1024, request_max_tokens - 1)
        return cls(
            payload={"type": "enabled", "budget_tokens": budget},
            warnings=warnings,
        )

    def to_payload(self) -> dict[str, Any]:
        return self.payload

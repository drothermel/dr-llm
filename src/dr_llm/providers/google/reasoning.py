from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.providers.models import CallMode, ReasoningWarning, ReasoningWarningCode
from dr_llm.providers.reasoning import ReasoningConfig


_GOOGLE_THINKING_LEVEL = {
    "xhigh": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "minimal": "minimal",
    "none": "minimal",
}


class GoogleReasoningConfig(BaseModel):
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
    ) -> GoogleReasoningConfig:
        if config is None:
            return cls()
        payload: dict[str, Any] = {}
        warnings: list[ReasoningWarning] = []
        if config.max_tokens is not None:
            payload["thinkingBudget"] = int(config.max_tokens)
        if config.effort is not None:
            payload["thinkingLevel"] = _GOOGLE_THINKING_LEVEL.get(
                config.effort, "medium"
            )
        if config.exclude is True:
            warnings.append(
                ReasoningWarning(
                    code=ReasoningWarningCode.partially_supported,
                    message="google reasoning.exclude is not directly supported and was ignored",
                    provider=provider,
                    mode=mode,
                )
            )
        if config.enabled is False:
            payload["thinkingBudget"] = 0
            payload["thinkingLevel"] = "minimal"
        return cls(payload=payload, warnings=warnings)

    def to_payload(self) -> dict[str, Any]:
        return self.payload

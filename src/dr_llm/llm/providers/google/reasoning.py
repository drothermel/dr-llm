from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.providers.reasoning import (
    GoogleReasoning,
    ReasoningBudget,
    ReasoningSpec,
    ThinkingLevel,
)


def _require_google_budget_tokens(budget_tokens: int | None) -> int:
    """Validate tokens for Google thinkingBudget (non-negative int)."""
    if budget_tokens is None:
        raise ValueError(
            "google budget thinking requires budget_tokens when thinking_level is 'budget'"
        )
    if type(budget_tokens) is not int:
        raise ValueError(
            f"google budget_tokens must be int, got {type(budget_tokens).__name__}"
        )
    if budget_tokens < 0:
        raise ValueError("google budget_tokens must be >= 0")
    return budget_tokens


class GoogleReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> GoogleReasoningConfig:
        if config is None:
            return cls()
        match config:
            case ReasoningBudget(tokens=tokens):
                return cls(payload={"thinkingBudget": tokens})
            case GoogleReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                include_thoughts=include_thoughts,
            ):
                payload: dict[str, Any] = {}
                if thinking_level == ThinkingLevel.NA:
                    return cls()
                if thinking_level == ThinkingLevel.OFF:
                    payload["thinkingBudget"] = 0
                elif thinking_level == ThinkingLevel.ADAPTIVE:
                    payload["thinkingBudget"] = -1
                elif thinking_level == ThinkingLevel.BUDGET:
                    payload["thinkingBudget"] = _require_google_budget_tokens(
                        budget_tokens
                    )
                elif thinking_level in {
                    ThinkingLevel.MINIMAL,
                    ThinkingLevel.LOW,
                    ThinkingLevel.MEDIUM,
                    ThinkingLevel.HIGH,
                }:
                    payload["thinkingLevel"] = str(thinking_level)
                else:
                    raise ProviderSemanticError(
                        "google reasoning config did not contain a serializable setting"
                    )
                if include_thoughts is not None:
                    payload["includeThoughts"] = include_thoughts
                return cls(payload=payload)
            case _:
                raise ProviderSemanticError(
                    f"google reasoning serializer received unsupported config kind={config.kind!r}"
                )

    def to_payload(self) -> dict[str, Any]:
        return self.payload

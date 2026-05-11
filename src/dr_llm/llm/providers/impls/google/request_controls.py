from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    GoogleReasoning,
    ReasoningBudget,
    ReasoningSpec,
    ReasoningWarning,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
)


class GoogleThinkingPayloadKey(StrEnum):
    THINKING_BUDGET = "thinkingBudget"
    THINKING_LEVEL = "thinkingLevel"
    INCLUDE_THOUGHTS = "includeThoughts"


class GoogleThinkingBudget(IntEnum):
    OFF = 0
    ADAPTIVE = -1


class GoogleRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> GoogleRequestControls:
        if reasoning is None:
            return cls()
        match reasoning:
            case ReasoningBudget(tokens=tokens):
                return cls(
                    payload={GoogleThinkingPayloadKey.THINKING_BUDGET: tokens}
                )
            case GoogleReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                include_thoughts=include_thoughts,
            ):
                if thinking_level == ThinkingLevel.NA:
                    return cls()
                payload = _build_thinking_payload(
                    thinking_level=thinking_level,
                    budget_tokens=budget_tokens,
                )
                if include_thoughts is not None:
                    payload[GoogleThinkingPayloadKey.INCLUDE_THOUGHTS] = (
                        include_thoughts
                    )
                return cls(payload=payload)
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.GOOGLE, reasoning
                    )
                )


_GOOGLE_LITERAL_LEVELS = {
    ThinkingLevel.MINIMAL,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
}


def _build_thinking_payload(
    *,
    thinking_level: ThinkingLevel,
    budget_tokens: int | None,
) -> dict[str, Any]:
    if thinking_level == ThinkingLevel.OFF:
        return {
            GoogleThinkingPayloadKey.THINKING_BUDGET: GoogleThinkingBudget.OFF
        }
    if thinking_level == ThinkingLevel.ADAPTIVE:
        return {
            GoogleThinkingPayloadKey.THINKING_BUDGET: (
                GoogleThinkingBudget.ADAPTIVE
            )
        }
    if thinking_level == ThinkingLevel.BUDGET:
        return {
            GoogleThinkingPayloadKey.THINKING_BUDGET: require_budget_tokens(
                budget_tokens, label=ProviderName.GOOGLE, min_value=0
            )
        }
    if thinking_level in _GOOGLE_LITERAL_LEVELS:
        return {GoogleThinkingPayloadKey.THINKING_LEVEL: thinking_level}
    raise ProviderSemanticError(
        f"Unsupported google thinking_level={thinking_level!r}"
    )

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    ReasoningBudget,
    ReasoningSpec,
    ReasoningWarning,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
)


class AnthropicThinkingType(StrEnum):
    ENABLED = "enabled"
    ADAPTIVE = "adaptive"


class AnthropicRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    thinking: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> AnthropicRequestControls:
        if reasoning is None:
            return cls()
        match reasoning:
            case ReasoningBudget(tokens=tokens):
                budget_tokens = require_budget_tokens(
                    tokens,
                    label=ProviderName.ANTHROPIC,
                    min_value=1,
                )
                return cls(
                    thinking={
                        "type": AnthropicThinkingType.ENABLED,
                        "budget_tokens": budget_tokens,
                    }
                )
            case AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                display=display,
            ):
                thinking: dict[str, Any] = {}
                if thinking_level == ThinkingLevel.BUDGET:
                    tokens = require_budget_tokens(
                        budget_tokens,
                        label=ProviderName.ANTHROPIC,
                        min_value=1,
                    )
                    thinking = {
                        "type": AnthropicThinkingType.ENABLED,
                        "budget_tokens": tokens,
                    }
                elif thinking_level == ThinkingLevel.ADAPTIVE:
                    thinking = {"type": AnthropicThinkingType.ADAPTIVE}
                if display is not None:
                    thinking["display"] = display
                return cls(thinking=thinking)
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.ANTHROPIC, reasoning
                    )
                )

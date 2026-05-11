from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    ReasoningSpec,
    ReasoningWarning,
    require_budget_tokens,
    unsupported_reasoning_kind_message,
)


class KimiCodeThinkingType(StrEnum):
    DISABLED = "disabled"
    ADAPTIVE = "adaptive"
    ENABLED = "enabled"


class KimiCodeRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    thinking: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> KimiCodeRequestControls:
        if reasoning is None:
            return cls()
        match reasoning:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                budget_tokens=None,
                display=None,
            ):
                return cls()
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.OFF,
                budget_tokens=None,
                display=None,
            ):
                return cls(thinking={"type": KimiCodeThinkingType.DISABLED})
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=None,
                display=None,
            ):
                return cls(thinking={"type": KimiCodeThinkingType.ADAPTIVE})
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=budget_tokens,
                display=None,
            ):
                tokens = require_budget_tokens(
                    budget_tokens, label=ProviderName.KIMI_CODE, min_value=1
                )
                return cls(
                    thinking={
                        "type": KimiCodeThinkingType.ENABLED,
                        "budget_tokens": tokens,
                    }
                )
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.KIMI_CODE, reasoning
                    )
                )

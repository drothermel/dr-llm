from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    ReasoningSpec,
    ReasoningWarning,
    unsupported_reasoning_kind_message,
)


class MiniMaxRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    thinking: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> MiniMaxRequestControls:
        if reasoning is None:
            raise ProviderSemanticError(
                f"{ProviderName.MINIMAX} requires explicit AnthropicReasoning(thinking_level='na')"
            )
        match reasoning:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                budget_tokens=None,
                display=None,
            ):
                return cls()
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.MINIMAX, reasoning
                    )
                )

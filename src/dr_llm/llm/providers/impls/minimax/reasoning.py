from __future__ import annotations

from typing import Any

from pydantic import Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    BaseProviderReasoningConfig,
    ReasoningBudget,
    ReasoningSpec,
    unsupported_reasoning_kind_message,
)


def validate_reasoning_for_minimax(
    *, model: str, reasoning: ReasoningSpec | None
) -> None:
    if reasoning is None:
        raise ValueError(
            f"reasoning is required for provider='{ProviderName.MINIMAX}' model={model!r}"
        )
    if isinstance(reasoning, ReasoningBudget):
        raise ValueError(
            "minimax requires anthropic reasoning configs; "
            "use AnthropicReasoning(thinking_level='na')"
        )
    if not isinstance(reasoning, AnthropicReasoning):
        raise ValueError(
            f"minimax reasoning is not supported for kind={reasoning.kind!r}"
        )
    if reasoning.display is not None:
        raise ValueError("minimax does not support anthropic display controls")
    if reasoning.thinking_level != ThinkingLevel.NA:
        raise ValueError(
            "minimax does not support explicit anthropic thinking; use thinking_level='na'"
        )


class MiniMaxReasoningConfig(BaseProviderReasoningConfig):
    thinking: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> MiniMaxReasoningConfig:
        if config is None:
            raise ProviderSemanticError(
                f"{ProviderName.MINIMAX} requires explicit AnthropicReasoning(thinking_level='na')"
            )
        match config:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA,
                budget_tokens=None,
                display=None,
            ):
                return cls()
            case _:
                raise ProviderSemanticError(
                    unsupported_reasoning_kind_message(
                        ProviderName.MINIMAX, config
                    )
                )

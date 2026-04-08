from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    ReasoningBudget,
    ReasoningSpec,
    ThinkingLevel,
)


def _require_anthropic_api_budget_tokens(
    budget_tokens: int | None, *, label: str
) -> int:
    """Validate tokens for Anthropic-style thinking payloads (strictly positive int)."""
    if budget_tokens is None:
        raise ValueError(
            f"{label} budget thinking requires budget_tokens when thinking_level is 'budget'"
        )
    if type(budget_tokens) is not int:
        raise ValueError(
            f"{label} budget_tokens must be int, got {type(budget_tokens).__name__}"
        )
    if budget_tokens <= 0:
        raise ValueError(f"{label} budget_tokens must be > 0")
    return budget_tokens


class AnthropicReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    thinking: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> AnthropicReasoningConfig:
        if config is None:
            return cls()
        match config:
            case ReasoningBudget(tokens=tokens):
                return cls(thinking={"type": "enabled", "budget_tokens": tokens})
            case AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=budget_tokens,
                display=display,
            ):
                thinking: dict[str, Any] = {}
                if thinking_level == ThinkingLevel.BUDGET:
                    tokens = _require_anthropic_api_budget_tokens(
                        budget_tokens, label="anthropic"
                    )
                    thinking = {"type": "enabled", "budget_tokens": tokens}
                elif thinking_level == ThinkingLevel.ADAPTIVE:
                    thinking = {"type": "adaptive"}
                if display is not None:
                    thinking["display"] = display
                return cls(thinking=thinking)
            case _:
                raise ProviderSemanticError(
                    f"anthropic reasoning serializer received unsupported config kind={config.kind!r}"
                )

    def thinking_payload(self) -> dict[str, Any]:
        return self.thinking


class KimiCodeReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    thinking: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> KimiCodeReasoningConfig:
        if config is None:
            return cls()
        match config:
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
                return cls(thinking={"type": "disabled"})
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=None,
                display=None,
            ):
                return cls(thinking={"type": "adaptive"})
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.BUDGET,
                budget_tokens=budget_tokens,
                display=None,
            ):
                tokens = _require_anthropic_api_budget_tokens(
                    budget_tokens, label="kimi-code"
                )
                return cls(thinking={"type": "enabled", "budget_tokens": tokens})
            case _:
                raise ProviderSemanticError(
                    f"kimi-code reasoning serializer received unsupported config kind={config.kind!r}"
                )

    def thinking_payload(self) -> dict[str, Any]:
        return self.thinking


class MiniMaxReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    thinking: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> MiniMaxReasoningConfig:
        if config is None:
            raise ProviderSemanticError(
                "minimax requires explicit AnthropicReasoning(thinking_level='na')"
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
                    f"minimax reasoning serializer received unsupported config kind={config.kind!r}"
                )

    def thinking_payload(self) -> dict[str, Any]:
        return self.thinking

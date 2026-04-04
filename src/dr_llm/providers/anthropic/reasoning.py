from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.reasoning import ReasoningWarning
from dr_llm.providers.reasoning import (
    AnthropicReasoning,
    ReasoningBudget,
    ReasoningSpec,
    ThinkingLevel,
)


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
                    assert budget_tokens is not None
                    thinking = {"type": "enabled", "budget_tokens": budget_tokens}
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
                assert budget_tokens is not None
                return cls(thinking={"type": "enabled", "budget_tokens": budget_tokens})
            case _:
                raise ProviderSemanticError(
                    f"kimi-code reasoning serializer received unsupported config kind={config.kind!r}"
                )

    def thinking_payload(self) -> dict[str, Any]:
        return self.thinking

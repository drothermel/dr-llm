from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.reasoning import ReasoningWarning
from dr_llm.providers.reasoning import (
    AnthropicReasoning,
    ReasoningBudget,
    ReasoningEffort,
    ReasoningSpec,
)


class AnthropicReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    thinking: dict[str, Any] = Field(default_factory=dict)
    output_config: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> AnthropicReasoningConfig:
        if config is None:
            return cls()
        match config:
            case ReasoningEffort(level=level):
                return cls(output_config={"effort": level})
            case ReasoningBudget(tokens=tokens):
                return cls(thinking={"type": "enabled", "budget_tokens": tokens})
            case AnthropicReasoning(
                effort=effort,
                budget_tokens=budget_tokens,
                thinking_mode=thinking_mode,
                display=display,
            ):
                thinking: dict[str, Any] = {}
                output_config: dict[str, Any] = {}
                if effort is not None:
                    output_config["effort"] = effort
                if (
                    budget_tokens is not None
                    or thinking_mode is not None
                    or display is not None
                ):
                    thinking_type = thinking_mode or "enabled"
                    thinking["type"] = thinking_type
                    if budget_tokens is not None:
                        thinking["budget_tokens"] = budget_tokens
                    if display is not None:
                        thinking["display"] = display
                return cls(thinking=thinking, output_config=output_config)
            case _:
                raise ProviderSemanticError(
                    f"anthropic reasoning serializer received unsupported config kind={config.kind!r}"
                )

    def thinking_payload(self) -> dict[str, Any]:
        return self.thinking

    def output_config_payload(self) -> dict[str, Any]:
        return self.output_config

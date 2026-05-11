from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.names import ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    ReasoningSpec,
    ReasoningWarning,
    unsupported_reasoning_kind_message,
)


class ClaudeCodeRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    cli_args: list[str] = Field(default_factory=list)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> ClaudeCodeRequestControls:
        if reasoning is None:
            return cls()
        match reasoning:
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.NA, display=None
            ):
                return cls()
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=None,
                display=None,
            ):
                return cls()
        raise HeadlessExecutionError(
            unsupported_reasoning_kind_message("claude headless", reasoning)
        )

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.names import ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    CodexReasoning,
    ReasoningSpec,
    ReasoningWarning,
    unsupported_reasoning_kind_message,
)


class CodexCliConfigKey(StrEnum):
    MODEL_REASONING_EFFORT = "model_reasoning_effort"


class CodexReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class CodexRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    cli_args: list[str] = Field(default_factory=list)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> CodexRequestControls:
        if reasoning is None:
            return cls()
        match reasoning:
            case CodexReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case CodexReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(
                    cli_args=[
                        "-c",
                        _codex_reasoning_effort_config(
                            CodexReasoningEffort.NONE
                        ),
                    ]
                )
            case CodexReasoning(
                thinking_level=ThinkingLevel.MINIMAL
                | ThinkingLevel.LOW
                | ThinkingLevel.MEDIUM
                | ThinkingLevel.HIGH
                | ThinkingLevel.XHIGH
            ):
                return cls(
                    cli_args=[
                        "-c",
                        _codex_reasoning_effort_config(
                            CodexReasoningEffort(reasoning.thinking_level)
                        ),
                    ]
                )
        raise HeadlessExecutionError(
            unsupported_reasoning_kind_message("codex headless", reasoning)
        )


def _codex_reasoning_effort_config(effort: CodexReasoningEffort) -> str:
    return f'{CodexCliConfigKey.MODEL_REASONING_EFFORT}="{effort}"'

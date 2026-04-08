from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.llm.providers.reasoning import ReasoningWarning
from dr_llm.llm.providers.reasoning import (
    AnthropicReasoning,
    CodexReasoning,
    ReasoningSpec,
    ThinkingLevel,
)


class ClaudeHeadlessReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cli_args: list[str] = Field(default_factory=list)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> ClaudeHeadlessReasoningConfig:
        if config is None:
            return cls()
        match config:
            case AnthropicReasoning(thinking_level=ThinkingLevel.NA, display=None):
                return cls()
            case AnthropicReasoning(
                thinking_level=ThinkingLevel.ADAPTIVE,
                budget_tokens=None,
                display=None,
            ):
                return cls()
        raise HeadlessExecutionError(
            f"claude headless reasoning serializer received unsupported config kind={config.kind!r}"
        )

    def to_cli_args(self) -> list[str]:
        return self.cli_args


class CodexHeadlessReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cli_args: list[str] = Field(default_factory=list)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> CodexHeadlessReasoningConfig:
        if config is None:
            return cls()
        match config:
            case CodexReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case CodexReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(cli_args=["-c", 'model_reasoning_effort="none"'])
            case CodexReasoning(
                thinking_level=ThinkingLevel.MINIMAL
                | ThinkingLevel.LOW
                | ThinkingLevel.MEDIUM
                | ThinkingLevel.HIGH
            ):
                thinking_level = config.thinking_level
                return cls(
                    cli_args=["-c", f'model_reasoning_effort="{thinking_level}"']
                )
        raise HeadlessExecutionError(
            f"codex headless reasoning serializer received unsupported config kind={config.kind!r}"
        )

    def to_cli_args(self) -> list[str]:
        return self.cli_args

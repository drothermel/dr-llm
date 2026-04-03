from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import HeadlessExecutionError
from dr_llm.providers.reasoning import ReasoningWarning
from dr_llm.providers.reasoning import ReasoningSpec


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
        raise HeadlessExecutionError(
            f"codex headless does not support reasoning config kind={config.kind!r}"
        )

    def to_cli_args(self) -> list[str]:
        return self.cli_args

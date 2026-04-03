from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.reasoning import ReasoningWarning
from dr_llm.providers.reasoning import OpenAIReasoning, ReasoningSpec, ThinkingLevel


class OpenAICompatReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    ) = None
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> OpenAICompatReasoningConfig:
        if config is None:
            return cls()
        match config:
            case OpenAIReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(reasoning_effort="none")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL):
                return cls(reasoning_effort="minimal")
            case OpenAIReasoning(thinking_level=ThinkingLevel.LOW):
                return cls(reasoning_effort="low")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MEDIUM):
                return cls(reasoning_effort="medium")
            case OpenAIReasoning(thinking_level=ThinkingLevel.HIGH):
                return cls(reasoning_effort="high")
            case OpenAIReasoning(thinking_level=ThinkingLevel.XHIGH):
                return cls(reasoning_effort="xhigh")
        raise ProviderSemanticError(
            f"OpenAI-compatible reasoning serializer received unsupported config kind={config.kind!r}"
        )

    def to_reasoning_effort(self) -> str | None:
        return self.reasoning_effort

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    ReasoningSpec,
    ReasoningWarning,
    unsupported_reasoning_kind_message,
)


class OpenAIReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OpenAIRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning_effort: OpenAIReasoningEffort | None = None
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> OpenAIRequestControls:
        if reasoning is None:
            return cls()
        match reasoning:
            case OpenAIReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(reasoning_effort=OpenAIReasoningEffort.NONE)
            case OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL):
                return cls(reasoning_effort=OpenAIReasoningEffort.MINIMAL)
            case OpenAIReasoning(thinking_level=ThinkingLevel.LOW):
                return cls(reasoning_effort=OpenAIReasoningEffort.LOW)
            case OpenAIReasoning(thinking_level=ThinkingLevel.MEDIUM):
                return cls(reasoning_effort=OpenAIReasoningEffort.MEDIUM)
            case OpenAIReasoning(thinking_level=ThinkingLevel.HIGH):
                return cls(reasoning_effort=OpenAIReasoningEffort.HIGH)
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(ProviderName.OPENAI, reasoning)
        )

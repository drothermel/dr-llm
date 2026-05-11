from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningSpec,
    ReasoningWarning,
    unsupported_reasoning_kind_message,
)
from dr_llm.llm.providers.impls.openai.request_controls import (
    OpenAIReasoningEffort,
)


class OpenRouterReasoningKey(StrEnum):
    REASONING = "reasoning"
    ENABLED = "enabled"
    EFFORT = "effort"


class OpenRouterRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning_effort: OpenAIReasoningEffort | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> OpenRouterRequestControls:
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
            case OpenRouterReasoning(enabled=enabled, effort=effort):
                reasoning_payload: dict[str, Any]
                if enabled is not None:
                    reasoning_payload = {
                        OpenRouterReasoningKey.ENABLED: enabled
                    }
                elif effort is not None:
                    reasoning_payload = {OpenRouterReasoningKey.EFFORT: effort}
                else:
                    raise ProviderSemanticError(
                        "OpenRouter reasoning serializer received invalid config"
                    )
                return cls(
                    extra_body={
                        OpenRouterReasoningKey.REASONING: reasoning_payload
                    }
                )
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(
                ProviderName.OPENROUTER, reasoning
            )
        )

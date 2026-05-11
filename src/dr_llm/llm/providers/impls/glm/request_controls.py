from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    GlmReasoning,
    ReasoningSpec,
    ReasoningWarning,
    unsupported_reasoning_kind_message,
)


class GlmThinkingType(StrEnum):
    DISABLED = "disabled"
    ENABLED = "enabled"


class GlmRequestControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    extra_body: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_reasoning(
        cls,
        reasoning: ReasoningSpec | None,
    ) -> GlmRequestControls:
        if reasoning is None:
            return cls()
        match reasoning:
            case GlmReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(
                    extra_body={"thinking": {"type": GlmThinkingType.DISABLED}}
                )
            case GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE):
                return cls(
                    extra_body={"thinking": {"type": GlmThinkingType.ENABLED}}
                )
        raise ProviderSemanticError(
            unsupported_reasoning_kind_message(ProviderName.GLM, reasoning)
        )

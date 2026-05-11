from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec


class ReasoningControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    supported_thinking_levels: tuple[ThinkingLevel, ...]
    default_thinking_level: ThinkingLevel
    supported_effort_levels: tuple[EffortSpec, ...]
    default_effort: EffortSpec
    default_reasoning: ReasoningSpec | None

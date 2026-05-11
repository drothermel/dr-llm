from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

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
    default_reasoning: ReasoningSpec | None = None

    @model_validator(mode="after")
    def _validate_defaults(self) -> ReasoningControls:
        if self.default_thinking_level not in self.supported_thinking_levels:
            allowed = ", ".join(
                str(level) for level in self.supported_thinking_levels
            )
            raise ValueError(
                f"default_thinking_level={self.default_thinking_level!r} "
                f"must be one of supported_thinking_levels: {allowed}"
            )
        if (
            self.default_effort != EffortSpec.NA
            and self.default_effort not in self.supported_effort_levels
        ):
            allowed = ", ".join(
                str(effort) for effort in self.supported_effort_levels
            )
            raise ValueError(
                f"default_effort={self.default_effort!r} "
                f"must be one of supported_effort_levels: {allowed}"
            )
        return self

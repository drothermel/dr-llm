from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dr_llm.providers.models import Message
from dr_llm.providers.reasoning import ReasoningSpec, validate_reasoning


class LlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning: ReasoningSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reasoning(self) -> LlmRequest:
        validate_reasoning(
            provider=self.provider,
            model=self.model,
            reasoning=self.reasoning,
        )
        return self

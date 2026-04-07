from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.providers.effort import EffortSpec, validate_effort
from dr_llm.llm.request import LlmRequest, validate_max_tokens
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import ReasoningSpec, validate_reasoning


class LlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None

    @model_validator(mode="after")
    def _validate_reasoning(self) -> LlmConfig:
        validate_max_tokens(
            provider=self.provider,
            max_tokens=self.max_tokens,
        )
        validate_effort(
            provider=self.provider,
            model=self.model,
            effort=self.effort,
        )
        validate_reasoning(
            provider=self.provider,
            model=self.model,
            reasoning=self.reasoning,
        )
        return self

    def to_request(self, messages: list[Message]) -> LlmRequest:
        return LlmRequest(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
        )

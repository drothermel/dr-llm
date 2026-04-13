from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.providers.effort import EffortSpec
from dr_llm.llm.request import LlmRequest, validate_llm_constraints
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import ReasoningSpec


class LlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    temperature: float | None = 1.0
    top_p: float | None = 0.95
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None

    @model_validator(mode="after")
    def _validate_generation_params(self) -> LlmConfig:
        validate_llm_constraints(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
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

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dr_llm.llm.providers.effort import EffortSpec, validate_effort
from dr_llm.llm.messages import Message
from dr_llm.llm.providers.reasoning import ReasoningSpec
from dr_llm.llm.providers.reasoning_validation import validate_reasoning


def validate_max_tokens(*, provider: str, max_tokens: int | None) -> None:
    if provider in {"anthropic", "kimi-code"} and max_tokens is None:
        raise ValueError(f"max_tokens is required for provider={provider!r}")


def validate_llm_constraints(
    *,
    provider: str,
    model: str,
    max_tokens: int | None,
    effort: EffortSpec,
    reasoning: ReasoningSpec | None,
) -> None:
    validate_max_tokens(provider=provider, max_tokens=max_tokens)
    validate_effort(provider=provider, model=model, effort=effort)
    validate_reasoning(provider=provider, model=model, reasoning=reasoning)


class LlmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    messages: list[Message]
    temperature: float | None = 1.0
    top_p: float | None = 0.95
    max_tokens: int | None = None
    effort: EffortSpec = EffortSpec.NA
    reasoning: ReasoningSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_request_constraints(self) -> LlmRequest:
        validate_llm_constraints(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            reasoning=self.reasoning,
        )
        return self

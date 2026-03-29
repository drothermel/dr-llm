from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    effort: Literal["xhigh", "high", "medium", "low", "minimal", "none"] | None = None
    max_tokens: int | None = None
    exclude: bool | None = None
    enabled: bool | None = None

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("reasoning.max_tokens must be > 0")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self) -> ReasoningConfig:
        if self.effort is not None and self.max_tokens is not None:
            raise ValueError(
                "reasoning.effort and reasoning.max_tokens are mutually exclusive"
            )
        if self.enabled is False and (
            self.effort is not None or self.max_tokens is not None
        ):
            raise ValueError(
                "reasoning.enabled=false cannot be combined with effort or max_tokens"
            )
        return self

    @computed_field
    @property
    def effective_enabled(self) -> bool:
        if self.enabled is not None:
            return self.enabled
        return self.effort is not None or self.max_tokens is not None

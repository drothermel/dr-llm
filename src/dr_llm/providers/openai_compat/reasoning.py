from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.providers.models import CallMode, ReasoningWarning
from dr_llm.providers.reasoning import ReasoningConfig


class OpenAICompatReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningConfig | None,
        *,
        provider: str,
        mode: CallMode,
    ) -> OpenAICompatReasoningConfig:
        if config is None:
            return cls()
        return cls(
            payload=config.model_dump(
                mode="json",
                exclude_none=True,
                exclude_computed_fields=True,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return self.payload

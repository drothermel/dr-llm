from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.reasoning import ReasoningWarning
from dr_llm.providers.reasoning import ReasoningEffort, ReasoningOff, ReasoningSpec


class OpenAICompatReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> OpenAICompatReasoningConfig:
        if config is None:
            return cls()
        match config:
            case ReasoningEffort(level=level):
                return cls(payload={"effort": level})
            case ReasoningOff():
                return cls(payload={"effort": "none"})
            case _:
                raise ProviderSemanticError(
                    f"OpenAI-compatible reasoning serializer received unsupported config kind={config.kind!r}"
                )

    def to_payload(self) -> dict[str, Any]:
        return self.payload

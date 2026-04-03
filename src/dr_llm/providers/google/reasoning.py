from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.providers.reasoning import ReasoningWarning
from dr_llm.providers.reasoning import (
    GoogleReasoning,
    ReasoningBudget,
    ReasoningEffort,
    ReasoningOff,
    ReasoningSpec,
)


class GoogleReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
    ) -> GoogleReasoningConfig:
        if config is None:
            return cls()
        match config:
            case ReasoningOff():
                return cls(payload={"thinkingBudget": 0})
            case ReasoningBudget(tokens=tokens):
                return cls(payload={"thinkingBudget": tokens})
            case ReasoningEffort(level=level):
                return cls(payload={"thinkingLevel": level})
            case GoogleReasoning(
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                dynamic=dynamic,
            ):
                if thinking_level is not None:
                    return cls(payload={"thinkingLevel": thinking_level})
                if thinking_budget is not None:
                    return cls(payload={"thinkingBudget": thinking_budget})
                if dynamic:
                    return cls(payload={"thinkingBudget": -1})
                raise ProviderSemanticError(
                    "google reasoning config did not contain a serializable setting"
                )
            case _:
                raise ProviderSemanticError(
                    f"google reasoning serializer received unsupported config kind={config.kind!r}"
                )

    def to_payload(self) -> dict[str, Any]:
        return self.payload

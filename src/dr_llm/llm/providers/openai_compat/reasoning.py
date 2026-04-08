from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.reasoning import (
    GlmReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningWarning,
    ReasoningSpec,
    ThinkingLevel,
)


class OpenAICompatReasoningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ReasoningWarning] = Field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        config: ReasoningSpec | None,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> OpenAICompatReasoningConfig:
        if config is None:
            return cls()
        match config:
            case OpenAIReasoning(thinking_level=ThinkingLevel.NA):
                return cls()
            case OpenAIReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(reasoning_effort="none")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MINIMAL):
                return cls(reasoning_effort="minimal")
            case OpenAIReasoning(thinking_level=ThinkingLevel.LOW):
                return cls(reasoning_effort="low")
            case OpenAIReasoning(thinking_level=ThinkingLevel.MEDIUM):
                return cls(reasoning_effort="medium")
            case OpenAIReasoning(thinking_level=ThinkingLevel.HIGH):
                return cls(reasoning_effort="high")
            case GlmReasoning(thinking_level=ThinkingLevel.OFF):
                return cls(extra_body={"thinking": {"type": "disabled"}})
            case GlmReasoning(thinking_level=ThinkingLevel.ADAPTIVE):
                return cls(extra_body={"thinking": {"type": "enabled"}})
            case OpenRouterReasoning(enabled=enabled, effort=effort):
                if provider != "openrouter":
                    raise ProviderSemanticError(
                        "OpenRouter reasoning serializer requires provider='openrouter'"
                    )
                reasoning_payload: dict[str, Any]
                if enabled is not None:
                    reasoning_payload = {"enabled": enabled}
                elif effort is not None:
                    reasoning_payload = {"effort": effort}
                else:
                    raise ProviderSemanticError(
                        f"OpenRouter reasoning serializer received invalid config for model={model!r}"
                    )
                return cls(extra_body={"reasoning": reasoning_payload})
        raise ProviderSemanticError(
            f"OpenAI-compatible reasoning serializer received unsupported config kind={config.kind!r}"
        )

    def to_reasoning_effort(self) -> str | None:
        return self.reasoning_effort

    def to_extra_body(self) -> dict[str, Any]:
        return self.extra_body

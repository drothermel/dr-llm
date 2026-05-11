from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import (
    BaseProviderReasoningConfig,
    GlmReasoning,
    OpenAIReasoning,
    OpenRouterReasoning,
    ReasoningSpec,
    unsupported_reasoning_kind_message,
)


class OpenAICompatReasoningConfig(BaseProviderReasoningConfig):
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high"] | None
    ) = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

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
                if provider != ProviderName.OPENROUTER:
                    raise ProviderSemanticError(
                        f"OpenRouter reasoning serializer requires provider='{ProviderName.OPENROUTER}'"
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
            unsupported_reasoning_kind_message("OpenAI-compatible", config)
        )

from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import (
    _OPENAI_COMMON_MODELS,
    build_static_catalog_entries,
)
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    OpenAIReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.transports.openai_compat.orchestrator import (
    BaseOpenAICompatOrchestrator,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.providers.impls.openai.reasoning import (
    validate_reasoning_for_openai,
)
from dr_llm.llm.providers.impls.openai.thinking import (
    openai_supports_configurable_thinking,
    openai_supports_minimal_thinking,
    openai_supports_off_thinking,
    reasoning_capabilities_for_openai,
    validate_openai_sampling_controls,
)
from dr_llm.llm.request import LlmRequest


class OpenAIOrchestrator(BaseOpenAICompatOrchestrator):
    @property
    def name(self) -> ProviderName:
        return ProviderName.OPENAI

    def reasoning_capabilities(
        self, model: str
    ) -> ReasoningCapabilities | None:
        return reasoning_capabilities_for_openai(model)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        validate_reasoning_for_openai(
            model=request.model, reasoning=request.reasoning
        )
        validate_openai_sampling_controls(
            model=request.model,
            reasoning=request.reasoning,
            temperature=getattr(request, "temperature", None),
            top_p=getattr(request, "top_p", None),
        )
        return warnings

    def request_defaults(self, model: str) -> ProviderRequestDefaults:
        defaults = super().request_defaults(model)
        return defaults.model_copy(
            update={
                "temperature": None,
                "top_p": None,
            }
        )

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=_OPENAI_COMMON_MODELS,
            docs_url="https://platform.openai.com/docs/models",
            supports_vision=None,
            capabilities_fn=self.reasoning_capabilities,
        )

    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]:
        del capabilities
        if not openai_supports_configurable_thinking(model):
            return (ThinkingLevel.NA,)
        levels: list[ThinkingLevel] = []
        if openai_supports_off_thinking(model):
            levels.append(ThinkingLevel.OFF)
        elif openai_supports_minimal_thinking(model):
            levels.append(ThinkingLevel.MINIMAL)
        levels.extend(
            [ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH]
        )
        return tuple(levels)

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None:
        del model, budget_tokens
        if thinking_level == ThinkingLevel.NA:
            return None
        return OpenAIReasoning(thinking_level=thinking_level)

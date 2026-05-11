from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import (
    _GLM_COMMON_MODELS,
    build_static_catalog_entries,
)
from dr_llm.llm.names import ProviderName, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    GlmReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.impls.glm.capabilities import (
    reasoning_capabilities_for_glm,
)
from dr_llm.llm.providers.transports.openai_compat.orchestrator import (
    BaseOpenAICompatOrchestrator,
)
from dr_llm.llm.providers.impls.glm.reasoning import validate_reasoning_for_glm
from dr_llm.llm.request import LlmRequest


class GlmOrchestrator(BaseOpenAICompatOrchestrator):
    @property
    def name(self) -> ProviderName:
        return ProviderName.GLM

    def reasoning_capabilities(
        self, model: str
    ) -> ReasoningCapabilities | None:
        return reasoning_capabilities_for_glm(model)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        validate_reasoning_for_glm(
            model=request.model, reasoning=request.reasoning
        )
        return warnings

    def _supported_thinking_levels(
        self, *, model: str, capabilities: ModelCapabilities
    ) -> tuple[ThinkingLevel, ...]:
        reasoning = capabilities.reasoning
        if reasoning.mode == ReasoningMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if reasoning.mode == ReasoningMode.GLM:
            return (ThinkingLevel.OFF, ThinkingLevel.ADAPTIVE)
        raise ValueError(
            f"unexpected reasoning mode for provider={self.name!r} "
            f"model={model!r}: {reasoning.mode!r}"
        )

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
        return GlmReasoning(thinking_level=thinking_level)

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=_GLM_COMMON_MODELS,
            docs_url="https://docs.z.ai/guides/llm/glm-4.5",
            supports_vision=None,
            capabilities_fn=self.reasoning_capabilities,
        )

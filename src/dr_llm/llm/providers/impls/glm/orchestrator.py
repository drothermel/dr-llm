from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
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
from dr_llm.llm.providers.impls.glm.controls import (
    reasoning_capabilities_for_glm,
    validate_reasoning_for_glm,
)
from dr_llm.llm.providers.impls.glm.families import (
    GlmStaticCatalogModel,
)
from dr_llm.llm.providers.impls.openai_compat_base import (
    BaseOpenAICompatOrchestrator,
)
from dr_llm.llm.providers.core.orchestrator_base import CatalogResult
from dr_llm.llm.providers.impls.glm.provider import GlmProvider, GlmUrls
from dr_llm.llm.request import LlmRequest


class GlmOrchestrator(BaseOpenAICompatOrchestrator):
    def __init__(self, provider: GlmProvider | None = None) -> None:
        super().__init__(provider or GlmProvider())

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

    def supported_thinking_levels(
        self,
        model: str,
        *,
        capabilities: ModelCapabilities | None = None,
    ) -> tuple[ThinkingLevel, ...]:
        resolved_capabilities = (
            self.model_capabilities(model)
            if capabilities is None
            else capabilities
        )
        reasoning = resolved_capabilities.reasoning
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

    def fallback_models(self) -> CatalogResult:
        return build_static_catalog_entries(
            provider=self._provider,
            models=GlmStaticCatalogModel.values(),
            docs_url=GlmUrls.MODELS_DOCS,
            supports_vision=None,
            capabilities_fn=self.reasoning_capabilities,
        )

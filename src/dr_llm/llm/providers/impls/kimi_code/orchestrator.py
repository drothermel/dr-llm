from __future__ import annotations

from dr_llm.llm.catalog.fetchers.kimi import fetch_kimi_models
from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName, ReasoningMode, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    build_model_capabilities,
)
from dr_llm.llm.providers.concepts.reasoning import (
    AnthropicReasoning,
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.impls.kimi_code.controls import (
    reasoning_capabilities_for_kimi_code,
    supported_effort_levels_for_kimi_code,
    validate_reasoning_for_kimi_code,
)
from dr_llm.llm.providers.impls.kimi_code.families import (
    KimiCodeStaticCatalogModel,
)
from dr_llm.llm.providers.impls.kimi_code.provider import (
    KimiCodeProvider,
    KimiCodeUrls,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.request import LlmRequest


class KimiCodeOrchestrator(BaseProviderOrchestrator):
    _provider: KimiCodeProvider

    def __init__(self, provider: KimiCodeProvider | None = None) -> None:
        super().__init__(provider or KimiCodeProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.KIMI_CODE

    def model_capabilities(self, model: str) -> ModelCapabilities:
        return build_model_capabilities(
            reasoning=reasoning_capabilities_for_kimi_code(model),
            supported_effort_levels=supported_effort_levels_for_kimi_code(
                model
            ),
        )

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
        if reasoning.mode == ReasoningMode.KIMI_CODE_EFFORT_AND_BUDGET:
            return (
                ThinkingLevel.OFF,
                ThinkingLevel.ADAPTIVE,
                ThinkingLevel.BUDGET,
            )
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
        del model
        if thinking_level == ThinkingLevel.NA:
            return None
        if thinking_level == ThinkingLevel.BUDGET:
            return AnthropicReasoning(
                thinking_level=thinking_level,
                budget_tokens=self._require_budget_tokens(budget_tokens),
            )
        return AnthropicReasoning(thinking_level=thinking_level)

    def validate_request(self, request: LlmRequest) -> list[ReasoningWarning]:
        warnings = super().validate_request(request)
        self._validate_max_tokens_required(request)
        validate_reasoning_for_kimi_code(
            model=request.model, reasoning=request.reasoning
        )
        return warnings

    def request_defaults(self, model: str) -> ProviderRequestDefaults:
        defaults = super().request_defaults(model)
        return defaults.model_copy(
            update={
                "max_tokens": 16384,
                "max_tokens_required": True,
            }
        )

    def fetch_models(self):
        return fetch_kimi_models(
            self._provider,
            capabilities_fn=reasoning_capabilities_for_kimi_code,
        )

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=KimiCodeStaticCatalogModel.values(),
            docs_url=KimiCodeUrls.MODELS_DOCS,
            supports_vision=True,
            capabilities_fn=reasoning_capabilities_for_kimi_code,
        )

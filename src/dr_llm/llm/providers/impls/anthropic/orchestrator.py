from __future__ import annotations

from dr_llm.llm.catalog.fetchers.anthropic import fetch_anthropic_models
from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.anthropic.controls import (
    AnthropicControls,
)
from dr_llm.llm.providers.impls.anthropic.static_catalog import (
    AnthropicStaticCatalogModel,
)
from dr_llm.llm.providers.impls.anthropic.provider import (
    AnthropicProvider,
    AnthropicUrls,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)


class AnthropicOrchestrator(BaseProviderOrchestrator):
    _provider: AnthropicProvider

    def __init__(self, provider: AnthropicProvider | None = None) -> None:
        super().__init__(provider or AnthropicProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.ANTHROPIC

    def controls(self, model: str) -> AnthropicControls:
        return AnthropicControls(model=model, mode=self.mode)

    def fetch_models(self):
        return fetch_anthropic_models(
            self._provider,
            controls_fn=self.controls,
        )

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=AnthropicStaticCatalogModel.values(),
            docs_url=AnthropicUrls.MODELS_DOCS,
            supports_vision=True,
            controls_fn=self.controls,
        )

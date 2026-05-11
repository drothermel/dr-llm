from __future__ import annotations

from dr_llm.llm.catalog.fetchers.google import fetch_google_models
from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.google.controls import (
    GoogleControls,
)
from dr_llm.llm.providers.impls.google.static_catalog import (
    GoogleStaticCatalogModel,
)
from dr_llm.llm.providers.impls.google.provider import (
    GoogleProvider,
    GoogleUrls,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)


class GoogleOrchestrator(BaseProviderOrchestrator):
    _provider: GoogleProvider

    def __init__(self, provider: GoogleProvider | None = None) -> None:
        super().__init__(provider or GoogleProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.GOOGLE

    def controls(self, model: str) -> GoogleControls:
        return GoogleControls(model=model, mode=self.mode)

    def fetch_models(self):
        return fetch_google_models(
            self._provider,
            controls_fn=self.controls,
        )

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=GoogleStaticCatalogModel.values(),
            docs_url=GoogleUrls.MODELS_DOCS,
            supports_vision=None,
            controls_fn=self.controls,
        )

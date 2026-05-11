from __future__ import annotations

from dr_llm.llm.catalog.fetchers.kimi import fetch_kimi_models
from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.kimi_code.controls import (
    KimiCodeControls,
)
from dr_llm.llm.providers.impls.kimi_code.static_catalog import (
    KimiCodeStaticCatalogModel,
)
from dr_llm.llm.providers.impls.kimi_code.provider import (
    KimiCodeProvider,
    KimiCodeUrls,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)


class KimiCodeOrchestrator(BaseProviderOrchestrator):
    _provider: KimiCodeProvider

    def __init__(self, provider: KimiCodeProvider | None = None) -> None:
        super().__init__(provider or KimiCodeProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.KIMI_CODE

    def controls(self, model: str) -> KimiCodeControls:
        return KimiCodeControls(model=model, mode=self.mode)

    def fetch_models(self):
        return fetch_kimi_models(
            self._provider,
            controls_fn=self.controls,
        )

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=KimiCodeStaticCatalogModel.values(),
            docs_url=KimiCodeUrls.MODELS_DOCS,
            supports_vision=True,
            controls_fn=self.controls,
        )

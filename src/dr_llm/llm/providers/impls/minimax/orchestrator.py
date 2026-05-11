from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.minimax.controls import (
    MiniMaxControls,
)
from dr_llm.llm.providers.impls.minimax.static_catalog import (
    MiniMaxStaticCatalogModel,
)
from dr_llm.llm.providers.impls.minimax.provider import (
    MiniMaxProvider,
    MiniMaxUrls,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)


class MiniMaxOrchestrator(BaseProviderOrchestrator):
    _provider: MiniMaxProvider

    def __init__(self, provider: MiniMaxProvider | None = None) -> None:
        super().__init__(provider or MiniMaxProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.MINIMAX

    def controls(self, model: str) -> MiniMaxControls:
        return MiniMaxControls(model=model, mode=self.mode)

    def fetch_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=MiniMaxStaticCatalogModel.values(),
            docs_url=MiniMaxUrls.MODELS_DOCS,
            supports_vision=None,
            controls_fn=self.controls,
        )

    def fallback_models(self):
        return self.fetch_models()

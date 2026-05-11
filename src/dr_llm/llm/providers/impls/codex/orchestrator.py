from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.codex.controls import (
    CodexControls,
)
from dr_llm.llm.providers.impls.codex.static_catalog import (
    CodexStaticCatalogModel,
)
from dr_llm.llm.providers.impls.codex.provider import (
    CodexProvider,
    CodexUrls,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)


class CodexOrchestrator(BaseProviderOrchestrator):
    _provider: CodexProvider

    def __init__(self, provider: CodexProvider | None = None) -> None:
        super().__init__(provider or CodexProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.CODEX

    def controls(self, model: str) -> CodexControls:
        return CodexControls(model=model, mode=self.mode)

    def fetch_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=CodexStaticCatalogModel.values(),
            docs_url=CodexUrls.MODELS_DOCS,
            supports_vision=None,
            controls_fn=self.controls,
        )

    def fallback_models(self):
        return self.fetch_models()

from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.glm.controls import (
    GlmControls,
)
from dr_llm.llm.providers.impls.glm.static_catalog import (
    GlmStaticCatalogModel,
)
from dr_llm.llm.providers.impls.openai_compat_base import (
    BaseOpenAICompatOrchestrator,
)
from dr_llm.llm.providers.core.orchestrator_base import CatalogResult
from dr_llm.llm.providers.impls.glm.provider import GlmProvider, GlmUrls


class GlmOrchestrator(BaseOpenAICompatOrchestrator):
    def __init__(self, provider: GlmProvider | None = None) -> None:
        super().__init__(provider or GlmProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.GLM

    def controls(self, model: str) -> GlmControls:
        return GlmControls(model=model, mode=self.mode)

    def fallback_models(self) -> CatalogResult:
        return build_static_catalog_entries(
            provider=self._provider,
            models=GlmStaticCatalogModel.values(),
            docs_url=GlmUrls.MODELS_DOCS,
            supports_vision=None,
            controls_fn=self.controls,
        )

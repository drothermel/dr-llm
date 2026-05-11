from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.openai.static_catalog import (
    OpenAIStaticCatalogModel,
)
from dr_llm.llm.providers.impls.openai_compat_base import (
    BaseOpenAICompatOrchestrator,
)
from dr_llm.llm.providers.impls.openai.controls import (
    OpenAIControls,
)
from dr_llm.llm.providers.impls.openai.provider import (
    OpenAIProvider,
    OpenAIUrls,
)


class OpenAIOrchestrator(BaseOpenAICompatOrchestrator):
    def __init__(self, provider: OpenAIProvider | None = None) -> None:
        super().__init__(provider or OpenAIProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.OPENAI

    def controls(self, model: str) -> OpenAIControls:
        return OpenAIControls(model=model, mode=self.mode)

    def fallback_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=OpenAIStaticCatalogModel.values(),
            docs_url=OpenAIUrls.MODELS_DOCS,
            supports_vision=None,
            controls_fn=self.controls,
        )

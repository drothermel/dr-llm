from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.claude_code.controls import (
    ClaudeCodeControls,
)
from dr_llm.llm.providers.impls.claude_code.static_catalog import (
    ClaudeCodeStaticCatalogModel,
)
from dr_llm.llm.providers.impls.claude_code.provider import (
    ClaudeCodeUrls,
    ClaudeCodeProvider,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
)


class ClaudeCodeOrchestrator(BaseProviderOrchestrator):
    _provider: ClaudeCodeProvider

    def __init__(self, provider: ClaudeCodeProvider | None = None) -> None:
        super().__init__(provider or ClaudeCodeProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.CLAUDE_CODE

    def controls(self, model: str) -> ClaudeCodeControls:
        return ClaudeCodeControls(model=model, mode=self.mode)

    def fetch_models(self):
        return build_static_catalog_entries(
            provider=self._provider,
            models=ClaudeCodeStaticCatalogModel.values(),
            docs_url=ClaudeCodeUrls.MODELS_DOCS,
            supports_vision=True,
            controls_fn=self.controls,
        )

    def fallback_models(self):
        return self.fetch_models()

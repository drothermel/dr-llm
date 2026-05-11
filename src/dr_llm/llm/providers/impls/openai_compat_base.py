from __future__ import annotations

from dr_llm.llm.catalog.fetchers.openai_compat import (
    fetch_openai_compat_models,
)
from dr_llm.llm.providers.core.controls import ProviderControls
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
    CatalogResult,
)
from dr_llm.llm.providers.transports.api_provider import ApiProvider


class BaseOpenAICompatOrchestrator(BaseProviderOrchestrator):
    _provider: ApiProvider

    def __init__(self, provider: ApiProvider) -> None:
        super().__init__(provider)

    def fetch_models(self) -> CatalogResult:
        return fetch_openai_compat_models(
            self._provider,
            controls_fn=self.controls,
        )

    def controls(self, model: str) -> ProviderControls:
        raise NotImplementedError

from __future__ import annotations

from dr_llm.llm.catalog.fetchers.static import build_static_catalog_entries
from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.impls.openai_compat_base import (
    BaseOpenAICompatOrchestrator,
)
from dr_llm.llm.providers.impls.openrouter.controls import (
    OpenRouterControls,
    apply_openrouter_model_policies,
)
from dr_llm.llm.providers.impls.openrouter.families import (
    OPENROUTER_FAMILIES,
)
from dr_llm.llm.providers.impls.openrouter.provider import (
    OpenRouterProvider,
    OpenRouterUrls,
)


class OpenRouterOrchestrator(BaseOpenAICompatOrchestrator):
    def __init__(self, provider: OpenRouterProvider | None = None) -> None:
        super().__init__(provider or OpenRouterProvider())

    @property
    def name(self) -> ProviderName:
        return ProviderName.OPENROUTER

    def controls(self, model: str) -> OpenRouterControls:
        return OpenRouterControls(model=model, mode=self.mode)

    def fetch_models(self):
        entries, raw_payload = super().fetch_models()
        return apply_openrouter_model_policies(entries), raw_payload

    def fallback_models(self):
        entries, raw_payload = build_static_catalog_entries(
            provider=self._provider,
            models=OPENROUTER_FAMILIES.allowed_models(),
            docs_url=OpenRouterUrls.MODELS_DOCS,
            supports_vision=None,
            controls_fn=self.controls,
        )
        return apply_openrouter_model_policies(entries), raw_payload

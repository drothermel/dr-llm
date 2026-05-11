from __future__ import annotations

from abc import abstractmethod

from dr_llm.llm.catalog.fetchers.openai_compat import (
    fetch_openai_compat_models,
)
from dr_llm.llm.config import SamplingControls
from dr_llm.llm.providers.concepts.capabilities import (
    ModelCapabilities,
    ReasoningCapabilities,
    build_model_capabilities,
)
from dr_llm.llm.providers.core.orchestrator_base import (
    BaseProviderOrchestrator,
    CatalogResult,
)
from dr_llm.llm.providers.core.request_defaults import (
    ProviderRequestDefaults,
)
from dr_llm.llm.providers.transports.api_provider import ApiProvider


class BaseOpenAICompatOrchestrator(BaseProviderOrchestrator):
    _provider: ApiProvider

    def __init__(self, provider: ApiProvider) -> None:
        super().__init__(provider)

    def model_capabilities(self, model: str) -> ModelCapabilities:
        return build_model_capabilities(
            reasoning=self.reasoning_capabilities(model)
        )

    def fetch_models(self) -> CatalogResult:
        return fetch_openai_compat_models(
            self._provider,
            capabilities_fn=self.reasoning_capabilities,
        )

    def request_defaults(self, model: str) -> ProviderRequestDefaults:
        defaults = super().request_defaults(model)
        return defaults.model_copy(
            update={
                "sampling_supported": True,
                "sampling": SamplingControls(temperature=1.0, top_p=0.95),
            }
        )

    @abstractmethod
    def reasoning_capabilities(
        self, model: str
    ) -> ReasoningCapabilities | None: ...

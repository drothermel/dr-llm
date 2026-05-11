from __future__ import annotations

from typing import Any, Protocol

from dr_llm.llm.catalog.models import ModelCatalogEntry
from dr_llm.llm.names import EffortSpec, ThinkingLevel
from dr_llm.llm.providers.concepts.capabilities import ModelCapabilities
from dr_llm.llm.providers.concepts.reasoning import (
    ReasoningSpec,
    ReasoningWarning,
)
from dr_llm.llm.providers.core.config import ProviderAvailabilityStatus
from dr_llm.llm.providers.core.reasoning_controls import ReasoningControls
from dr_llm.llm.providers.core.request_defaults import ProviderRequestDefaults
from dr_llm.llm.request import LlmRequest, Message
from dr_llm.llm.response import CallMode, LlmResponse


class ProviderOrchestrator(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def mode(self) -> CallMode: ...

    def model_capabilities(self, model: str) -> ModelCapabilities: ...

    def reasoning_controls(self, model: str) -> ReasoningControls: ...

    def request_defaults(self, model: str) -> ProviderRequestDefaults: ...

    def build_request(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        effort: EffortSpec = EffortSpec.NA,
        reasoning: ReasoningSpec | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LlmRequest: ...

    def reasoning_for_thinking_level(
        self,
        *,
        model: str,
        thinking_level: ThinkingLevel,
        budget_tokens: int | None = None,
    ) -> ReasoningSpec | None: ...

    def validate_request(
        self, request: LlmRequest
    ) -> list[ReasoningWarning]: ...

    def generate(self, request: LlmRequest) -> LlmResponse: ...

    def fetch_models(
        self,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]: ...

    def fallback_models(
        self,
    ) -> tuple[list[ModelCatalogEntry], dict[str, Any]]: ...

    def availability_status(self) -> ProviderAvailabilityStatus: ...

    def is_available(self) -> bool: ...

    def close(self) -> None: ...

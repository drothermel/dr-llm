from __future__ import annotations

from typing import Protocol

from dr_llm.llm.names import ProviderName
from dr_llm.llm.providers.concepts.capabilities import ModelCapabilities
from dr_llm.llm.providers.concepts.reasoning import ReasoningWarning
from dr_llm.llm.request import LlmRequest
from dr_llm.llm.response import LlmResponse


class ProviderOrchestrator(Protocol):
    name: ProviderName

    def model_capabilities(self, model: str) -> ModelCapabilities: ...

    def validate_request(
        self, request: LlmRequest
    ) -> list[ReasoningWarning]: ...

    def generate(self, request: LlmRequest) -> LlmResponse: ...

    def is_available(self) -> bool: ...

    def close(self) -> None: ...

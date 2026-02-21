from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from llm_pool.types import LlmRequest, LlmResponse


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    supports_native_tools: bool = False
    supports_structured_output: bool = False


class ProviderAdapter(ABC):
    """Abstract provider adapter interface."""

    name: str
    mode: str

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    @abstractmethod
    def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_pool.types import LlmRequest, LlmResponse


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
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

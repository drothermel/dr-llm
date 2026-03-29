from __future__ import annotations

from abc import ABC, abstractmethod

from dr_llm.generation.models import LlmRequest, LlmResponse

from dr_llm.providers.provider_config import (
    ProviderAvailabilityStatus,
    ProviderConfig,
)


class ProviderAdapter(ABC):
    """Abstract provider adapter interface."""

    _config: ProviderConfig

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def mode(self) -> str:
        return self._config.mode

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def availability_status(self) -> ProviderAvailabilityStatus:
        return self._config.availability_status()

    def is_available(self) -> bool:
        return self.availability_status().available

    @abstractmethod
    def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError

    def close(self) -> None:
        """Release any provider-owned resources."""

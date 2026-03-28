from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from dr_llm.catalog.models import ModelCatalogEntry
from dr_llm.generation.models import LlmRequest, LlmResponse
from pydantic import BaseModel, ConfigDict, Field


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    supports_structured_output: bool = False


class ProviderRuntimeRequirements(BaseModel):
    model_config = ConfigDict(frozen=True)

    required_env_vars: list[str] = Field(default_factory=list)
    required_executables: list[str] = Field(default_factory=list)


class ProviderAdapter(ABC):
    """Abstract provider adapter interface."""

    name: str
    mode: str

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    @property
    @abstractmethod
    def runtime_requirements(self) -> ProviderRuntimeRequirements:
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError


class ProviderModelCatalogAdapter(Protocol):
    name: str

    def list_models(self) -> list[ModelCatalogEntry]: ...

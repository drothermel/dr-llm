from __future__ import annotations

from abc import ABC, abstractmethod
import os
import shutil
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

    def missing_env_vars(self) -> tuple[str, ...]:
        return tuple(
            env_var for env_var in self.required_env_vars if not os.getenv(env_var)
        )

    def missing_executables(self) -> tuple[str, ...]:
        return tuple(
            executable
            for executable in self.required_executables
            if shutil.which(executable) is None
        )


class ProviderAvailabilityStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    available: bool
    missing_env_vars: tuple[str, ...] = Field(default_factory=tuple)
    missing_executables: tuple[str, ...] = Field(default_factory=tuple)
    supports_structured_output: bool = False


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

    def availability_status(self) -> ProviderAvailabilityStatus:
        requirements = self.runtime_requirements
        missing_env_vars = requirements.missing_env_vars()
        missing_executables = requirements.missing_executables()
        return ProviderAvailabilityStatus(
            provider=self.name,
            available=not missing_env_vars and not missing_executables,
            missing_env_vars=missing_env_vars,
            missing_executables=missing_executables,
            supports_structured_output=self.capabilities.supports_structured_output,
        )

    def is_available(self) -> bool:
        return self.availability_status().available

    @abstractmethod
    def generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError

    def close(self) -> None:
        """Release any provider-owned resources."""


class ProviderModelCatalogAdapter(Protocol):
    name: str

    def list_models(self) -> list[ModelCatalogEntry]: ...

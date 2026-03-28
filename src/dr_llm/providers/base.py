from __future__ import annotations

from abc import ABC, abstractmethod
import os
import shutil
from typing import Self

from dr_llm.generation.models import LlmRequest, LlmResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    mode: str = "api"
    supports_structured_output: bool = False
    required_env_vars: list[str] = Field(default_factory=list)
    required_executables: list[str] = Field(default_factory=list)
    timeout_seconds: float = 120.0

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

    def availability_status(self) -> ProviderAvailabilityStatus:
        missing_env = self.missing_env_vars()
        missing_exec = self.missing_executables()
        return ProviderAvailabilityStatus(
            provider=self.name,
            available=not missing_env and not missing_exec,
            missing_env_vars=missing_env,
            missing_executables=missing_exec,
            supports_structured_output=self.supports_structured_output,
        )


class ProviderAvailabilityStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    available: bool
    missing_env_vars: tuple[str, ...] = Field(default_factory=tuple)
    missing_executables: tuple[str, ...] = Field(default_factory=tuple)
    supports_structured_output: bool = False


class APIProviderConfig(ProviderConfig):
    mode: str = "api"
    supports_structured_output: bool = True
    base_url: str
    api_key_env: str
    api_key: str | None = None

    @model_validator(mode="after")
    def _compute_api_env_requirements(self) -> Self:
        if not self.api_key and self.api_key_env not in self.required_env_vars:
            object.__setattr__(
                self, "required_env_vars", [*self.required_env_vars, self.api_key_env]
            )
        return self


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

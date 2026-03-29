from __future__ import annotations

import os
import shutil

from pydantic import BaseModel, ConfigDict, Field


class ProviderAvailabilityStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    available: bool
    missing_env_vars: tuple[str, ...] = Field(default_factory=tuple)
    missing_executables: tuple[str, ...] = Field(default_factory=tuple)
    supports_structured_output: bool = False


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

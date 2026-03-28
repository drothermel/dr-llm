from __future__ import annotations

import os
import shutil

from pydantic import BaseModel, ConfigDict, Field

from dr_llm.providers.registry import ProviderRegistry


class ProviderAvailabilityStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    available: bool
    missing_env_vars: tuple[str, ...] = Field(default_factory=tuple)
    missing_executables: tuple[str, ...] = Field(default_factory=tuple)
    supports_native_tools: bool = False
    supports_structured_output: bool = False


def supported_provider_names(registry: ProviderRegistry) -> list[str]:
    return sorted(registry.names())


def supported_provider_statuses(
    registry: ProviderRegistry,
) -> list[ProviderAvailabilityStatus]:
    statuses: list[ProviderAvailabilityStatus] = []
    for provider in supported_provider_names(registry):
        adapter = registry.get(provider)
        requirements = adapter.runtime_requirements
        missing_env_vars = [
            env_var
            for env_var in requirements.required_env_vars
            if not os.getenv(env_var)
        ]
        missing_executables = [
            executable
            for executable in requirements.required_executables
            if shutil.which(executable) is None
        ]
        capabilities = adapter.capabilities
        statuses.append(
            ProviderAvailabilityStatus(
                provider=provider,
                available=not missing_env_vars and not missing_executables,
                missing_env_vars=tuple(missing_env_vars),
                missing_executables=tuple(missing_executables),
                supports_native_tools=capabilities.supports_native_tools,
                supports_structured_output=capabilities.supports_structured_output,
            )
        )
    return statuses


def available_provider_names(
    registry: ProviderRegistry,
    *,
    statuses: list[ProviderAvailabilityStatus] | None = None,
) -> list[str]:
    if statuses is None:
        statuses = supported_provider_statuses(registry)
    return [status.provider for status in statuses if status.available]

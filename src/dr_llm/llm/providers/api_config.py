from __future__ import annotations

import os
from typing import Self

from pydantic import model_validator

from dr_llm.errors import ProviderSemanticError
from dr_llm.llm.providers.config import ProviderConfig


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


def resolve_api_key(config: APIProviderConfig, *, label: str | None = None) -> str:
    """Resolve a provider's API key from inline config or env, raising on miss."""
    key = config.api_key or os.getenv(config.api_key_env)
    if not key:
        provider_label = label or config.name
        raise ProviderSemanticError(
            f"Missing API key for {provider_label}. Set {config.api_key_env} or pass config.api_key"
        )
    return key

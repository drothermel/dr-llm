from __future__ import annotations

from typing import Self

from pydantic import model_validator

from dr_llm.providers.provider_config import ProviderConfig


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

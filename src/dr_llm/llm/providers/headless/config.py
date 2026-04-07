from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from dr_llm.llm.providers.config import ProviderConfig


class HeadlessProviderConfig(ProviderConfig):
    mode: str = "headless"
    supports_structured_output: bool = True
    timeout_seconds: float = 180.0
    command: list[str]
    env_overrides: dict[str, str] = Field(default_factory=dict)
    log_full_io: bool = False
    redact_io: bool = True
    max_logged_chars: int = 512

    @model_validator(mode="after")
    def _compute_headless_exec_requirements(self) -> Self:
        if self.command:
            object.__setattr__(
                self,
                "required_executables",
                [*self.required_executables, self.command[0]],
            )
        return self


class ClaudeHeadlessProviderConfig(HeadlessProviderConfig):
    api_key_env: str | None = None

    @model_validator(mode="after")
    def _compute_claude_env_requirements(self) -> Self:
        if self.api_key_env:
            object.__setattr__(
                self,
                "required_env_vars",
                [*self.required_env_vars, self.api_key_env],
            )
        return self

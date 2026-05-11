from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig
from dr_llm.llm.names import EffortSpec, ProviderName, ThinkingLevel
from dr_llm.llm.providers.core.authoring import build_provider_config
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.anthropic.capabilities import (
    anthropic_supports_adaptive_thinking,
)
from dr_llm.llm.providers.impls.claude_code.capabilities import (
    supported_effort_levels_for_claude_code,
)
from dr_llm.llm.providers.impls.claude_code.families import (
    CLAUDE_CODE_SUPPORTED_MODEL_FAMILIES,
    ClaudeCodeModelFamily,
)

type _ClaudeCodeEffort = Literal[
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
    EffortSpec.MAX,
]


class _ClaudeCodeBaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.CLAUDE_CODE] = ProviderName.CLAUDE_CODE
    model: str

    @model_validator(mode="after")
    def _validate_model_family(self) -> _ClaudeCodeBaseConfig:
        if _is_claude_code_model(self.model):
            return self
        raise ValueError(
            f"{type(self).__name__} only supports provider={self.provider!r} "
            f"model family={str(ClaudeCodeModelFamily.CLAUDE)!r}; "
            f"got model={self.model!r}"
        )

    def _effort(self) -> EffortSpec | None:
        return None

    def _thinking_level(self) -> ThinkingLevel | None:
        return None

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            effort=self._effort(),
            thinking_level=self._thinking_level(),
            registry=registry,
        )


class ClaudeCodeLegacyConfig(_ClaudeCodeBaseConfig):
    @model_validator(mode="after")
    def _validate_legacy(self) -> ClaudeCodeLegacyConfig:
        if _is_adaptive_model(self.model):
            raise ValueError(
                f"ClaudeCodeLegacyConfig does not support adaptive model "
                f"{self.model!r}; use ClaudeCodeAdaptiveConfig"
            )
        if supported_effort_levels_for_claude_code(self.model):
            raise ValueError(
                f"ClaudeCodeLegacyConfig does not support effort model "
                f"{self.model!r}; use ClaudeCodeEffortConfig"
            )
        return self


class ClaudeCodeAdaptiveConfig(_ClaudeCodeBaseConfig):
    thinking_level: Literal[ThinkingLevel.ADAPTIVE] = ThinkingLevel.ADAPTIVE

    @model_validator(mode="after")
    def _validate_adaptive(self) -> ClaudeCodeAdaptiveConfig:
        if _is_adaptive_model(self.model):
            return self
        raise ValueError(
            f"ClaudeCodeAdaptiveConfig only supports Anthropic adaptive "
            f"thinking models; got model={self.model!r}"
        )

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class ClaudeCodeEffortConfig(_ClaudeCodeBaseConfig):
    effort: _ClaudeCodeEffort | None = None

    @model_validator(mode="after")
    def _validate_effort(self) -> ClaudeCodeEffortConfig:
        allowed = supported_effort_levels_for_claude_code(self.model)
        if not allowed:
            raise ValueError(
                f"ClaudeCodeEffortConfig does not support model={self.model!r}"
            )
        if self.effort is None or self.effort in allowed:
            return self
        allowed_values = ", ".join(str(level) for level in allowed)
        raise ValueError(
            f"ClaudeCodeEffortConfig effort={str(self.effort)!r} is not "
            f"supported for provider={self.provider!r} model={self.model!r}; "
            f"allowed levels: {allowed_values}"
        )

    def _effort(self) -> EffortSpec | None:
        return self.effort


def _is_claude_code_model(model: str) -> bool:
    return any(
        model.startswith(family)
        for family in CLAUDE_CODE_SUPPORTED_MODEL_FAMILIES
    )


def _is_adaptive_model(model: str) -> bool:
    return anthropic_supports_adaptive_thinking(model)

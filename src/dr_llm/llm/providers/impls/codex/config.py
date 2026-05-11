from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.core.authoring import (
    build_provider_config,
    reject_model_family,
    require_model_family,
)
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.codex.controls import (
    CODEX_THINKING_SUPPORTED_MODELS,
    CodexModelFamily,
)

_CODEX_GPT5_FAMILIES = (CodexModelFamily.GPT5,)
_CODEX_GPT51_FAMILIES = (CodexModelFamily.GPT51,)
_CODEX_GPT52_FAMILIES = (CodexModelFamily.GPT52,)
_CODEX_GPT54_FAMILIES = (
    CodexModelFamily.GPT54,
    CodexModelFamily.GPT54_MINI,
)
_CODEX_GPT5_CODEX_FAMILIES = (
    CodexModelFamily.GPT5_CODEX,
    CodexModelFamily.GPT51_CODEX,
    CodexModelFamily.GPT51_CODEX_MINI,
    CodexModelFamily.GPT51_CODEX_MAX,
    CodexModelFamily.GPT52_CODEX,
    CodexModelFamily.GPT53_CODEX,
    CodexModelFamily.GPT53_CODEX_SPARK,
)

type _CodexMinimalThinkingLevel = Literal[
    ThinkingLevel.MINIMAL,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
    ThinkingLevel.XHIGH,
]
type _CodexOffThinkingLevel = Literal[
    ThinkingLevel.OFF,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
    ThinkingLevel.XHIGH,
]
type _CodexCodexThinkingLevel = Literal[
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
    ThinkingLevel.XHIGH,
]


class _CodexBaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.CODEX] = ProviderName.CODEX
    model: str

    _families: ClassVar[tuple[CodexModelFamily, ...]] = ()

    @model_validator(mode="after")
    def _validate_model_family(self) -> _CodexBaseConfig:
        require_model_family(
            provider=self.provider,
            model=self.model,
            families=self._families,
            config_name=type(self).__name__,
        )
        return self

    def _thinking_level(self) -> ThinkingLevel | None:
        return None

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            thinking_level=self._thinking_level(),
            registry=registry,
        )


class CodexLegacyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.CODEX] = ProviderName.CODEX
    model: str

    @model_validator(mode="after")
    def _validate_model_family(self) -> CodexLegacyConfig:
        reject_model_family(
            provider=self.provider,
            model=self.model,
            families=CODEX_THINKING_SUPPORTED_MODELS,
            config_name=type(self).__name__,
        )
        return self

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            registry=registry,
        )


class CodexGpt5Config(_CodexBaseConfig):
    _families: ClassVar[tuple[CodexModelFamily, ...]] = _CODEX_GPT5_FAMILIES

    thinking_level: _CodexMinimalThinkingLevel | None = None

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class CodexGpt51Config(_CodexBaseConfig):
    _families: ClassVar[tuple[CodexModelFamily, ...]] = _CODEX_GPT51_FAMILIES

    thinking_level: _CodexOffThinkingLevel | None = None

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class CodexGpt52Config(_CodexBaseConfig):
    _families: ClassVar[tuple[CodexModelFamily, ...]] = _CODEX_GPT52_FAMILIES

    thinking_level: _CodexOffThinkingLevel | None = None

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class CodexGpt54Config(_CodexBaseConfig):
    _families: ClassVar[tuple[CodexModelFamily, ...]] = _CODEX_GPT54_FAMILIES

    thinking_level: _CodexOffThinkingLevel | None = None

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class CodexGpt5CodexConfig(_CodexBaseConfig):
    _families: ClassVar[tuple[CodexModelFamily, ...]] = (
        _CODEX_GPT5_CODEX_FAMILIES
    )

    thinking_level: _CodexCodexThinkingLevel | None = None

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level

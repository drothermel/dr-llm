from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import ProviderName, ThinkingLevel
from dr_llm.llm.providers.core.authoring import (
    build_provider_config,
    model_matches_any_family,
    reject_model_family,
    reject_sampling,
    require_model_family,
)
from dr_llm.llm.providers.core.registry import ProviderRegistry
from dr_llm.llm.providers.impls.openai.families import (
    OPENAI_GPT51_FAMILIES,
    OPENAI_GPT52_FAMILIES,
    OPENAI_GPT53_FAMILIES,
    OPENAI_GPT54_FAMILIES,
    OPENAI_GPT5_FAMILIES,
    OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS,
    OPENAI_THINKING_SUPPORTED_MODELS,
    OpenAIModelFamily,
)

type _OpenAIMinimalThinkingLevel = Literal[
    ThinkingLevel.MINIMAL,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
]
type _OpenAIOffThinkingLevel = Literal[
    ThinkingLevel.OFF,
    ThinkingLevel.LOW,
    ThinkingLevel.MEDIUM,
    ThinkingLevel.HIGH,
]


class _OpenAIBaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.OPENAI] = ProviderName.OPENAI
    model: str
    max_tokens: int | None = None

    _families: ClassVar[tuple[OpenAIModelFamily, ...]] = ()

    @model_validator(mode="after")
    def _validate_model_family(self) -> _OpenAIBaseConfig:
        require_model_family(
            provider=self.provider,
            model=self.model,
            families=self._families,
            config_name=type(self).__name__,
        )
        return self

    def _thinking_level(self) -> ThinkingLevel | None:
        return None

    def _sampling(self) -> SamplingControls | None:
        return None

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            thinking_level=self._thinking_level(),
            sampling=self._sampling(),
            registry=registry,
        )


class OpenAILegacyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal[ProviderName.OPENAI] = ProviderName.OPENAI
    model: str
    max_tokens: int | None = None
    sampling: SamplingControls | None = None

    @model_validator(mode="after")
    def _validate_model_family(self) -> OpenAILegacyConfig:
        reject_model_family(
            provider=self.provider,
            model=self.model,
            families=OPENAI_THINKING_SUPPORTED_MODELS,
            config_name=type(self).__name__,
        )
        return self

    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig:
        return build_provider_config(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            sampling=self.sampling,
            registry=registry,
        )


class OpenAIGpt5Config(_OpenAIBaseConfig):
    _families: ClassVar[tuple[OpenAIModelFamily, ...]] = OPENAI_GPT5_FAMILIES

    thinking_level: _OpenAIMinimalThinkingLevel | None = None

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class OpenAIGpt51Config(_OpenAIBaseConfig):
    _families: ClassVar[tuple[OpenAIModelFamily, ...]] = OPENAI_GPT51_FAMILIES

    thinking_level: _OpenAIOffThinkingLevel | None = None

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class OpenAIGpt52Config(_OpenAIBaseConfig):
    _families: ClassVar[tuple[OpenAIModelFamily, ...]] = OPENAI_GPT52_FAMILIES

    thinking_level: _OpenAIOffThinkingLevel | None = None
    sampling: SamplingControls | None = None

    @model_validator(mode="after")
    def _validate_sampling(self) -> OpenAIGpt52Config:
        if (
            self.sampling is not None
            and not self.sampling.is_empty()
            and self.thinking_level not in {None, ThinkingLevel.OFF}
        ):
            raise ValueError(
                f"{type(self).__name__} custom sampling requires "
                "thinking_level='off'"
            )
        if not model_matches_any_family(
            self.model, OPENAI_GPT5_SAMPLING_SUPPORTED_MODELS
        ):
            reject_sampling(
                provider=self.provider,
                model=self.model,
                sampling=self.sampling,
                config_name=type(self).__name__,
            )
        return self

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level

    def _sampling(self) -> SamplingControls | None:
        return self.sampling


class OpenAIGpt53Config(_OpenAIBaseConfig):
    _families: ClassVar[tuple[OpenAIModelFamily, ...]] = OPENAI_GPT53_FAMILIES

    thinking_level: _OpenAIOffThinkingLevel | None = None

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level


class OpenAIGpt54Config(_OpenAIBaseConfig):
    _families: ClassVar[tuple[OpenAIModelFamily, ...]] = OPENAI_GPT54_FAMILIES

    thinking_level: _OpenAIOffThinkingLevel | None = None
    sampling: SamplingControls | None = None

    @model_validator(mode="after")
    def _validate_sampling(self) -> OpenAIGpt54Config:
        if (
            self.sampling is not None
            and not self.sampling.is_empty()
            and self.thinking_level not in {None, ThinkingLevel.OFF}
        ):
            raise ValueError(
                f"{type(self).__name__} custom sampling requires "
                "thinking_level='off'"
            )
        return self

    def _thinking_level(self) -> ThinkingLevel | None:
        return self.thinking_level

    def _sampling(self) -> SamplingControls | None:
        return self.sampling

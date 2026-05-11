from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec, ControlMode, ThinkingLevel
from dr_llm.llm.providers.concepts.model_family import (
    is_snapshot_of_family,
    model_matches_any_family,
)


class OpenAIModelFamily(StrEnum):
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"
    GPT51 = "gpt-5.1"
    GPT51_MINI = "gpt-5.1-mini"
    GPT51_NANO = "gpt-5.1-nano"
    GPT51_CODEX = "gpt-5.1-codex"
    GPT51_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT51_CODEX_MAX = "gpt-5.1-codex-max"
    GPT52 = "gpt-5.2"
    GPT52_MINI = "gpt-5.2-mini"
    GPT52_NANO = "gpt-5.2-nano"
    GPT52_CODEX = "gpt-5.2-codex"
    GPT53 = "gpt-5.3"
    GPT53_MINI = "gpt-5.3-mini"
    GPT53_NANO = "gpt-5.3-nano"
    GPT53_CODEX = "gpt-5.3-codex"
    GPT54 = "gpt-5.4"
    GPT54_MINI = "gpt-5.4-mini"
    GPT54_NANO = "gpt-5.4-nano"

    def in_family(self, model: str) -> bool:
        normalized = _normalize_openai_model(model)
        return normalized == self or is_snapshot_of_family(
            model=normalized, family=str(self)
        )


def _normalize_openai_model(model: str) -> str:
    if model.startswith("openai/"):
        return model[len("openai/") :]
    return model


class OpenAIFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpt5: tuple[OpenAIModelFamily, ...] = (
        OpenAIModelFamily.GPT5,
        OpenAIModelFamily.GPT5_MINI,
        OpenAIModelFamily.GPT5_NANO,
    )
    gpt51: tuple[OpenAIModelFamily, ...] = (
        OpenAIModelFamily.GPT51,
        OpenAIModelFamily.GPT51_MINI,
        OpenAIModelFamily.GPT51_NANO,
        OpenAIModelFamily.GPT51_CODEX,
        OpenAIModelFamily.GPT51_CODEX_MINI,
        OpenAIModelFamily.GPT51_CODEX_MAX,
    )
    gpt52: tuple[OpenAIModelFamily, ...] = (
        OpenAIModelFamily.GPT52,
        OpenAIModelFamily.GPT52_MINI,
        OpenAIModelFamily.GPT52_NANO,
        OpenAIModelFamily.GPT52_CODEX,
    )
    gpt53: tuple[OpenAIModelFamily, ...] = (
        OpenAIModelFamily.GPT53,
        OpenAIModelFamily.GPT53_MINI,
        OpenAIModelFamily.GPT53_NANO,
        OpenAIModelFamily.GPT53_CODEX,
    )
    gpt54: tuple[OpenAIModelFamily, ...] = (
        OpenAIModelFamily.GPT54,
        OpenAIModelFamily.GPT54_MINI,
        OpenAIModelFamily.GPT54_NANO,
    )
    sampling_with_reasoning_off: tuple[OpenAIModelFamily, ...] = (
        OpenAIModelFamily.GPT52,
        OpenAIModelFamily.GPT52_MINI,
        OpenAIModelFamily.GPT52_NANO,
        OpenAIModelFamily.GPT54,
        OpenAIModelFamily.GPT54_MINI,
        OpenAIModelFamily.GPT54_NANO,
    )

    @property
    def thinking_supported(self) -> tuple[OpenAIModelFamily, ...]:
        return (
            *self.gpt5,
            *self.gpt51,
            *self.gpt52,
            *self.gpt53,
            *self.gpt54,
        )

    @property
    def minimal_thinking_supported(self) -> tuple[OpenAIModelFamily, ...]:
        return self.gpt5

    @property
    def off_thinking_supported(self) -> tuple[OpenAIModelFamily, ...]:
        return (*self.gpt51, *self.gpt52, *self.gpt53, *self.gpt54)

    def matches_any(
        self, model: str, families: tuple[OpenAIModelFamily, ...]
    ) -> bool:
        return model_matches_any_family(model, families)

    def supports_configurable_thinking(self, model: str) -> bool:
        return self.matches_any(model, self.thinking_supported)

    def supports_minimal_thinking(self, model: str) -> bool:
        return self.matches_any(model, self.minimal_thinking_supported)

    def supports_off_thinking(self, model: str) -> bool:
        return self.matches_any(model, self.off_thinking_supported)

    def supports_sampling_with_reasoning_off(self, model: str) -> bool:
        return self.matches_any(model, self.sampling_with_reasoning_off)

    def control_mode(self, model: str) -> ControlMode:
        if self.supports_configurable_thinking(model):
            return ControlMode.OPENAI_EFFORT
        return ControlMode.UNSUPPORTED

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        if not self.supports_configurable_thinking(model):
            return (ThinkingLevel.NA,)
        levels: list[ThinkingLevel] = []
        if self.supports_off_thinking(model):
            levels.append(ThinkingLevel.OFF)
        elif self.supports_minimal_thinking(model):
            levels.append(ThinkingLevel.MINIMAL)
        levels.extend(
            [ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH]
        )
        return tuple(levels)

    def default_thinking_level(self, model: str) -> ThinkingLevel:
        levels = self.supported_thinking_levels(model)
        for level in (
            ThinkingLevel.OFF,
            ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW,
        ):
            if level in levels:
                return level
        return ThinkingLevel.NA

    def supported_effort_levels(self, model: str) -> tuple[EffortSpec, ...]:
        del model
        return ()

    def default_effort(self, model: str) -> EffortSpec:
        del model
        return EffortSpec.NA


OPENAI_FAMILIES = OpenAIFamilies()

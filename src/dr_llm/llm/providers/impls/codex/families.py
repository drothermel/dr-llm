from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import EffortSpec, ControlMode, ThinkingLevel
from dr_llm.llm.providers.concepts.model_family import (
    is_snapshot_of_family,
    model_matches_any_family,
)


class CodexModelFamily(StrEnum):
    GPT5 = "gpt-5"
    GPT51 = "gpt-5.1"
    GPT52 = "gpt-5.2"
    GPT54 = "gpt-5.4"
    GPT5_CODEX = "gpt-5-codex"
    GPT51_CODEX = "gpt-5.1-codex"
    GPT51_CODEX_MINI = "gpt-5.1-codex-mini"
    GPT51_CODEX_MAX = "gpt-5.1-codex-max"
    GPT52_CODEX = "gpt-5.2-codex"
    GPT53_CODEX = "gpt-5.3-codex"
    GPT53_CODEX_SPARK = "gpt-5.3-codex-spark"
    GPT54_MINI = "gpt-5.4-mini"

    def in_family(self, model: str) -> bool:
        return model == self or is_snapshot_of_family(
            model=model, family=str(self)
        )


class CodexFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpt5: tuple[CodexModelFamily, ...] = (CodexModelFamily.GPT5,)
    gpt51: tuple[CodexModelFamily, ...] = (CodexModelFamily.GPT51,)
    gpt52: tuple[CodexModelFamily, ...] = (CodexModelFamily.GPT52,)
    gpt54: tuple[CodexModelFamily, ...] = (
        CodexModelFamily.GPT54,
        CodexModelFamily.GPT54_MINI,
    )
    codex: tuple[CodexModelFamily, ...] = (
        CodexModelFamily.GPT5_CODEX,
        CodexModelFamily.GPT51_CODEX,
        CodexModelFamily.GPT51_CODEX_MINI,
        CodexModelFamily.GPT51_CODEX_MAX,
        CodexModelFamily.GPT52_CODEX,
        CodexModelFamily.GPT53_CODEX,
        CodexModelFamily.GPT53_CODEX_SPARK,
    )

    @property
    def thinking_supported(self) -> tuple[CodexModelFamily, ...]:
        return (*self.gpt5, *self.gpt51, *self.gpt52, *self.gpt54, *self.codex)

    @property
    def minimal_thinking_supported(self) -> tuple[CodexModelFamily, ...]:
        return self.gpt5

    @property
    def off_thinking_supported(self) -> tuple[CodexModelFamily, ...]:
        return (*self.gpt51, *self.gpt52, *self.gpt54)

    def matches_any(
        self, model: str, families: tuple[CodexModelFamily, ...]
    ) -> bool:
        return model_matches_any_family(model, families)

    def supports_configurable_thinking(self, model: str) -> bool:
        return self.matches_any(model, self.thinking_supported)

    def supports_minimal_thinking(self, model: str) -> bool:
        return self.matches_any(model, self.minimal_thinking_supported)

    def supports_off_thinking(self, model: str) -> bool:
        return self.matches_any(model, self.off_thinking_supported)

    def control_mode(self, model: str) -> ControlMode:
        if self.supports_configurable_thinking(model):
            return ControlMode.CODEX_CLI_EFFORT
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
            [
                ThinkingLevel.LOW,
                ThinkingLevel.MEDIUM,
                ThinkingLevel.HIGH,
                ThinkingLevel.XHIGH,
            ]
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


CODEX_FAMILIES = CodexFamilies()

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import (
    EffortSpec,
    ProviderName,
    ControlMode,
    ThinkingLevel,
)
from dr_llm.llm.providers.concepts.model_family import (
    model_matches_any_family,
)
from dr_llm.llm.providers.concepts.reasoning import (
    google_literal_to_thinking_level,
)


class GoogleThinkingLevel(StrEnum):
    MINIMAL = ThinkingLevel.MINIMAL
    LOW = ThinkingLevel.LOW
    MEDIUM = ThinkingLevel.MEDIUM
    HIGH = ThinkingLevel.HIGH


class GoogleModelFamily(StrEnum):
    GEMINI_25_FLASH_LITE_PREVIEW = "gemini-2.5-flash-lite-preview"
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_25_FLASH_PREVIEW = "gemini-2.5-flash-preview"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_3 = "gemini-3"
    GEMMA_4 = "gemma-4"

    def in_family(self, model: str) -> bool:
        return model == self.value or model.startswith(f"{self.value}-")


class GoogleBudgetFamilySpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    families: tuple[GoogleModelFamily, ...]
    min_budget_tokens: int
    max_budget_tokens: int

    def includes(self, model: str) -> bool:
        return model_matches_any_family(model, self.families)


class GoogleLevelFamilySpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    families: tuple[GoogleModelFamily, ...]
    thinking_levels: tuple[GoogleThinkingLevel, ...]

    def includes(self, model: str) -> bool:
        return model_matches_any_family(model, self.families)


class GoogleFamilies(BaseModel):
    model_config = ConfigDict(frozen=True)

    budget_specs: tuple[GoogleBudgetFamilySpec, ...] = (
        GoogleBudgetFamilySpec(
            families=(
                GoogleModelFamily.GEMINI_25_FLASH_LITE_PREVIEW,
                GoogleModelFamily.GEMINI_25_FLASH_LITE,
            ),
            min_budget_tokens=512,
            max_budget_tokens=24576,
        ),
        GoogleBudgetFamilySpec(
            families=(
                GoogleModelFamily.GEMINI_25_FLASH_PREVIEW,
                GoogleModelFamily.GEMINI_25_FLASH,
            ),
            min_budget_tokens=1,
            max_budget_tokens=24576,
        ),
        GoogleBudgetFamilySpec(
            families=(GoogleModelFamily.GEMINI_25_PRO,),
            min_budget_tokens=128,
            max_budget_tokens=32768,
        ),
    )
    level_specs: tuple[GoogleLevelFamilySpec, ...] = (
        GoogleLevelFamilySpec(
            families=(GoogleModelFamily.GEMINI_3,),
            thinking_levels=(
                GoogleThinkingLevel.MINIMAL,
                GoogleThinkingLevel.LOW,
                GoogleThinkingLevel.MEDIUM,
                GoogleThinkingLevel.HIGH,
            ),
        ),
        GoogleLevelFamilySpec(
            families=(GoogleModelFamily.GEMMA_4,),
            thinking_levels=(
                GoogleThinkingLevel.MINIMAL,
                GoogleThinkingLevel.HIGH,
            ),
        ),
    )

    def budget_spec_for_model(
        self, model: str
    ) -> GoogleBudgetFamilySpec | None:
        return next(
            (spec for spec in self.budget_specs if spec.includes(model)),
            None,
        )

    def level_spec_for_model(self, model: str) -> GoogleLevelFamilySpec | None:
        return next(
            (spec for spec in self.level_specs if spec.includes(model)),
            None,
        )

    def control_mode(self, model: str) -> ControlMode:
        if self.budget_spec_for_model(model) is not None:
            return ControlMode.GOOGLE_BUDGET
        if self.level_spec_for_model(model) is not None:
            return ControlMode.GOOGLE_LEVEL
        return ControlMode.UNSUPPORTED

    def min_budget_tokens(self, model: str) -> int | None:
        spec = self.budget_spec_for_model(model)
        if spec is None:
            return None
        return spec.min_budget_tokens

    def max_budget_tokens(self, model: str) -> int | None:
        spec = self.budget_spec_for_model(model)
        if spec is None:
            return None
        return spec.max_budget_tokens

    def supports_dynamic(self, model: str) -> bool:
        return self.control_mode(model) == ControlMode.GOOGLE_BUDGET

    def google_thinking_levels(
        self, model: str
    ) -> tuple[GoogleThinkingLevel, ...]:
        spec = self.level_spec_for_model(model)
        if spec is None:
            return ()
        return spec.thinking_levels

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]:
        control_mode = self.control_mode(model)
        if control_mode == ControlMode.UNSUPPORTED:
            return (ThinkingLevel.NA,)
        if control_mode == ControlMode.GOOGLE_BUDGET:
            return (
                ThinkingLevel.ADAPTIVE,
                ThinkingLevel.OFF,
                ThinkingLevel.BUDGET,
            )
        if control_mode == ControlMode.GOOGLE_LEVEL:
            return tuple(
                google_literal_to_thinking_level(level)
                for level in self.google_thinking_levels(model)
            )
        raise ValueError(
            f"unexpected control mode for provider={ProviderName.GOOGLE!r} "
            f"model={model!r}: {control_mode!r}"
        )

    def default_thinking_level(self, model: str) -> ThinkingLevel:
        levels = self.supported_thinking_levels(model)
        for level in (
            ThinkingLevel.OFF,
            ThinkingLevel.MINIMAL,
            ThinkingLevel.ADAPTIVE,
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


GOOGLE_FAMILIES = GoogleFamilies()

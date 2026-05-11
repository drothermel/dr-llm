from __future__ import annotations

from enum import IntEnum

from dr_llm.llm.names import ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    GoogleThinkingLevel,
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)
from dr_llm.llm.providers.impls.google.families import (
    GEMMA_4_FAMILIES,
    GOOGLE_25_FLASH_FAMILIES,
    GOOGLE_25_FLASH_LITE_FAMILIES,
    GOOGLE_25_PRO_FAMILIES,
    GOOGLE_3_FAMILIES,
)


class GoogleMinBudget(IntEnum):
    GEMINI_25_FLASH = 1
    GEMINI_25_FLASH_LITE = 512
    GEMINI_25_PRO = 128


class GoogleMaxBudget(IntEnum):
    GEMINI_25_FLASH = 24576
    GEMINI_25_FLASH_LITE = 24576
    GEMINI_25_PRO = 32768


_GOOGLE_25_FLASH_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_BUDGET,
    min_budget_tokens=GoogleMinBudget.GEMINI_25_FLASH,
    max_budget_tokens=GoogleMaxBudget.GEMINI_25_FLASH,
    supports_dynamic=True,
)
_GOOGLE_25_FLASH_LITE_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_BUDGET,
    min_budget_tokens=GoogleMinBudget.GEMINI_25_FLASH_LITE,
    max_budget_tokens=GoogleMaxBudget.GEMINI_25_FLASH_LITE,
    supports_dynamic=True,
)
_GOOGLE_25_PRO_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_BUDGET,
    min_budget_tokens=GoogleMinBudget.GEMINI_25_PRO,
    max_budget_tokens=GoogleMaxBudget.GEMINI_25_PRO,
    supports_dynamic=True,
)
_GOOGLE_3_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_LEVEL,
    google_levels=(
        GoogleThinkingLevel.MINIMAL,
        GoogleThinkingLevel.LOW,
        GoogleThinkingLevel.MEDIUM,
        GoogleThinkingLevel.HIGH,
    ),
)
_GEMMA_4_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_LEVEL,
    google_levels=(GoogleThinkingLevel.MINIMAL, GoogleThinkingLevel.HIGH),
)

GOOGLE_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    *(
        ReasoningCapabilityRule(
            family=family, capabilities=_GOOGLE_25_FLASH_LITE_CAPS
        )
        for family in GOOGLE_25_FLASH_LITE_FAMILIES
    ),
    *(
        ReasoningCapabilityRule(
            family=family, capabilities=_GOOGLE_25_FLASH_CAPS
        )
        for family in GOOGLE_25_FLASH_FAMILIES
    ),
    *(
        ReasoningCapabilityRule(
            family=family, capabilities=_GOOGLE_25_PRO_CAPS
        )
        for family in GOOGLE_25_PRO_FAMILIES
    ),
    *(
        ReasoningCapabilityRule(family=family, capabilities=_GOOGLE_3_CAPS)
        for family in GOOGLE_3_FAMILIES
    ),
    *(
        ReasoningCapabilityRule(family=family, capabilities=_GEMMA_4_CAPS)
        for family in GEMMA_4_FAMILIES
    ),
)


def reasoning_capabilities_for_google(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(GOOGLE_CAPABILITY_RULES, model)

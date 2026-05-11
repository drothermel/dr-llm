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


_GOOGLE_25_FLASH_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_BUDGET,
            min_budget_tokens=GoogleMinBudget.GEMINI_25_FLASH,
            max_budget_tokens=GoogleMaxBudget.GEMINI_25_FLASH,
            supports_dynamic=True,
        ),
    )
    for family in GOOGLE_25_FLASH_FAMILIES
)
_GOOGLE_25_FLASH_LITE_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_BUDGET,
            min_budget_tokens=GoogleMinBudget.GEMINI_25_FLASH_LITE,
            max_budget_tokens=GoogleMaxBudget.GEMINI_25_FLASH_LITE,
            supports_dynamic=True,
        ),
    )
    for family in GOOGLE_25_FLASH_LITE_FAMILIES
)
_GOOGLE_25_PRO_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_BUDGET,
            min_budget_tokens=GoogleMinBudget.GEMINI_25_PRO,
            max_budget_tokens=GoogleMaxBudget.GEMINI_25_PRO,
            supports_dynamic=True,
        ),
    )
    for family in GOOGLE_25_PRO_FAMILIES
)
_GOOGLE_3_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_LEVEL,
            google_thinking_levels=(
                GoogleThinkingLevel.MINIMAL,
                GoogleThinkingLevel.LOW,
                GoogleThinkingLevel.MEDIUM,
                GoogleThinkingLevel.HIGH,
            ),
        ),
    )
    for family in GOOGLE_3_FAMILIES
)
_GEMMA_4_RULES: tuple[ReasoningCapabilityRule, ...] = tuple(
    ReasoningCapabilityRule(
        family=family,
        capabilities=ReasoningCapabilities(
            mode=ReasoningMode.GOOGLE_LEVEL,
            google_thinking_levels=(
                GoogleThinkingLevel.MINIMAL,
                GoogleThinkingLevel.HIGH,
            ),
        ),
    )
    for family in GEMMA_4_FAMILIES
)


def reasoning_capabilities_for_google(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(
        (
            *_GOOGLE_25_FLASH_LITE_RULES,
            *_GOOGLE_25_FLASH_RULES,
            *_GOOGLE_25_PRO_RULES,
            *_GOOGLE_3_RULES,
            *_GEMMA_4_RULES,
        ),
        model,
    )

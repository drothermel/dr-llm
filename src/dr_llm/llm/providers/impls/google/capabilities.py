from __future__ import annotations

from dr_llm.llm.names import ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
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

_GOOGLE_25_FLASH_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_BUDGET,
    min_budget_tokens=1,
    max_budget_tokens=24576,
    supports_dynamic=True,
)
_GOOGLE_25_FLASH_LITE_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_BUDGET,
    min_budget_tokens=512,
    max_budget_tokens=24576,
    supports_dynamic=True,
)
_GOOGLE_25_PRO_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_BUDGET,
    min_budget_tokens=128,
    max_budget_tokens=32768,
    supports_dynamic=True,
)
_GOOGLE_3_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_LEVEL,
    google_levels=("minimal", "low", "medium", "high"),
)
_GEMMA_4_CAPS = ReasoningCapabilities(
    mode=ReasoningMode.GOOGLE_LEVEL,
    google_levels=("minimal", "high"),
)

GOOGLE_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    *(
        ReasoningCapabilityRule(
            model_prefix=family, capabilities=_GOOGLE_25_FLASH_LITE_CAPS
        )
        for family in GOOGLE_25_FLASH_LITE_FAMILIES
    ),
    *(
        ReasoningCapabilityRule(
            model_prefix=family, capabilities=_GOOGLE_25_FLASH_CAPS
        )
        for family in GOOGLE_25_FLASH_FAMILIES
    ),
    *(
        ReasoningCapabilityRule(
            model_prefix=family, capabilities=_GOOGLE_25_PRO_CAPS
        )
        for family in GOOGLE_25_PRO_FAMILIES
    ),
    *(
        ReasoningCapabilityRule(
            model_prefix=family, capabilities=_GOOGLE_3_CAPS
        )
        for family in GOOGLE_3_FAMILIES
    ),
    *(
        ReasoningCapabilityRule(
            model_prefix=family, capabilities=_GEMMA_4_CAPS
        )
        for family in GEMMA_4_FAMILIES
    ),
)


def reasoning_capabilities_for_google(
    model: str,
) -> ReasoningCapabilities | None:
    return resolve_capability_rules(GOOGLE_CAPABILITY_RULES, model)

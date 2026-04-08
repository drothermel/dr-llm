from __future__ import annotations

from dr_llm.llm.providers.reasoning_capability_types import (
    ReasoningCapabilities,
    ReasoningCapabilityRule,
    resolve_capability_rules,
)

_GOOGLE_25_FLASH_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=1,
    max_budget_tokens=24576,
    supports_dynamic=True,
)
_GOOGLE_25_FLASH_LITE_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=512,
    max_budget_tokens=24576,
    supports_dynamic=True,
)
_GOOGLE_25_PRO_CAPS = ReasoningCapabilities(
    mode="google_budget",
    min_budget_tokens=128,
    max_budget_tokens=32768,
    supports_dynamic=True,
)
_GOOGLE_3_CAPS = ReasoningCapabilities(
    mode="google_level",
    google_levels=("minimal", "low", "medium", "high"),
)
_GEMMA_4_CAPS = ReasoningCapabilities(
    mode="google_level",
    google_levels=("minimal", "high"),
)

GOOGLE_CAPABILITY_RULES: tuple[ReasoningCapabilityRule, ...] = (
    ReasoningCapabilityRule(
        model_prefix="gemini-2.5-flash-lite-preview",
        capabilities=_GOOGLE_25_FLASH_LITE_CAPS,
    ),
    ReasoningCapabilityRule(
        model_prefix="gemini-2.5-flash-lite",
        capabilities=_GOOGLE_25_FLASH_LITE_CAPS,
    ),
    ReasoningCapabilityRule(
        model_prefix="gemini-2.5-flash-preview",
        capabilities=_GOOGLE_25_FLASH_CAPS,
    ),
    ReasoningCapabilityRule(
        model_prefix="gemini-2.5-flash",
        capabilities=_GOOGLE_25_FLASH_CAPS,
    ),
    ReasoningCapabilityRule(
        model_prefix="gemini-2.5-pro",
        capabilities=_GOOGLE_25_PRO_CAPS,
    ),
    ReasoningCapabilityRule(model_prefix="gemini-3", capabilities=_GOOGLE_3_CAPS),
    ReasoningCapabilityRule(model_prefix="gemma-4", capabilities=_GEMMA_4_CAPS),
)


def reasoning_capabilities_for_google(model: str) -> ReasoningCapabilities | None:
    return resolve_capability_rules(GOOGLE_CAPABILITY_RULES, model)

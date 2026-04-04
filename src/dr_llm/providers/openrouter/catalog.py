from __future__ import annotations

from dr_llm.catalog.models import ModelCatalogEntry
from dr_llm.providers.openrouter.policy import (
    OpenRouterReasoningRequestStyle,
    openrouter_model_policy,
)
from dr_llm.providers.reasoning_capabilities import ReasoningCapabilities

_OPENROUTER_TOGGLE_CAPABILITIES = ReasoningCapabilities(mode="openrouter_toggle")
_OPENROUTER_EFFORT_CAPABILITIES = ReasoningCapabilities(mode="openrouter_effort")
_OPENROUTER_UNSUPPORTED_CAPABILITIES = ReasoningCapabilities(mode="unsupported")


def apply_openrouter_model_policies(
    entries: list[ModelCatalogEntry],
) -> list[ModelCatalogEntry]:
    filtered: list[ModelCatalogEntry] = []
    for entry in entries:
        if entry.provider != "openrouter":
            filtered.append(entry)
            continue
        policy = openrouter_model_policy(entry.model)
        if policy is None:
            continue
        capabilities = _capabilities_for_policy(policy.request_style)
        filtered.append(
            ModelCatalogEntry.model_validate(
                entry.model_dump(mode="python")
                | {
                    "supports_reasoning": capabilities.supports_reasoning,
                    "reasoning_capabilities": capabilities,
                }
            )
        )
    return filtered


def _capabilities_for_policy(
    request_style: OpenRouterReasoningRequestStyle,
) -> ReasoningCapabilities:
    if request_style == OpenRouterReasoningRequestStyle.ENABLED_FLAG:
        return _OPENROUTER_TOGGLE_CAPABILITIES
    if request_style == OpenRouterReasoningRequestStyle.EFFORT:
        return _OPENROUTER_EFFORT_CAPABILITIES
    return _OPENROUTER_UNSUPPORTED_CAPABILITIES

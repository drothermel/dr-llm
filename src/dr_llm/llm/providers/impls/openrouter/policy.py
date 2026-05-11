from __future__ import annotations

from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict

from dr_llm.llm.names import OpenRouterEffortLevel, ProviderName, ReasoningMode
from dr_llm.llm.providers.concepts.capabilities import (
    ReasoningCapabilities,
)

if TYPE_CHECKING:
    from dr_llm.llm.catalog.models import ModelCatalogEntry


class OpenRouterReasoningRequestStyle(StrEnum):
    NONE = "none"
    ENABLED_FLAG = "enabled_flag"
    EFFORT = "effort"


class OpenRouterModelPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    request_style: OpenRouterReasoningRequestStyle
    supports_disable: bool
    allowed_efforts: tuple[OpenRouterEffortLevel, ...] = ()
    default_enabled: bool | None = None
    verified: bool = False
    notes: str | None = None


@cache
def _policies() -> dict[str, OpenRouterModelPolicy]:
    raw = yaml.safe_load(
        files("dr_llm.llm.providers.impls.openrouter.data")
        .joinpath("model_policies.yml")
        .read_text(encoding="utf-8")
    )
    return {
        model: OpenRouterModelPolicy(model=model, **fields)
        for model, fields in raw.items()
    }


def openrouter_model_policy(model: str) -> OpenRouterModelPolicy | None:
    return _policies().get(model)


def reasoning_capabilities_for_openrouter(
    model: str,
) -> ReasoningCapabilities | None:
    policy = openrouter_model_policy(model)
    if policy is None:
        return None
    return _capabilities_for_policy(policy.request_style)


def openrouter_allowed_models() -> tuple[str, ...]:
    return tuple(_policies())


def apply_openrouter_model_policies(
    entries: list[ModelCatalogEntry],
) -> list[ModelCatalogEntry]:
    filtered: list[ModelCatalogEntry] = []
    for entry in entries:
        if entry.provider != ProviderName.OPENROUTER:
            filtered.append(entry)
            continue
        policy = openrouter_model_policy(entry.model)
        if policy is None:
            continue
        capabilities = _capabilities_for_policy(policy.request_style)
        filtered.append(
            entry.model_copy(
                update={
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
        return ReasoningCapabilities(mode=ReasoningMode.OPENROUTER_TOGGLE)
    if request_style == OpenRouterReasoningRequestStyle.EFFORT:
        return ReasoningCapabilities(mode=ReasoningMode.OPENROUTER_EFFORT)
    return ReasoningCapabilities(mode=ReasoningMode.UNSUPPORTED)

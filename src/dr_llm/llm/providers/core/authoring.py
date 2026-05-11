from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from dr_llm.llm.config import LlmConfig, SamplingControls
from dr_llm.llm.names import EffortSpec, ThinkingLevel
from dr_llm.llm.providers.concepts.reasoning import ReasoningSpec
from dr_llm.llm.providers.concepts.thinking_utils import matches_family

if TYPE_CHECKING:
    from dr_llm.llm.names import ProviderName
    from dr_llm.llm.providers.core.registry import ProviderRegistry


class LlmAuthoringConfig(Protocol):
    def to_llm_config(
        self, registry: ProviderRegistry | None = None
    ) -> LlmConfig: ...


def model_matches_any_family(model: str, families: Sequence[str]) -> bool:
    return matches_family(normalized=model, families=list(families))


def require_model_family(
    *,
    provider: str,
    model: str,
    families: Sequence[str],
    config_name: str,
) -> None:
    if model_matches_any_family(model, families):
        return
    joined = ", ".join(families)
    raise ValueError(
        f"{config_name} only supports provider={provider!r} "
        f"model families: {joined}; got model={model!r}"
    )


def reject_model_family(
    *,
    provider: str,
    model: str,
    families: Sequence[str],
    config_name: str,
) -> None:
    if not model_matches_any_family(model, families):
        return
    joined = ", ".join(families)
    raise ValueError(
        f"{config_name} does not support provider={provider!r} "
        f"model families: {joined}; got model={model!r}"
    )


def validate_budget_range(
    *,
    provider: str,
    model: str,
    budget_tokens: int,
    min_tokens: int,
    max_tokens: int,
    label: str = "budget_tokens",
) -> None:
    if min_tokens <= budget_tokens <= max_tokens:
        return
    raise ValueError(
        f"{label} must be between {min_tokens} and {max_tokens} "
        f"for provider={provider!r} model={model!r}"
    )


def reject_sampling(
    *,
    provider: str,
    model: str,
    sampling: SamplingControls | None,
    config_name: str,
) -> None:
    if sampling is None or sampling.is_empty():
        return
    raise ValueError(
        f"{config_name} does not support custom sampling for "
        f"provider={provider!r} model={model!r}"
    )


def build_provider_config(
    *,
    provider: ProviderName,
    model: str,
    registry: ProviderRegistry | None,
    max_tokens: int | None = None,
    effort: EffortSpec | None = None,
    reasoning: ReasoningSpec | None = None,
    thinking_level: ThinkingLevel | None = None,
    budget_tokens: int | None = None,
    sampling: SamplingControls | None = None,
) -> LlmConfig:
    from dr_llm.llm.providers.default_registry import build_default_registry

    owns_registry = registry is None
    resolved_registry = registry or build_default_registry()
    try:
        return resolved_registry.get(provider).build_config(
            model=model,
            max_tokens=max_tokens,
            effort=effort,
            reasoning=reasoning,
            thinking_level=thinking_level,
            budget_tokens=budget_tokens,
            sampling=sampling,
        )
    finally:
        if owns_registry:
            resolved_registry.close()

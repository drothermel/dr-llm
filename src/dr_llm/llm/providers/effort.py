from __future__ import annotations

from enum import StrEnum

from dr_llm.llm.providers.anthropic.effort import ANTHROPIC_EFFORT_SUPPORTED_MODELS
from dr_llm.llm.providers.reasoning_capabilities import reasoning_capabilities_for_model

_DIRECT_ANTHROPIC_EFFORT_MODELS = frozenset(ANTHROPIC_EFFORT_SUPPORTED_MODELS)
_EFFORT_CAPABILITY_MODES = frozenset(
    {
        "anthropic_effort",
        "anthropic_effort_and_budget",
        "claude_cli_effort",
        "kimi_code_effort_and_budget",
        "minimax_effort",
    }
)


class EffortSpec(StrEnum):
    NA = "na"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


def supported_effort_levels(*, provider: str, model: str) -> tuple[EffortSpec, ...]:
    capabilities = reasoning_capabilities_for_model(provider=provider, model=model)
    if capabilities is None or capabilities.mode not in _EFFORT_CAPABILITY_MODES:
        return ()

    if provider == "anthropic":
        if model not in _DIRECT_ANTHROPIC_EFFORT_MODELS:
            return ()
        levels = [EffortSpec.LOW, EffortSpec.MEDIUM, EffortSpec.HIGH]
        if model.startswith("claude-opus-4-6"):
            levels.append(EffortSpec.MAX)
        return tuple(levels)

    if provider == "claude-code":
        if model not in _DIRECT_ANTHROPIC_EFFORT_MODELS:
            return ()
        return (
            EffortSpec.LOW,
            EffortSpec.MEDIUM,
            EffortSpec.HIGH,
            EffortSpec.MAX,
        )

    if provider == "minimax":
        return (
            EffortSpec.LOW,
            EffortSpec.MEDIUM,
            EffortSpec.HIGH,
            EffortSpec.MAX,
        )

    if provider == "kimi-code":
        return (
            EffortSpec.LOW,
            EffortSpec.MEDIUM,
            EffortSpec.HIGH,
            EffortSpec.MAX,
        )

    return ()


def validate_effort(*, provider: str, model: str, effort: EffortSpec) -> None:
    allowed_levels = supported_effort_levels(provider=provider, model=model)
    if not allowed_levels:
        if effort != EffortSpec.NA:
            raise ValueError(
                f"effort is not supported for provider={provider!r} model={model!r}"
            )
        return

    if effort == EffortSpec.NA:
        raise ValueError(
            f"effort is required for provider={provider!r} model={model!r}"
        )

    if effort not in allowed_levels:
        allowed = ", ".join(level.value for level in allowed_levels)
        raise ValueError(
            f"effort={effort.value!r} is not supported for provider={provider!r} model={model!r}; allowed levels: {allowed}"
        )

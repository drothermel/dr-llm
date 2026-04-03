from __future__ import annotations

from enum import StrEnum

from dr_llm.providers.anthropic.effort import ANTHROPIC_EFFORT_SUPPORTED_MODELS

_ANTHROPIC_EFFORT_PROVIDERS = frozenset({"anthropic", "claude-code"})


class EffortSpec(StrEnum):
    NA = "na"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def validate_effort(*, provider: str, model: str, effort: EffortSpec) -> None:
    if provider in _ANTHROPIC_EFFORT_PROVIDERS:
        if model in ANTHROPIC_EFFORT_SUPPORTED_MODELS:
            if effort == EffortSpec.NA:
                raise ValueError(
                    f"effort is required for provider={provider!r} model={model!r}"
                )
            return
        if effort != EffortSpec.NA:
            raise ValueError(
                f"effort is not supported for provider={provider!r} model={model!r}"
            )
        return

    if effort != EffortSpec.NA:
        raise ValueError(
            f"effort is not supported for provider={provider!r} model={model!r}"
        )

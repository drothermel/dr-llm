from __future__ import annotations

from dr_llm.llm.names import EffortSpec

FULL_EFFORT: tuple[EffortSpec, ...] = (
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
    EffortSpec.MAX,
)


def default_effort(
    supported_effort_levels: tuple[EffortSpec, ...],
) -> EffortSpec:
    if supported_effort_levels:
        return supported_effort_levels[0]
    return EffortSpec.NA


def validate_effort(
    *,
    provider: str,
    model: str,
    effort: EffortSpec,
    supported_effort_levels: tuple[EffortSpec, ...],
) -> None:
    if not supported_effort_levels:
        if effort != EffortSpec.NA:
            raise ValueError(
                f"effort is not supported for provider={provider!r} "
                f"model={model!r}"
            )
        return
    if effort == EffortSpec.NA:
        raise ValueError(
            f"effort is required for provider={provider!r} model={model!r}"
        )
    if effort not in supported_effort_levels:
        allowed = ", ".join(str(level) for level in supported_effort_levels)
        raise ValueError(
            f"effort={effort!r} is not supported for provider={provider!r} "
            f"model={model!r}; allowed levels: {allowed}"
        )

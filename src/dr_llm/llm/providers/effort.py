from __future__ import annotations

from collections.abc import Callable

from dr_llm.llm.providers.anthropic.effort import (
    supported_effort_levels_for_anthropic,
)
from dr_llm.llm.providers.effort_types import FULL_EFFORT, EffortSpec
from dr_llm.llm.providers.headless.claude_capabilities import (
    supported_effort_levels_for_claude_code,
)
from dr_llm.llm.providers.kimi_code_capabilities import (
    supported_effort_levels_for_kimi_code,
)
from dr_llm.llm.providers.minimax_capabilities import (
    supported_effort_levels_for_minimax,
)

__all__ = [
    "FULL_EFFORT",
    "EffortSpec",
    "supported_effort_levels",
    "validate_effort",
]


EffortResolver = Callable[[str], tuple[EffortSpec, ...]]

_EFFORT_RESOLVERS: dict[str, EffortResolver] = {
    "anthropic": supported_effort_levels_for_anthropic,
    "claude-code": supported_effort_levels_for_claude_code,
    "kimi-code": supported_effort_levels_for_kimi_code,
    "minimax": supported_effort_levels_for_minimax,
}


def supported_effort_levels(*, provider: str, model: str) -> tuple[EffortSpec, ...]:
    resolver = _EFFORT_RESOLVERS.get(provider)
    if resolver is None:
        return ()
    return resolver(model)


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

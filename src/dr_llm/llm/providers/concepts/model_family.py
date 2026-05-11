from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from dr_llm.llm.names import ControlMode, EffortSpec, ThinkingLevel


@runtime_checkable
class ModelFamily(Protocol):
    def in_family(self, model: str) -> bool: ...


class ProviderFamilies(Protocol):
    def control_mode(self, model: str) -> ControlMode: ...

    def supported_thinking_levels(
        self, model: str
    ) -> tuple[ThinkingLevel, ...]: ...

    def default_thinking_level(self, model: str) -> ThinkingLevel: ...

    def supported_effort_levels(
        self, model: str
    ) -> tuple[EffortSpec, ...]: ...

    def default_effort(self, model: str) -> EffortSpec: ...


def model_matches_any_family(
    model: str, families: Sequence[ModelFamily]
) -> bool:
    return any(family.in_family(model) for family in families)


def is_snapshot_of_family(*, model: str, family: str) -> bool:
    prefix = f"{family}-"
    if not model.startswith(prefix):
        return False
    suffix = model[len(prefix) :]
    return bool(suffix) and suffix[0].isdigit()

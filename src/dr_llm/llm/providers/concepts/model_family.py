from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelFamily(Protocol):
    def in_family(self, model: str) -> bool: ...


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

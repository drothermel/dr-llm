from __future__ import annotations

from enum import StrEnum


class EffortSpec(StrEnum):
    NA = "na"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


FULL_EFFORT: tuple[EffortSpec, ...] = (
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
    EffortSpec.MAX,
)

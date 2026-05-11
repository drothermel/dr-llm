from __future__ import annotations

from dr_llm.llm.names import EffortSpec

FULL_EFFORT: tuple[EffortSpec, ...] = (
    EffortSpec.LOW,
    EffortSpec.MEDIUM,
    EffortSpec.HIGH,
    EffortSpec.MAX,
)

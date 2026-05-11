from __future__ import annotations

from enum import StrEnum


class MiniMaxModelFamily(StrEnum):
    MINIMAX = "MiniMax-"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


MINIMAX_SUPPORTED_MODEL_FAMILIES = (MiniMaxModelFamily.MINIMAX,)

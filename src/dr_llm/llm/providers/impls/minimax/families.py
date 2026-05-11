from __future__ import annotations

from enum import StrEnum


class MiniMaxModelFamily(StrEnum):
    MINIMAX = "MiniMax-"


MINIMAX_SUPPORTED_MODEL_FAMILIES = (MiniMaxModelFamily.MINIMAX,)

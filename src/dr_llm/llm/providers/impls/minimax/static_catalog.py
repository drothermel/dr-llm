from __future__ import annotations

from enum import StrEnum


class MiniMaxStaticCatalogModel(StrEnum):
    MINIMAX_M27 = "MiniMax-M2.7"
    MINIMAX_M25 = "MiniMax-M2.5"
    MINIMAX_M21 = "MiniMax-M2.1"
    MINIMAX_M2 = "MiniMax-M2"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]

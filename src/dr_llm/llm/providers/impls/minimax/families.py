from __future__ import annotations

from enum import StrEnum


class MiniMaxModelFamily(StrEnum):
    MINIMAX = "MiniMax-"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


class MiniMaxStaticCatalogModel(StrEnum):
    MINIMAX_M27 = "MiniMax-M2.7"
    MINIMAX_M25 = "MiniMax-M2.5"
    MINIMAX_M21 = "MiniMax-M2.1"
    MINIMAX_M2 = "MiniMax-M2"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]


MINIMAX_SUPPORTED_MODEL_FAMILIES = (MiniMaxModelFamily.MINIMAX,)

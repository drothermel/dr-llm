from __future__ import annotations

from enum import StrEnum


class KimiCodeStaticCatalogModel(StrEnum):
    KIMI_FOR_CODING = "kimi-for-coding"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]

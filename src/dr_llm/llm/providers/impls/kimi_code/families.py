from __future__ import annotations

from enum import StrEnum


class KimiCodeModelFamily(StrEnum):
    KIMI_FOR_CODING = "kimi-for-coding"

    def in_family(self, model: str) -> bool:
        return model == self


class KimiCodeStaticCatalogModel(StrEnum):
    KIMI_FOR_CODING = "kimi-for-coding"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]


KIMI_CODE_SUPPORTED_MODELS = (KimiCodeModelFamily.KIMI_FOR_CODING,)

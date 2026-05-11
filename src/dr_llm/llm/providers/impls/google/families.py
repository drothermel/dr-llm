from __future__ import annotations

from enum import StrEnum


class GoogleModelFamily(StrEnum):
    GEMINI_25_FLASH_LITE_PREVIEW = "gemini-2.5-flash-lite-preview"
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_25_FLASH_PREVIEW = "gemini-2.5-flash-preview"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_3 = "gemini-3"
    GEMMA_4 = "gemma-4"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


class GoogleStaticCatalogModel(StrEnum):
    GEMINI_25_PRO_PREVIEW_0506 = "gemini-2.5-pro-preview-05-06"
    GEMINI_25_FLASH_PREVIEW_0417 = "gemini-2.5-flash-preview-04-17"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "gemini-2.0-flash-lite"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]


GOOGLE_25_FLASH_LITE_FAMILIES = (
    GoogleModelFamily.GEMINI_25_FLASH_LITE_PREVIEW,
    GoogleModelFamily.GEMINI_25_FLASH_LITE,
)
GOOGLE_25_FLASH_FAMILIES = (
    GoogleModelFamily.GEMINI_25_FLASH_PREVIEW,
    GoogleModelFamily.GEMINI_25_FLASH,
)
GOOGLE_25_PRO_FAMILIES = (GoogleModelFamily.GEMINI_25_PRO,)
GOOGLE_3_FAMILIES = (GoogleModelFamily.GEMINI_3,)
GEMMA_4_FAMILIES = (GoogleModelFamily.GEMMA_4,)

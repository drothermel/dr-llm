from __future__ import annotations

from enum import StrEnum


class GoogleStaticCatalogModel(StrEnum):
    GEMINI_25_PRO_PREVIEW_0506 = "gemini-2.5-pro-preview-05-06"
    GEMINI_25_FLASH_PREVIEW_0417 = "gemini-2.5-flash-preview-04-17"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "gemini-2.0-flash-lite"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]

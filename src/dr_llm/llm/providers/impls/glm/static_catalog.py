from __future__ import annotations

from enum import StrEnum


class GlmStaticCatalogModel(StrEnum):
    GLM45 = "glm-4.5"
    GLM4_AIR = "glm-4-air"
    GLM4_FLASH = "glm-4-flash"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]

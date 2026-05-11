from __future__ import annotations

from enum import StrEnum


class GlmModelFamily(StrEnum):
    GLM5 = "glm-5"
    GLM47 = "glm-4.7"
    GLM46 = "glm-4.6"
    GLM45 = "glm-4.5"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


class GlmStaticCatalogModel(StrEnum):
    GLM45 = "glm-4.5"
    GLM4_AIR = "glm-4-air"
    GLM4_FLASH = "glm-4-flash"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]


GLM_THINKING_SUPPORTED_FAMILIES = (
    GlmModelFamily.GLM5,
    GlmModelFamily.GLM47,
    GlmModelFamily.GLM46,
    GlmModelFamily.GLM45,
)

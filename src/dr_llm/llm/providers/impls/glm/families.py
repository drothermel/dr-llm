from __future__ import annotations

from enum import StrEnum


class GlmModelFamily(StrEnum):
    GLM5 = "glm-5"
    GLM47 = "glm-4.7"
    GLM46 = "glm-4.6"
    GLM45 = "glm-4.5"


GLM_THINKING_SUPPORTED_FAMILIES = (
    GlmModelFamily.GLM5,
    GlmModelFamily.GLM47,
    GlmModelFamily.GLM46,
    GlmModelFamily.GLM45,
)

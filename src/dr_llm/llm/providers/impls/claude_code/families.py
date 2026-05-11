from __future__ import annotations

from enum import StrEnum


class ClaudeCodeModelFamily(StrEnum):
    CLAUDE = "claude-"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


class ClaudeCodeStaticCatalogModel(StrEnum):
    CLAUDE_OPUS_46 = "claude-opus-4-6"
    CLAUDE_SONNET_46 = "claude-sonnet-4-6"
    CLAUDE_HAIKU_45_20251001 = "claude-haiku-4-5-20251001"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]


CLAUDE_CODE_SUPPORTED_MODEL_FAMILIES = (ClaudeCodeModelFamily.CLAUDE,)

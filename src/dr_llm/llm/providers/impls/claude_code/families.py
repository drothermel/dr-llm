from __future__ import annotations

from enum import StrEnum


class ClaudeCodeModelFamily(StrEnum):
    CLAUDE = "claude-"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


CLAUDE_CODE_SUPPORTED_MODEL_FAMILIES = (ClaudeCodeModelFamily.CLAUDE,)

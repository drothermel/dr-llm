from __future__ import annotations

from enum import StrEnum


class AnthropicModelFamily(StrEnum):
    CLAUDE_HAIKU_45 = "claude-haiku-4-5"
    CLAUDE_HAIKU_45_20251001 = "claude-haiku-4-5-20251001"
    CLAUDE_OPUS_46 = "claude-opus-4-6"
    CLAUDE_OPUS_46_20250514 = "claude-opus-4-6-20250514"
    CLAUDE_OPUS_45 = "claude-opus-4-5"
    CLAUDE_OPUS_45_20251101 = "claude-opus-4-5-20251101"
    CLAUDE_OPUS_41 = "claude-opus-4-1"
    CLAUDE_OPUS_41_20250805 = "claude-opus-4-1-20250805"
    CLAUDE_OPUS_4 = "claude-opus-4-"
    CLAUDE_OPUS_4_20250514 = "claude-opus-4-20250514"
    CLAUDE_SONNET_46 = "claude-sonnet-4-6"
    CLAUDE_SONNET_46_20250514 = "claude-sonnet-4-6-20250514"
    CLAUDE_SONNET_45 = "claude-sonnet-4-5"
    CLAUDE_SONNET_45_20250929 = "claude-sonnet-4-5-20250929"
    CLAUDE_SONNET_4 = "claude-sonnet-4-"
    CLAUDE_SONNET_4_20250514 = "claude-sonnet-4-20250514"
    CLAUDE_37_SONNET = "claude-3-7-sonnet"
    CLAUDE_37_SONNET_20250219 = "claude-3-7-sonnet-20250219"

    def in_family(self, model: str) -> bool:
        return model.startswith(self)


class AnthropicStaticCatalogModel(StrEnum):
    CLAUDE_OPUS_46 = "claude-opus-4-6"
    CLAUDE_SONNET_46 = "claude-sonnet-4-6"
    CLAUDE_HAIKU_45_20251001 = "claude-haiku-4-5-20251001"

    @classmethod
    def values(cls) -> list[str]:
        return [model.value for model in cls]


ANTHROPIC_BUDGET_CAPABILITY_FAMILIES = (
    AnthropicModelFamily.CLAUDE_OPUS_41,
    AnthropicModelFamily.CLAUDE_OPUS_4,
    AnthropicModelFamily.CLAUDE_SONNET_45,
    AnthropicModelFamily.CLAUDE_SONNET_4,
    AnthropicModelFamily.CLAUDE_37_SONNET,
    AnthropicModelFamily.CLAUDE_HAIKU_45,
)
ANTHROPIC_BUDGET_THINKING_SUPPORTED = (
    AnthropicModelFamily.CLAUDE_HAIKU_45_20251001,
    AnthropicModelFamily.CLAUDE_OPUS_45,
    AnthropicModelFamily.CLAUDE_OPUS_45_20251101,
    AnthropicModelFamily.CLAUDE_SONNET_45,
    AnthropicModelFamily.CLAUDE_SONNET_45_20250929,
    AnthropicModelFamily.CLAUDE_OPUS_41,
    AnthropicModelFamily.CLAUDE_OPUS_41_20250805,
    AnthropicModelFamily.CLAUDE_OPUS_4_20250514,
    AnthropicModelFamily.CLAUDE_SONNET_4_20250514,
    AnthropicModelFamily.CLAUDE_37_SONNET,
    AnthropicModelFamily.CLAUDE_37_SONNET_20250219,
)
ANTHROPIC_ADAPTIVE_THINKING_SUPPORTED = (
    AnthropicModelFamily.CLAUDE_SONNET_46,
    AnthropicModelFamily.CLAUDE_SONNET_46_20250514,
    AnthropicModelFamily.CLAUDE_OPUS_46,
    AnthropicModelFamily.CLAUDE_OPUS_46_20250514,
)
ANTHROPIC_EFFORT_SUPPORTED_MODELS = (
    AnthropicModelFamily.CLAUDE_SONNET_46,
    AnthropicModelFamily.CLAUDE_OPUS_46,
    AnthropicModelFamily.CLAUDE_OPUS_45_20251101,
)

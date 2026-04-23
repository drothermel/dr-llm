from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from marimo_utils.ui import BadgeVariant


class PaletteToneName(StrEnum):
    NEUTRAL = "neutral"
    SUCCESS = "success"
    WARNING = "warning"
    INFO = "info"
    DANGER = "danger"


class Style(BaseModel):
    model_config = ConfigDict(frozen=True)

    @classmethod
    def default(cls) -> Style:
        return cls()


def badge_variant_for_tone(tone: PaletteToneName) -> BadgeVariant:
    if tone == PaletteToneName.DANGER:
        return BadgeVariant.DESTRUCTIVE
    if tone == PaletteToneName.INFO:
        return BadgeVariant.OUTLINE
    if tone == PaletteToneName.WARNING:
        return BadgeVariant.SECONDARY
    if tone == PaletteToneName.SUCCESS:
        return BadgeVariant.DEFAULT
    return BadgeVariant.SECONDARY


def width_to_tailwind(width: str) -> str:
    width_map = {
        "18rem": "w-72",
        "20rem": "w-80",
    }
    return width_map.get(width, width if width.startswith("w-") else "w-80")


__all__ = [
    "PaletteToneName",
    "Style",
    "badge_variant_for_tone",
    "width_to_tailwind",
]

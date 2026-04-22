from __future__ import annotations

from mohtml import div  # type: ignore
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.models import PendingStatusCounts
from marimo_utils.style import (
    ColorPalette,
    DataItem,
    HtmlRenderable,
    PaletteToneName,
    SpacingScale,
    Typography,
)


class PendingDataItems(BaseModel):
    model_config = ConfigDict(frozen=True)

    pending_counts: PendingStatusCounts
    palette: ColorPalette
    typography: Typography
    spacing: SpacingScale
    field_tones: dict[str, PaletteToneName | None] = Field(
        default_factory=lambda: {
            "pending": PaletteToneName.WARNING,
            "failed": PaletteToneName.DANGER,
            "leased": None,
        }
    )

    def items(self) -> list[DataItem]:
        pending_counts_model = type(self.pending_counts)
        return [
            DataItem(
                palette=self.palette,
                typography=self.typography,
                spacing=self.spacing,
                label=field_name.title(),
                value=str(getattr(self.pending_counts, field_name)),
                value_tone=self.field_tones.get(field_name),
            )
            for field_name in pending_counts_model.model_fields
        ]

    def render(self) -> HtmlRenderable:
        return div(*[item.render() for item in self.items()])


__all__ = ["PendingDataItems"]

from __future__ import annotations

import marimo as mo
from pydantic import BaseModel, ConfigDict, Field, computed_field

from dr_llm.pool.models import PoolInspection
from dr_llm.style.pool_card import PoolCard
from marimo_utils.style import (
    Card,
    ColorPalette,
    PaletteToneName,
    PieChart,
    PieSlice,
    SpacingScale,
    Title,
    Typography,
)


class PiePoolCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    pool: PoolInspection
    palette: ColorPalette
    typography: Typography = Field(default_factory=Typography.default)
    spacing: SpacingScale = Field(default_factory=SpacingScale.default)
    width: str = "20rem"

    def base_card(self) -> PoolCard:
        return PoolCard(
            pool=self.pool,
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            width=self.width,
        )

    def pie_slices(self) -> list[PieSlice]:
        return [
            PieSlice(
                label="Samples",
                value=self.pool.sample_count,
                tone=PaletteToneName.SUCCESS,
            ),
            PieSlice(
                label="Pending",
                value=self.pool.pending_counts.pending,
                tone=PaletteToneName.WARNING,
            ),
            PieSlice(
                label="Leased",
                value=self.pool.pending_counts.leased,
                tone=PaletteToneName.INFO,
            ),
            PieSlice(
                label="Failed",
                value=self.pool.pending_counts.failed,
                tone=PaletteToneName.DANGER,
            ),
        ]

    @computed_field
    @property
    def total_samples(self) -> int:
        return sum(slice_.value for slice_ in self.pie_slices())

    def total_title(self) -> mo.Html:
        return Title(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            drop_text="Total Samples In Pie",
            text=f"{self.total_samples:,}",
        ).render()

    def pie_chart(self) -> object:
        return PieChart(
            palette=self.palette,
            typography=self.typography,
            slices=self.pie_slices(),
        ).render()

    def content(self) -> mo.Html:
        return mo.vstack(
            [self.total_title(), self.pie_chart()],
            align="center",
            gap=0.5,
        )

    def render(self) -> mo.Html:
        base = self.base_card()
        return Card(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            width=self.width,
            title=base.title().render(),
            header=base.header(),
            content=self.content(),
        ).render()


__all__ = ["PiePoolCard"]

from __future__ import annotations

from typing import Literal

import marimo as mo
from pydantic import computed_field

from dr_llm.style.pool_card import PoolCard
from marimo_utils.style import DataItem, PaletteToneName, PieChart, PieSlice


class PiePoolCard(PoolCard):
    card_type: Literal["Pie Pool"] = "Pie Pool"
    width: str = "20rem"

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

    def total_data_item(self) -> DataItem:
        return DataItem(
            style=self.style,
            label="Total",
            value=f"{self.total_samples:,}",
            value_tone=PaletteToneName.SUCCESS,
        )

    def pie_chart(self) -> PieChart:
        return PieChart(
            style=self.style,
            slices=self.pie_slices(),
            height=None,
        )

    def content(self) -> mo.Html:
        return mo.vstack(
            [self.total_data_item().render(), self.pie_chart()],
            align="center",
            gap=0.5,
        )


__all__ = ["PiePoolCard"]

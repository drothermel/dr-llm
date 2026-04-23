from __future__ import annotations

from typing import Literal

import marimo as mo
from pydantic import computed_field

from dr_llm.style.pool_card import PoolCard
from marimo_utils.ui import ChartColor, PieChart, PieSlice


class PiePoolCard(PoolCard):
    card_type: Literal["Pie Pool"] = "Pie Pool"
    width: str = "20rem"

    def divider(self) -> mo.Html:
        return mo.Html("<div class='w-full border-t border-border'></div>")

    def pie_slices(self) -> list[PieSlice]:
        return [
            PieSlice(
                label="Succeeded",
                value=self.pool.sample_count,
                color=ChartColor.TWO,
            ),
            PieSlice(
                label="Pending",
                value=self.pool.pending_counts.pending,
            ),
            PieSlice(
                label="Leased",
                value=self.pool.pending_counts.leased,
            ),
            PieSlice(
                label="Failed",
                value=self.pool.pending_counts.failed,
                color=ChartColor.ONE,
            ),
        ]

    @computed_field
    @property
    def total_samples(self) -> int:
        return sum(slice_.value for slice_ in self.pie_slices())

    def pie_chart(self) -> PieChart:
        return PieChart(
            slices=self.pie_slices(),
            height=None,
            title=f"Total Rows: {self.total_samples:,}",
        )

    def content(self) -> mo.Html:
        return mo.vstack(
            [
                self.axes_list().render(),
                self.divider(),
                self.pie_chart(),
            ],
            align="stretch",
            gap=0.75,
        )


__all__ = ["PiePoolCard"]

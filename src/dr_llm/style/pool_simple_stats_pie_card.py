from __future__ import annotations

import marimo as mo
from dr_widget.inline import ActiveHtml
from pydantic import BaseModel, ConfigDict, computed_field

from dr_llm.pool.models import PoolInspection
from dr_llm.style.components import AxesLabel, AxisBadge
from dr_llm.style.theme import width_to_tailwind
from marimo_utils.ui import Card, ChartColor, PieChart, PieSlice


def _titleize(value: str) -> str:
    return value.replace("_", " ").title()


class PoolSimpleStatsPieCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    pool: PoolInspection
    width: str = "20rem"

    def title_text(self) -> str:
        return _titleize(self.pool.name)

    def description_text(self) -> str:
        return f"Project: {self.pool.project_name}"

    def axes_list(self) -> AxesLabel:
        return AxesLabel(
            items=[
                AxisBadge(label=column.name).render()
                for column in self.pool.pool_schema.key_columns
            ],
        )

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

    def render(self) -> mo.Html | ActiveHtml:
        return Card(
            width=width_to_tailwind(self.width),
            title=self.title_text(),
            description=self.description_text(),
            content=self.content(),
        ).render()


__all__ = ["PoolSimpleStatsPieCard"]

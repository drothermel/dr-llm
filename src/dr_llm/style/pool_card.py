from __future__ import annotations

from typing import Literal

import marimo as mo
from dr_widget.inline import ActiveHtml
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.models import PoolInspection
from dr_llm.style.components import PendingDataItems
from marimo_utils.style import (
    Badge,
    Card,
    DataItem,
    DateStamp,
    LabeledList,
    PaletteToneName,
    ProjectStamp,
    Style,
    Title,
)


def norm_str(value: str) -> str:
    return value.replace("_", " ").title()


class PoolCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    card_type: Literal["Pool"] = "Pool"
    pool: PoolInspection
    style: Style = Field(default_factory=Style.default)
    width: str = "18rem"

    def status_tone_name(self) -> PaletteToneName:
        if self.pool.status.value == "complete":
            return PaletteToneName.SUCCESS
        if self.pool.status.value == "in_progress":
            return PaletteToneName.WARNING
        return PaletteToneName.NEUTRAL

    def title(self) -> Title:
        return Title(
            style=self.style,
            drop_text=f"{norm_str(self.card_type)} Card",
            text=norm_str(self.pool.name),
        )

    def project_stamp(self) -> ProjectStamp:
        return ProjectStamp(
            style=self.style,
            project_name=self.pool.project_name,
        )

    def created_stamp(self) -> DateStamp:
        return DateStamp(
            style=self.style,
            value=self.pool.created_at,
        )

    def status_badge(self) -> Badge:
        return Badge(
            style=self.style,
            label=self.pool.status.value.replace("_", " "),
            tone=self.status_tone_name(),
        )

    def header(self) -> mo.Html:
        return mo.vstack(
            [
                mo.hstack(
                    [
                        self.status_badge().render(),
                        self.project_stamp().render(),
                        self.created_stamp().render(),
                    ],
                    justify="start",
                    align="center",
                    wrap=True,
                    gap=0.4,
                ),
                self.axes_list().render(),
            ],
            gap=0,
        )

    def axes_list(self) -> LabeledList:
        return LabeledList(
            style=self.style,
            section_label="Axes",
            items=[
                Badge(style=self.style, label=column.name).render()
                for column in self.pool.pool_schema.key_columns
            ],
        )

    def pending_data_items(self) -> PendingDataItems:
        return PendingDataItems(
            pending_counts=self.pool.pending_counts,
            style=self.style,
        )

    def samples_data_item(self) -> DataItem:
        return DataItem(
            style=self.style,
            label="Samples",
            value=f"{self.pool.sample_count:,}",
            value_tone=PaletteToneName.SUCCESS,
        )

    def content(self) -> mo.Html:
        return mo.vstack(
            [self.samples_data_item().render(), self.pending_data_items().render()],
            gap=0,
        )

    def render(self) -> mo.Html | ActiveHtml:
        return Card(
            style=self.style,
            width=self.width,
            title=self.title().render(),
            header=self.header(),
            content=self.content(),
        ).render()


__all__ = ["PoolCard"]

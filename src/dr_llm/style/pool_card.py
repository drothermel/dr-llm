from __future__ import annotations

from typing import Literal

import marimo as mo
from dr_widget.inline import ActiveHtml
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.models import PoolInspection
from dr_llm.style.components import AxesLabel, AxisBadge, PendingDataItems
from dr_llm.style.theme import (
    PaletteToneName,
    Style,
    badge_variant_for_tone,
    width_to_tailwind,
)
from marimo_utils.ui import (
    Badge,
    Card,
    DataItem,
    DateStamp,
    ProjectStamp,
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

    def title_text(self) -> str:
        return norm_str(self.pool.name)

    def description_text(self) -> str:
        return f"Project: {self.pool.project_name}"

    def project_stamp(self) -> ProjectStamp:
        return ProjectStamp(
            project_name=self.pool.project_name,
        )

    def created_stamp(self) -> DateStamp:
        return DateStamp(
            value=self.pool.created_at,
        )

    def status_badge(self) -> Badge:
        return Badge(
            label=self.pool.status.value.replace("_", " "),
            variant=badge_variant_for_tone(self.status_tone_name()),
        )

    def axes_list(self) -> AxesLabel:
        return AxesLabel(
            items=[
                AxisBadge(label=column.name).render()
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
            label="Samples",
            value=f"{self.pool.sample_count:,}",
        )

    def content(self) -> mo.Html:
        return mo.vstack(
            [
                mo.hstack(
                    [
                        self.status_badge().render(),
                        self.created_stamp().render(),
                    ],
                    justify="start",
                    align="center",
                    wrap=True,
                    gap=0.4,
                ),
                self.axes_list().render(),
                self.samples_data_item().render(),
                self.pending_data_items().render(),
            ],
            gap=0.75,
        )

    def render(self) -> mo.Html | ActiveHtml:
        return Card(
            width=width_to_tailwind(self.width),
            title=self.title_text(),
            description=self.description_text(),
            content=self.content(),
        ).render()


__all__ = ["PoolCard"]

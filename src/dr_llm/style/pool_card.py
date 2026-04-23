from __future__ import annotations

from typing import Literal

import marimo as mo
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.models import PoolInspection
from dr_llm.style.components import PendingDataItems
from marimo_utils.style import (
    Badge,
    Card,
    ColorPalette,
    DataItem,
    DateStamp,
    LabeledList,
    PaletteToneName,
    ProjectStamp,
    SpacingScale,
    Title,
    Typography,
)


def norm_str(value: str) -> str:
    return value.replace("_", " ").title()


class PoolCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    card_type: Literal["Pool"] = "Pool"
    pool: PoolInspection
    palette: ColorPalette
    typography: Typography = Field(default_factory=Typography.default)
    spacing: SpacingScale = Field(default_factory=SpacingScale.default)
    width: str = "18rem"

    def status_tone_name(self) -> PaletteToneName:
        if self.pool.status.value == "complete":
            return PaletteToneName.SUCCESS
        if self.pool.status.value == "in_progress":
            return PaletteToneName.WARNING
        return PaletteToneName.NEUTRAL

    def title(self) -> Title:
        return Title(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            drop_text=f"{norm_str(self.card_type)} Card",
            text=norm_str(self.pool.name),
        )

    def project_stamp(self) -> ProjectStamp:
        return ProjectStamp(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            project_name=self.pool.project_name,
        )

    def created_stamp(self) -> DateStamp:
        return DateStamp(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            value=self.pool.created_at,
        )

    def status_badge(self) -> Badge:
        return Badge(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
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
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            section_label="Axes",
            items=[
                Badge(
                    palette=self.palette,
                    typography=self.typography,
                    spacing=self.spacing,
                    label=column.name,
                ).render()
                for column in self.pool.pool_schema.key_columns
            ],
        )

    def pending_data_items(self) -> PendingDataItems:
        return PendingDataItems(
            pending_counts=self.pool.pending_counts,
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
        )

    def samples_data_item(self) -> DataItem:
        return DataItem(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            label="Samples",
            value=f"{self.pool.sample_count:,}",
            value_tone=PaletteToneName.SUCCESS,
        )

    def content(self) -> mo.Html:
        return mo.vstack(
            [self.samples_data_item().render(), self.pending_data_items().render()],
            gap=0,
        )

    def render(self) -> mo.Html:
        return Card(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            width=self.width,
            title=self.title().render(),
            header=self.header(),
            content=self.content(),
        ).render()


__all__ = ["PoolCard"]

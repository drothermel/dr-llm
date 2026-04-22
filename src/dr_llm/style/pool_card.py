from __future__ import annotations

from typing import Literal

import marimo as mo
from mohtml import div  # type: ignore
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.models import PoolInspection
from dr_llm.style.card import Card
from dr_llm.style.components import (
    Badge,
    DataItem,
    DateStamp,
    LabeledList,
    PendingDataItems,
    ProjectStamp,
    Title,
)
from dr_llm.style.settings import (
    ColorPalette,
    LayoutToken,
    PaletteToneName,
    SpacingScale,
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
    header_display_styles: list[LayoutToken] = Field(
        default_factory=lambda: [
            LayoutToken.FLEX,
            LayoutToken.FLEX_WRAP,
            LayoutToken.ALIGN_CENTER,
        ]
    )

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

    def header(self) -> div:
        return div(
            div(
                self.status_badge().render(),
                self.project_stamp().render(),
                self.created_stamp().render(),
                style=(
                    f"margin-top: {self.spacing.sm}; "
                    f"gap: {self.spacing.md}; "
                    f"{LayoutToken.css(self.header_display_styles)}"
                ),
            ),
            self.axes_list().render(),
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

    def content(self) -> div:
        return div(
            DataItem(
                palette=self.palette,
                typography=self.typography,
                spacing=self.spacing,
                label="Samples",
                value=f"{self.pool.sample_count:,}",
                value_tone=PaletteToneName.SUCCESS,
            ).render(),
            DataItem(
                palette=self.palette,
                typography=self.typography,
                spacing=self.spacing,
                label="In flight",
                value=str(self.pool.in_flight),
                value_tone=PaletteToneName.INFO,
            ).render(),
            self.pending_data_items().render(),
        )

    def render(self) -> div:
        return Card(
            palette=self.palette,
            typography=self.typography,
            spacing=self.spacing,
            width=self.width,
            title=self.title(),
            header=self.header(),
            content=self.content(),
        ).render()

    def render_html(self) -> mo.Html:
        return mo.Html(str(self.render()))


__all__ = ["PoolCard"]

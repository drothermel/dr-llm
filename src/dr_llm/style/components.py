from __future__ import annotations

from datetime import datetime

from mohtml import div, p, path, rect, span, svg  # type: ignore
from pydantic import BaseModel, ConfigDict, Field

from dr_llm.pool.models import PendingStatusCounts
from dr_llm.style.settings import (
    ColorPalette,
    IconStyle,
    LayoutToken,
    PaletteToneName,
    SpacingScale,
    Typography,
)


class MetaStamp(BaseModel):
    model_config = ConfigDict(frozen=True)

    palette: ColorPalette
    typography: Typography
    spacing: SpacingScale
    icon_style: IconStyle = Field(default_factory=IconStyle.default)
    display_styles: list[LayoutToken] = Field(
        default_factory=lambda: [
            LayoutToken.INLINE_FLEX,
            LayoutToken.ALIGN_CENTER,
        ]
    )

    def icon(self) -> span:
        raise NotImplementedError("MetaStamp subclasses must implement icon().")

    def text(self) -> str:
        raise NotImplementedError("MetaStamp subclasses must implement text().")

    def render(self) -> div:
        return div(
            self.icon(),
            span(
                self.text(),
                style=self.typography.meta.css(color=self.palette.text_subtle),
            ),
            style=(
                f"margin-top: {self.spacing.sm}; "
                f"gap: {self.spacing.sm}; "
                f"{LayoutToken.css(self.display_styles)}"
            ),
        )


class DateStamp(MetaStamp):
    value: datetime | None

    def icon(self) -> svg:
        return svg(
            path(d="M8 2v4"),
            path(d="M16 2v4"),
            rect(width="18", height="18", x="3", y="4", rx="2"),
            path(d="M3 10h18"),
            **self.icon_style.svg_kwargs(),
            style=self.icon_style.css(color=self.palette.text_subtle),
        )

    def text(self) -> str:
        if self.value is None:
            return "--- --"
        return self.value.strftime("%b %d")


class ProjectStamp(MetaStamp):
    project_name: str

    def icon(self) -> svg:
        return svg(
            path(
                d=(
                    "M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9"
                    "L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
                )
            ),
            **self.icon_style.svg_kwargs(),
            style=self.icon_style.css(color=self.palette.text_subtle),
        )

    def text(self) -> str:
        return self.project_name


class Badge(BaseModel):
    model_config = ConfigDict(frozen=True)

    palette: ColorPalette
    typography: Typography
    spacing: SpacingScale
    label: str
    tone: PaletteToneName = PaletteToneName.INFO
    border_radius: str = "999px"
    border_type: str = "border: 1px solid"
    display_styles: list[LayoutToken] = Field(
        default_factory=lambda: [LayoutToken.INLINE_BLOCK, LayoutToken.NOWRAP]
    )

    def render(self) -> span:
        tone = self.palette.tone(self.tone)
        return span(
            self.label,
            style=(
                f"{LayoutToken.css(self.display_styles)}"
                f"padding: {self.spacing.xs} {self.spacing.md}; "
                f"border-radius: {self.border_radius}; "
                f"background: {tone.bg}; "
                f"{self.border_type} {tone.border}; "
                f"{self.typography.badge.css(color=tone.text)}"
            ),
        )


class DataItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    palette: ColorPalette
    typography: Typography
    spacing: SpacingScale
    label: str
    value: str
    value_tone: PaletteToneName | None = None
    label_min_width: str = "7rem"
    label_display_styles: list[LayoutToken] = Field(
        default_factory=lambda: [LayoutToken.INLINE_BLOCK]
    )

    def value_color(self) -> str:
        if self.value_tone is None:
            return self.palette.text_primary
        return self.palette.tone(self.value_tone).text

    def render(self) -> div:
        return div(
            span(
                self.label,
                style=(
                    f"{LayoutToken.css(self.label_display_styles)}"
                    f"min-width: {self.label_min_width}; "
                    f"{self.typography.label.css(color=self.palette.text_muted)}"
                ),
            ),
            span(
                self.value,
                style=self.typography.body.css(color=self.value_color()),
            ),
            style=(f"margin-top: {self.spacing.md}; "),
        )


class Title(BaseModel):
    model_config = ConfigDict(frozen=True)

    palette: ColorPalette
    typography: Typography
    spacing: SpacingScale
    drop_text: str
    text: str
    drop_text_margin: str = "0"
    text_margin_inline: str = "0"
    text_margin_bottom: str = "0"

    def render(self) -> div:
        return div(
            p(
                self.drop_text,
                style=(
                    f"margin: {self.drop_text_margin}; "
                    f"{self.typography.drop_title.css(color=self.palette.text_subtle)}"
                ),
            ),
            p(
                self.text,
                style=(
                    f"margin: {self.spacing.xxs} {self.text_margin_inline} {self.text_margin_bottom}; "
                    f"{self.typography.title.css(color=self.palette.text_primary)}"
                ),
            ),
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
            "promoted": None,
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

    def render(self) -> div:
        return div(*[item.render() for item in self.items()])


class LabeledList(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    palette: ColorPalette
    typography: Typography
    spacing: SpacingScale
    section_label: str
    items: list[object]
    display_styles: list[LayoutToken] = Field(
        default_factory=lambda: [
            LayoutToken.FLEX,
            LayoutToken.FLEX_WRAP,
            LayoutToken.ALIGN_CENTER,
        ]
    )

    def render(self) -> div:
        return div(
            span(
                f"{self.section_label}:",
                style=self.typography.label.css(color=self.palette.text_muted),
            ),
            *self.items,
            style=(
                f"margin-top: {self.spacing.lg}; "
                f"gap: {self.spacing.sm}; "
                f"line-height: {self.spacing.line_height_loose}; "
                f"{LayoutToken.css(self.display_styles)}"
            ),
        )


__all__ = [
    "Badge",
    "DataItem",
    "DateStamp",
    "LabeledList",
    "MetaStamp",
    "PendingDataItems",
    "ProjectStamp",
    "Title",
]

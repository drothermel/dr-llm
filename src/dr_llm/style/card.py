from __future__ import annotations

from mohtml import div  # type: ignore
from pydantic import BaseModel, ConfigDict

from dr_llm.style.components import Title
from dr_llm.style.settings import ColorPalette, SpacingScale, Typography


class Card(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    palette: ColorPalette
    typography: Typography
    spacing: SpacingScale
    title: Title | None = None
    header: div | None = None
    content: div | None = None
    width: str = "18rem"
    border_radius: str = "16px"
    border_type: str = "1px solid"
    divider_border_type: str = "1px solid"

    def divider(self) -> div:
        return div(
            style=(
                f"margin-top: {self.spacing.lg}; "
                f"padding-top: {self.spacing.sm}; "
                f"border-top: {self.divider_border_type} {self.palette.surface_border};"
            ),
        )

    def render(self) -> div:
        top_sections: list[div] = []
        if self.title is not None:
            top_sections.append(self.title.render())
        if self.header is not None:
            top_sections.append(self.header)

        sections: list[div] = [*top_sections]
        if top_sections and self.content is not None:
            sections.append(self.divider())
        if self.content is not None:
            sections.append(self.content)

        return div(
            *sections,
            style=(
                f"font-family: {self.typography.font_family}; "
                f"color: {self.palette.text_primary}; "
                f"width: {self.width}; "
                f"padding: {self.spacing.xl} {self.spacing.xxl}; "
                f"border-radius: {self.border_radius}; "
                f"border: {self.border_type} {self.palette.surface_border}; "
                f"background: {self.palette.surface_background}; "
                f"box-shadow: {self.palette.surface_shadow};"
            ),
        )


__all__ = ["Card"]

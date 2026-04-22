import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import sys
    from datetime import datetime
    from enum import StrEnum
    from pathlib import Path

    import marimo as mo
    from mohtml import div, p, path, rect, span, svg
    from pydantic import BaseModel, ConfigDict, Field
    from typing import Literal

    marimo_utils_src = Path(__file__).resolve().parents[2] / "marimo_utils" / "src"
    if str(marimo_utils_src) not in sys.path:
        sys.path.insert(0, str(marimo_utils_src))

    from marimo_utils import add_marimo_display
    from dr_llm.pool.admin_service import (
        assess_pool_creation,
        create_pool as create_pool_service,
        inspect_pool,
    )
    from dr_llm.pool.models import (
        CreatePoolRequest,
        PendingStatusCounts,
        PoolInspection,
        PoolInspectionRequest,
    )
    from dr_llm.project.models import (
        CreateProjectRequest,
        ProjectPoolInspectionStatus,
    )
    from dr_llm.project.project_info import ProjectInfo
    from dr_llm.project.project_service import (
        assess_project_creation,
        create_project as create_project_service,
        inspect_projects,
    )

    add_marimo_display()(ProjectInfo)


@app.function
def norm_str(st):
    return st.replace("_", " ").title()


@app.class_definition
class TonePalette(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    bg: str
    border: str


@app.class_definition
class PaletteToneName(StrEnum):
    NEUTRAL = "neutral"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"


@app.class_definition
class ColorPalette(BaseModel):
    model_config = ConfigDict(frozen=True)

    text_primary: str
    text_muted: str
    text_subtle: str
    surface_background: str
    surface_border: str
    surface_shadow: str
    neutral: TonePalette
    info: TonePalette
    success: TonePalette
    warning: TonePalette
    danger: TonePalette

    def tone(self, tone_name: PaletteToneName) -> TonePalette:
        return getattr(self, tone_name.value)

    @classmethod
    def default(cls) -> "ColorPalette":
        return cls(
            text_primary="#0f172a",
            text_muted="#475569",
            text_subtle="#64748b",
            surface_background="linear-gradient(160deg, #f8fafc 0%, #eef4ff 62%, #fff7ed 100%)",
            surface_border="rgba(148, 163, 184, 0.18)",
            surface_shadow="0 10px 24px rgba(15, 23, 42, 0.07)",
            neutral=TonePalette(
                text="#334155",
                bg="#e2e8f0",
                border="#475569",
            ),
            info=TonePalette(
                text="#1d4ed8",
                bg="#dbeafe",
                border="#2563eb",
            ),
            success=TonePalette(
                text="#0f766e",
                bg="#ccfbf1",
                border="#115e59",
            ),
            warning=TonePalette(
                text="#9a3412",
                bg="#ffedd5",
                border="#c2410c",
            ),
            danger=TonePalette(
                text="#b91c1c",
                bg="#fee2e2",
                border="#b91c1c",
            ),
        )


@app.class_definition
class TextStyle(BaseModel):
    model_config = ConfigDict(frozen=True)

    font_size: str
    font_weight: int
    letter_spacing: str = "0"
    line_height: str = "1.2"
    text_transform: str | None = None

    def css(self, *, color: str | None = None) -> str:
        parts = [
            f"font-size: {self.font_size}",
            f"font-weight: {self.font_weight}",
            f"letter-spacing: {self.letter_spacing}",
            f"line-height: {self.line_height}",
        ]
        if self.text_transform is not None:
            parts.append(f"text-transform: {self.text_transform}")
        if color is not None:
            parts.append(f"color: {color}")
        return "; ".join(parts) + ";"


@app.class_definition
class IconStyle(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: str
    height: str
    view_box: str
    fill: str
    stroke: str
    stroke_width: str
    stroke_linecap: str
    stroke_linejoin: str
    flex: str = "0 0 auto"

    def svg_kwargs(self) -> dict[str, str]:
        return {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": self.width,
            "height": self.height,
            "viewBox": self.view_box,
            "fill": self.fill,
            "stroke": self.stroke,
            "stroke_width": self.stroke_width,
            "stroke_linecap": self.stroke_linecap,
            "stroke_linejoin": self.stroke_linejoin,
        }

    def css(self, *, color: str | None = None) -> str:
        parts = [f"flex: {self.flex}"]
        if color is not None:
            parts.insert(0, f"color: {color}")
        return "; ".join(parts) + ";"

    @classmethod
    def default(cls) -> "IconStyle":
        return cls(
            width="14",
            height="14",
            view_box="0 0 24 24",
            fill="none",
            stroke="currentColor",
            stroke_width="2",
            stroke_linecap="round",
            stroke_linejoin="round",
        )


@app.class_definition
class LayoutToken(StrEnum):
    FLEX = "display: flex"
    INLINE_FLEX = "display: inline-flex"
    INLINE_BLOCK = "display: inline-block"
    NOWRAP = "white-space: nowrap"
    FLEX_WRAP = "flex-wrap: wrap"
    ALIGN_CENTER = "align-items: center"

    @classmethod
    def css(cls, tokens: list["LayoutToken"]) -> str:
        if not tokens:
            return ""
        return "; ".join(token.value for token in tokens) + ";"


@app.class_definition
class Typography(BaseModel):
    model_config = ConfigDict(frozen=True)

    font_family: str
    title: TextStyle
    drop_title: TextStyle
    body: TextStyle
    meta: TextStyle
    badge: TextStyle
    label: TextStyle

    @classmethod
    def default(cls) -> "Typography":
        return cls(
            font_family=(
                "ui-sans-serif, -apple-system, BlinkMacSystemFont, "
                "'Segoe UI', sans-serif"
            ),
            title=TextStyle(
                font_size="1rem",
                font_weight=800,
                line_height="1.15",
            ),
            drop_title=TextStyle(
                font_size="0.66rem",
                font_weight=700,
                letter_spacing="0.09em",
                line_height="1.1",
                text_transform="uppercase",
            ),
            body=TextStyle(
                font_size="0.82rem",
                font_weight=600,
                line_height="1.2",
            ),
            meta=TextStyle(
                font_size="0.68rem",
                font_weight=600,
                letter_spacing="0.04em",
                line_height="1.2",
            ),
            badge=TextStyle(
                font_size="0.68rem",
                font_weight=600,
                line_height="1.2",
            ),
            label=TextStyle(
                font_size="0.68rem",
                font_weight=700,
                letter_spacing="0.06em",
                line_height="1.2",
                text_transform="uppercase",
            ),
        )


@app.class_definition
class SpacingScale(BaseModel):
    model_config = ConfigDict(frozen=True)

    xxs: str
    xs: str
    sm: str
    md: str
    lg: str
    xl: str
    xxl: str
    line_height_loose: str

    @classmethod
    def default(cls) -> "SpacingScale":
        return cls(
            xxs="0.08rem",
            xs="0.12rem",
            sm="0.28rem",
            md="0.42rem",
            lg="0.55rem",
            xl="0.8rem",
            xxl="0.9rem",
            line_height_loose="1.8",
        )


@app.class_definition
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
        raise NotImplementedError(
            "MetaStamp subclasses must implement icon()."
        )

    def text(self) -> str:
        raise NotImplementedError(
            "MetaStamp subclasses must implement text()."
        )

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


@app.class_definition
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


@app.class_definition
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


@app.class_definition(column=1)
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


@app.class_definition
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


@app.class_definition
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


@app.class_definition
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


@app.class_definition
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


@app.class_definition(column=2)
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


@app.class_definition
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


@app.cell(column=3, hide_code=True)
def _():
    demo_pool_inspection = PoolInspection(
        project_name="demo_project",
        name="code_comp_demo",
        pool_schema={
            "name": "code_comp_demo",
            "key_columns": [
                {"name": "provider"},
                {"name": "model"},
                {"name": "reasoning_mode"},
            ],
        },
        created_at=datetime(2026, 4, 21, 14, 30),
        sample_count=1280,
        pending_counts={
            "pending": 36,
            "leased": 8,
            "promoted": 120,
            "failed": 3,
        },
        status="in_progress",
    )
    mo.vstack(
        [
            mo.md("## Pool Card Demo"),
            mo.Html(
                str(
                    PoolCard(
                        pool=demo_pool_inspection,
                        palette=ColorPalette.default(),
                        width="20rem",
                    ).render()
                )
            ),
        ]
    )
    return


@app.cell(column=4, hide_code=True)
def _(create_pool_form, create_project_form):
    _ = (create_project_form.value, create_pool_form.value)
    mo.vstack(
        [
            mo.md("## Docker Projects State"),
            mo.ui.table([summary.to_row() for summary in inspect_projects()]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(create_project_form):
    # Create Project Executor
    mo.stop(create_project_form.value is None)

    create_project_request = CreateProjectRequest(**create_project_form.value)
    create_project_readiness = assess_project_creation(
        create_project_request,
        cooldown_seconds=60,
    )
    if not create_project_readiness.allowed:
        assert create_project_readiness.blocked_message is not None
        raise ValueError(create_project_readiness.blocked_message)

    create_project_service(create_project_request)
    return


@app.cell(hide_code=True)
def _(create_pool_form):
    # Create Pool Executor
    mo.stop(create_pool_form.value is None)

    create_pool_request = CreatePoolRequest.from_csv(**create_pool_form.value)
    create_pool_readiness = assess_pool_creation(
        create_pool_request,
        max_pools_per_project=5,
    )
    if not create_pool_readiness.allowed:
        assert create_pool_readiness.blocked_message is not None
        raise ValueError(create_pool_readiness.blocked_message)

    create_pool_service(create_pool_request)
    return


@app.cell(hide_code=True)
def _(get_pool_info_form):
    # Get Pool Info Executor
    mo.stop(get_pool_info_form.value is None)
    pool_inspection = inspect_pool(
        PoolInspectionRequest(**get_pool_info_form.value)
    )
    return (pool_inspection,)


@app.cell(column=5, hide_code=True)
def _():
    create_project_form = (
        mo.md(
            """
            **Selections for Project Creation**

            {project_name}
            """
        )
        .batch(
            project_name=mo.ui.text(
                label="Project name",
                placeholder="demo_project",
            ),
        )
        .form(
            submit_button_label="Create project",
        )
    )

    create_project_form
    return (create_project_form,)


@app.cell(hide_code=True)
def _():
    create_pool_form = (
        mo.md(
            """
            **Selections for Pool Creation**

            {project_name}

            {pool_name}

            {axes_csv}
            """
        )
        .batch(
            project_name=mo.ui.text(
                label="Project name",
                placeholder="code_comp_v0",
            ),
            pool_name=mo.ui.text(
                label="Pool name",
                placeholder="demo_pool",
            ),
            axes_csv=mo.ui.text(
                label="Key axes (comma separated)",
                placeholder="provider, model",
                full_width=True,
            ),
        )
        .form(
            submit_button_label="Create pool",
        )
    )

    create_pool_form
    return (create_pool_form,)


@app.cell(hide_code=True)
def _():
    get_pool_info_form = (
        mo.md(
            """
            **Selections for Pool Info**

            {project_name}

            {pool_name}
            """
        )
        .batch(
            project_name=mo.ui.text(
                label="Project name",
                placeholder="code_comp_v0",
            ),
            pool_name=mo.ui.text(
                label="Pool name",
                placeholder="demo_pool",
            ),
        )
        .form(
            submit_button_label="Get pool info",
        )
    )

    get_pool_info_form
    return (get_pool_info_form,)


@app.cell(hide_code=True)
def _(default_card_palette, pool_inspection):
    mo.vstack(
        [
            mo.md("## Pool Inspection"),
            mo.Html(
                str(
                    PoolCard(
                        pool=pool_inspection,
                        palette=default_card_palette,
                    ).render()
                )
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

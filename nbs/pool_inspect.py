import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import sys
    from datetime import datetime
    from enum import StrEnum
    from pathlib import Path

    import marimo as mo
    from mohtml import div, p, span
    from pydantic import BaseModel, ConfigDict, Field

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
class Badge(BaseModel):
    model_config = ConfigDict(frozen=True)

    palette: ColorPalette
    label: str
    tone: PaletteToneName = PaletteToneName.INFO

    def render(self) -> span:
        tone = self.palette.tone(self.tone)
        return span(
            self.label,
            style=(
                "display: inline-block; padding: 0.12rem 0.42rem; "
                f"border-radius: 999px; background: {tone.bg}; "
                f"color: {tone.text}; border: 1px solid {tone.border}; "
                "font-size: 0.68rem; font-weight: 600;"
            ),
        )


@app.class_definition
class DataItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    palette: ColorPalette
    label: str
    value: str
    value_tone: PaletteToneName | None = None

    def value_color(self) -> str:
        if self.value_tone is None:
            return self.palette.text_primary
        return self.palette.tone(self.value_tone).text

    def render(self) -> div:
        return div(
            span(
                self.label,
                style=(
                    f"display: inline-block; min-width: 7rem; color: {self.palette.text_muted}; "
                    "font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; "
                    "font-weight: 700;"
                ),
            ),
            span(
                self.value,
                style=(
                    f"color: {self.value_color()}; font-size: 0.82rem; font-weight: 600;"
                ),
            ),
            style="margin-top: 0.38rem;",
        )


@app.class_definition
class MetaStamp(BaseModel):
    model_config = ConfigDict(frozen=True)

    palette: ColorPalette

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
                style=(
                    f"color: {self.palette.text_subtle}; font-size: 0.68rem; font-weight: 600; "
                    "letter-spacing: 0.04em;"
                ),
            ),
            style="margin-top: 0.28rem; display: inline-flex; align-items: center; gap: 0.32rem;",
        )


@app.class_definition
class DateStamp(MetaStamp):
    value: datetime | None

    def icon(self) -> span:
        return span(
            span(
                style=(
                    f"display: block; height: 0.22rem; background: {self.palette.text_subtle}; "
                    f"border-bottom: 1px solid {self.palette.text_subtle};"
                ),
            ),
            span(
                style="display: block; flex: 1; background: rgba(255, 255, 255, 0.45);",
            ),
            style=(
                "display: inline-flex; flex: 0 0 auto; flex-direction: column; overflow: hidden; "
                "width: 0.8rem; height: 0.8rem; border-radius: 0.22rem; box-sizing: border-box; "
                f"border: 1px solid {self.palette.text_subtle};"
            ),
        )

    def text(self) -> str:
        if self.value is None:
            return "--- --"
        return self.value.strftime("%b %d")


@app.class_definition
class ProjectStamp(MetaStamp):
    project_name: str

    def icon(self) -> span:
        return span(
            span(
                style=(
                    "position: absolute; left: 0.1rem; top: -0.16rem; width: 0.34rem; height: 0.18rem; "
                    f"border: 1px solid {self.palette.text_subtle}; border-bottom: none; "
                    f"border-radius: 0.14rem 0.14rem 0 0; background: rgba(255, 255, 255, 0.45);"
                ),
            ),
            style=(
                "position: relative; display: inline-block; width: 0.84rem; height: 0.56rem; "
                f"border: 1px solid {self.palette.text_subtle}; border-radius: 0.14rem; "
                f"background: rgba(255, 255, 255, 0.45); box-sizing: border-box;"
            ),
        )

    def text(self) -> str:
        return self.project_name


@app.class_definition
class PendingDataItems(BaseModel):
    model_config = ConfigDict(frozen=True)

    pending_counts: PendingStatusCounts
    palette: ColorPalette
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
                label=field_name.title(),
                value=str(getattr(self.pending_counts, field_name)),
                value_tone=self.field_tones.get(field_name),
            )
            for field_name in pending_counts_model.model_fields
        ]

    def render(self) -> div:
        return div(*[item.render() for item in self.items()])


@app.class_definition
class BadgeList(BaseModel):
    model_config = ConfigDict(frozen=True)

    palette: ColorPalette
    section_label: str
    items: list[Badge]

    def render(self) -> div:
        return div(
            span(
                f"{self.section_label}:",
                style=(
                    f"color: {self.palette.text_muted}; font-size: 0.68rem; text-transform: uppercase; "
                    "letter-spacing: 0.06em; font-weight: 700; margin-right: 0.35rem;"
                ),
            ),
            *[
                div(
                    item.render(),
                    style="display: inline-block; margin-right: 0.28rem;",
                )
                for item in self.items
            ],
            style="margin-top: 0.55rem; line-height: 1.8;",
        )


@app.class_definition(column=1)
class PoolCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    pool: PoolInspection
    palette: ColorPalette

    def status_tone_name(self) -> PaletteToneName:
        if self.pool.status.value == "complete":
            return PaletteToneName.SUCCESS
        if self.pool.status.value == "in_progress":
            return PaletteToneName.WARNING
        return PaletteToneName.NEUTRAL

    def project_stamp(self) -> ProjectStamp:
        return ProjectStamp(
            palette=self.palette,
            project_name=self.pool.project_name,
        )

    def created_stamp(self) -> DateStamp:
        return DateStamp(
            palette=self.palette,
            value=self.pool.created_at,
        )

    def status_badge(self) -> Badge:
        return Badge(
            palette=self.palette,
            label=self.pool.status.value.replace("_", " "),
            tone=self.status_tone_name(),
        )

    def axes_badges(self) -> BadgeList:
        return BadgeList(
            palette=self.palette,
            section_label="Axes",
            items=[
                Badge(
                    palette=self.palette,
                    label=column.name,
                )
                for column in self.pool.pool_schema.key_columns
            ],
        )

    def pending_data_items(self) -> PendingDataItems:
        return PendingDataItems(
            pending_counts=self.pool.pending_counts,
            palette=self.palette,
        )

    def render(self) -> div:
        return div(
            div(
                p(
                    "Pool Card",
                    style=(
                        f"margin: 0; color: {self.palette.text_subtle}; font-size: 0.66rem; "
                        "font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;"
                    ),
                ),
                p(
                    self.pool.name,
                    style=(
                        f"margin: 0.08rem 0 0; color: {self.palette.text_primary}; font-size: 1rem; "
                        "line-height: 1.15; font-weight: 800;"
                    ),
                ),
                div(
                    self.status_badge().render(),
                    self.project_stamp().render(),
                    self.created_stamp().render(),
                    style="margin-top: 0.28rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.42rem;",
                ),
            ),
            self.axes_badges().render(),
            div(
                DataItem(
                    palette=self.palette,
                    label="Samples",
                    value=f"{self.pool.sample_count:,}",
                    value_tone=PaletteToneName.SUCCESS,
                ).render(),
                DataItem(
                    palette=self.palette,
                    label="In flight",
                    value=str(self.pool.in_flight),
                    value_tone=PaletteToneName.INFO,
                ).render(),
                self.pending_data_items().render(),
                style=(
                    "margin-top: 0.55rem; padding-top: 0.28rem; "
                    f"border-top: 1px solid {self.palette.surface_border};"
                ),
            ),
            style=(
                "font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
                f"color: {self.palette.text_primary}; width: 18rem; padding: 0.8rem 0.9rem; border-radius: 16px; "
                f"border: 1px solid {self.palette.surface_border}; background: {self.palette.surface_background}; "
                f"box-shadow: {self.palette.surface_shadow};"
            ),
        )


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
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
                    ).render()
                )
            ),
        ]
    )
    return


@app.cell(column=3, hide_code=True)
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


@app.cell(column=4, hide_code=True)
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

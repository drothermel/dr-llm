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
    from pydantic import BaseModel, ConfigDict

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


@app.cell
def _():
    class PaletteToneName(StrEnum):
        NEUTRAL = "neutral"
        INFO = "info"
        SUCCESS = "success"
        WARNING = "warning"
        DANGER = "danger"


    class ColorPalette(BaseModel):
        model_config = ConfigDict(frozen=True)

        text_primary: str
        text_muted: str
        text_subtle: str
        surface_background: str
        surface_border: str
        surface_shadow: str
        chip_background: str
        chip_text: str
        neutral: TonePalette
        info: TonePalette
        success: TonePalette
        warning: TonePalette
        danger: TonePalette

        @classmethod
        def default(cls) -> "ColorPalette":
            return cls(
                text_primary="#0f172a",
                text_muted="#475569",
                text_subtle="#64748b",
                surface_background="linear-gradient(160deg, #f8fafc 0%, #eef4ff 62%, #fff7ed 100%)",
                surface_border="rgba(148, 163, 184, 0.18)",
                surface_shadow="0 10px 24px rgba(15, 23, 42, 0.07)",
                chip_background="#dbeafe",
                chip_text="#1d4ed8",
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

    return ColorPalette, PaletteToneName


@app.cell
def _(ColorPalette, PaletteToneName):
    class PoolCardTheme(BaseModel):
        model_config = ConfigDict(frozen=True)

        palette: ColorPalette
        status_complete: PaletteToneName = PaletteToneName.SUCCESS
        status_in_progress: PaletteToneName = PaletteToneName.WARNING
        status_default: PaletteToneName = PaletteToneName.NEUTRAL
        samples: PaletteToneName = PaletteToneName.SUCCESS
        in_flight: PaletteToneName = PaletteToneName.INFO
        pending: PaletteToneName = PaletteToneName.WARNING
        failed: PaletteToneName = PaletteToneName.DANGER

        @classmethod
        def default(cls) -> "PoolCardTheme":
            return cls(palette=ColorPalette.default())

        def tone(self, tone_name: PaletteToneName) -> TonePalette:
            return getattr(self.palette, tone_name.value)

        def status_tone(self, status: str) -> TonePalette:
            if status == "complete":
                return self.tone(self.status_complete)
            if status == "in_progress":
                return self.tone(self.status_in_progress)
            return self.tone(self.status_default)

    return (PoolCardTheme,)


@app.cell
def _(PoolCardTheme):
    class PoolCardRenderer(BaseModel):
        model_config = ConfigDict(frozen=True)

        theme: PoolCardTheme

        def format_datetime(self, value: datetime | None) -> str:
            if value is None:
                return "Not recorded"
            return value.strftime("%b %-d, %Y")

        def data_item(
            self,
            label: str,
            value: str,
            value_color: str | None = None,
        ) -> div:
            palette = self.theme.palette
            resolved_value_color = (
                palette.text_primary if value_color is None else value_color
            )
            return div(
                span(
                    label,
                    style=(
                        f"display: inline-block; min-width: 7rem; color: {palette.text_muted}; "
                        "font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; "
                        "font-weight: 700;"
                    ),
                ),
                span(
                    value,
                    style=(
                        f"color: {resolved_value_color}; font-size: 0.82rem; font-weight: 600;"
                    ),
                ),
                style="margin-top: 0.38rem;",
            )

        def axis_chip(self, name: str, column_type: str) -> span:
            _ = column_type
            palette = self.theme.palette
            return span(
                name,
                style=(
                    "display: inline-block; padding: 0.12rem 0.42rem; "
                    f"border-radius: 999px; background: {palette.chip_background}; "
                    f"color: {palette.chip_text}; "
                    "font-size: 0.68rem; font-weight: 600; margin-right: 0.28rem;"
                ),
            )

        def render(self, pool: PoolInspection) -> div:
            palette = self.theme.palette
            status_tone = self.theme.status_tone(pool.status.value)
            pending = pool.pending_counts
            key_columns = pool.pool_schema.key_columns

            return div(
                div(
                    div(
                        p(
                            pool.project_name,
                            style=(
                                f"margin: 0; color: {palette.text_subtle}; font-size: 0.7rem; "
                                "font-weight: 600; letter-spacing: 0.03em;"
                            ),
                        ),
                        p(
                            pool.name,
                            style=(
                                f"margin: 0.08rem 0 0; color: {palette.text_primary}; font-size: 1rem; "
                                "line-height: 1.15; font-weight: 800;"
                            ),
                        ),
                    ),
                    span(
                        pool.status.value.replace("_", " "),
                        style=(
                            f"display: inline-block; color: {status_tone.text}; background: {status_tone.bg}; "
                            f"border: 1px solid {status_tone.border}; border-radius: 999px; "
                            "padding: 0.2rem 0.45rem; font-size: 0.62rem; font-weight: 700; "
                            "text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap;"
                        ),
                    ),
                    style="display: flex; justify-content: space-between; gap: 0.7rem; align-items: start;",
                ),
                div(
                    span(
                        "Axes:",
                        style=(
                            f"color: {palette.text_muted}; font-size: 0.68rem; text-transform: uppercase; "
                            "letter-spacing: 0.06em; font-weight: 700; margin-right: 0.35rem;"
                        ),
                    ),
                    *[
                        self.axis_chip(column.name, column.type.value)
                        for column in key_columns
                    ],
                    style="margin-top: 0.55rem; line-height: 1.8;",
                ),
                div(
                    self.data_item(
                        "Created", self.format_datetime(pool.created_at)
                    ),
                    style="margin-top: 0.2rem;",
                ),
                div(
                    self.data_item(
                        "Samples",
                        f"{pool.sample_count:,}",
                        self.theme.tone(self.theme.samples).text,
                    ),
                    self.data_item(
                        "In flight",
                        str(pool.in_flight),
                        self.theme.tone(self.theme.in_flight).text,
                    ),
                    self.data_item(
                        "Pending",
                        str(pending.pending),
                        self.theme.tone(self.theme.pending).text,
                    ),
                    self.data_item(
                        "Failed",
                        str(pending.failed),
                        self.theme.tone(self.theme.failed).text,
                    ),
                    self.data_item("Promoted", str(pending.promoted)),
                    self.data_item("Leased", str(pending.leased)),
                    style=(
                        "margin-top: 0.55rem; padding-top: 0.28rem; "
                        f"border-top: 1px solid {palette.surface_border};"
                    ),
                ),
                style=(
                    "font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
                    f"color: {palette.text_primary}; width: 18rem; padding: 0.8rem 0.9rem; border-radius: 16px; "
                    f"border: 1px solid {palette.surface_border}; background: {palette.surface_background}; "
                    f"box-shadow: {palette.surface_shadow};"
                ),
            )

    return (PoolCardRenderer,)


@app.cell
def _(PoolCardRenderer, PoolCardTheme):
    pool_card_renderer = PoolCardRenderer(theme=PoolCardTheme.default())
    return (pool_card_renderer,)


@app.cell(column=1, hide_code=True)
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
    return (demo_pool_inspection,)


@app.cell(hide_code=True)
def _(demo_pool_inspection, pool_card_renderer):
    mo.vstack(
        [
            mo.md("## Pool Card Demo"),
            mo.Html(str(pool_card_renderer.render(demo_pool_inspection))),
        ]
    )
    return


@app.cell(hide_code=True)
def _(pool_card_renderer, pool_inspection):
    mo.vstack(
        [
            mo.md("## Pool Inspection"),
            mo.Html(str(pool_card_renderer.render(pool_inspection))),
        ]
    )
    return


@app.cell(column=2, hide_code=True)
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


@app.cell
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


@app.cell
def _(get_pool_info_form):
    # Get Pool Info Executor
    mo.stop(get_pool_info_form.value is None)
    pool_inspection = inspect_pool(
        PoolInspectionRequest(**get_pool_info_form.value)
    )
    return (pool_inspection,)


@app.cell(column=3, hide_code=True)
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


if __name__ == "__main__":
    app.run()

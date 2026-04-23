import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    from datetime import datetime

    import marimo as mo

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
    )
    from dr_llm.project.project_info import ProjectInfo
    from dr_llm.project.project_service import (
        assess_project_creation,
        create_project as create_project_service,
        inspect_projects,
    )
    from dr_llm.style import PiePoolCard, PoolCard, Style

    add_marimo_display()(ProjectInfo)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Goal:** Make a version of the pool card that has a pie chart with numeric labels instead of a list of numbers. The pie should use the disjoint card values as slices so the total is the sum of the listed numbers. That will be the first step towards visually representing the coverage of different providers/models, datasets, etc in a given pool. And then cards for seeing a given prompt or response or code snippet, etc.
    """)
    return


@app.cell(hide_code=True)
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
            "failed": 3,
        },
        status="in_progress",
    )
    mo.vstack(
        [
            mo.md("## Pool Card Demo"),
            PoolCard(
                pool=demo_pool_inspection,
                style=Style.default(),
                width="20rem",
            ).render(),
        ]
    )
    return (demo_pool_inspection,)


@app.cell
def _(demo_pool_inspection):
    mo.vstack(
        [
            mo.md("## Pie Pool Card Demo"),
            PiePoolCard(
                pool=demo_pool_inspection,
                style=Style.default(),
                width="20rem",
                height=None,
            ).render(),
        ]
    )
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
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
def _(get_pool_info_form):
    # Get Pool Info Executor
    mo.stop(get_pool_info_form.value is None)
    pool_inspection = inspect_pool(
        PoolInspectionRequest(**get_pool_info_form.value)
    )
    (
        PoolCard(
            pool=pool_inspection,
            style=Style.default(),
        ).render(),
    )
    return


@app.cell(column=2, hide_code=True)
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


if __name__ == "__main__":
    app.run()

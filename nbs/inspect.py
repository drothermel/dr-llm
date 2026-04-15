import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    from collections.abc import Sequence

    import marimo as mo

    from dr_llm.pool.admin_service import (
        assess_pool_creation,
        create_pool as create_pool_service,
        discover_pools as discover_pools_service,
        inspect_pool,
    )
    from dr_llm.pool.models import (
        CreatePoolRequest,
        PoolCreationViolation,
        PoolInspectionRequest,
    )
    from dr_llm.project.models import (
        CreateProjectRequest,
        ProjectCreationViolation,
    )
    from dr_llm.project.project_service import (
        assess_project_creation,
        create_project as create_project_service,
        inspect_projects,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **How this works**

    - The notebook is now a thin UI over typed library services.
    - `inspect_projects()` drives the visible project summary table.
    - `assess_project_creation(...)` returns a typed readiness result with
      structured violations; `create_project(...)` uses the same codepath.
    - `assess_pool_creation(...)` returns typed readiness for pool creation,
      including duplicate-name, in-progress, cooldown, and capacity checks.
    - `inspect_pool(...)` returns a typed pool inspection snapshot.
    - `create_pool(...)` creates the pool and returns a fresh inspection snapshot.
    - The project and pool creation UIs use marimo `form`s, so the notebook only
      receives values after you submit them.
    - If Docker or DB access fails, the exception bubbles up naturally.
    """)
    return


@app.function
def discover_pools(dsn: str) -> list[str]:
    return discover_pools_service(dsn)


@app.function
def get_pool_info(project_name: str, pool_name: str):
    return inspect_pool(
        PoolInspectionRequest(
            project_name=project_name,
            pool_name=pool_name,
        )
    )


@app.function
def format_blocked(
    violations: Sequence[PoolCreationViolation | ProjectCreationViolation],
) -> str:
    return "\n".join(violation.message for violation in violations)


@app.function
def check_pool_creation_guardrails(
    project_name: str,
    pool_name: str,
    key_axes: list[str],
    *,
    max_pools_per_project: int = 5,
    cooldown_seconds: int = 60,
):
    readiness = assess_pool_creation(
        CreatePoolRequest(
            project_name=project_name,
            pool_name=pool_name,
            key_axes=key_axes,
        ),
        max_pools_per_project=max_pools_per_project,
        cooldown_seconds=cooldown_seconds,
    )
    if not readiness.allowed:
        raise ValueError(format_blocked(readiness.violations))
    return readiness


@app.function
def build_project_display():
    summaries = inspect_projects()
    if not summaries:
        return None

    project_rows = [
        {
            "project": summary.project.name,
            "status": str(summary.project.status),
            "port": summary.project.port,
            "dsn": summary.project.dsn,
            "pool_count": summary.pool_count,
            "pools": ", ".join(summary.pool_names) if summary.pool_names else "(none)",
        }
        for summary in summaries
    ]

    running_count = sum(summary.project.status == "running" for summary in summaries)
    total_pools = sum(summary.pool_count for summary in summaries)
    return mo.vstack(
        [
            mo.md(
                f"**Projects:** {len(project_rows)}  \
**Running:** {running_count}  \
**Pools discovered:** {total_pools}"
            ),
            mo.ui.table(project_rows),
        ]
    )


@app.function
def create_pool(project_name: str, pool_name: str, key_axes: list[str]):
    inspection = create_pool_service(
        CreatePoolRequest(
            project_name=project_name,
            pool_name=pool_name,
            key_axes=key_axes,
        )
    )
    print(f"[created] project: {project_name}")
    print(f"[created] pool: {inspection.name}")
    print(f"[created] key axes: {inspection.pool_schema.key_column_names}")
    print("[created] tables:")
    print(f"  - {inspection.pool_schema.samples_table}")
    print(f"  - {inspection.pool_schema.claims_table}")
    print(f"  - {inspection.pool_schema.pending_table}")
    print(f"  - {inspection.pool_schema.metadata_table}")
    print(f"  - {inspection.pool_schema.call_stats_table}")
    return inspection


@app.function
def parse_create_pool_inputs(
    project_name: str,
    pool_name: str,
    axes_csv: str,
):
    request = CreatePoolRequest.from_csv(
        project_name=project_name,
        pool_name=pool_name,
        axes_csv=axes_csv,
    )
    return request.project_name, request.pool_name, request.key_axes


@app.function
def parse_create_project_inputs(project_name: str) -> str:
    return CreateProjectRequest(project_name=project_name).project_name


@app.function
def check_project_creation_guardrails(
    project_name: str,
    *,
    cooldown_seconds: int = 60,
):
    readiness = assess_project_creation(
        CreateProjectRequest(project_name=project_name),
        cooldown_seconds=cooldown_seconds,
    )
    if not readiness.allowed:
        raise ValueError(format_blocked(readiness.violations))
    return readiness


@app.function
def create_project(project_name: str):
    project = create_project_service(CreateProjectRequest(project_name=project_name))
    print(f"[created] project: {project.name}")
    print(f"[created] status: {project.status}")
    print(f"[created] port: {project.port}")
    print(f"[created] dsn: {project.dsn}")
    return project


@app.function
def parse_get_pool_info_inputs(project_name: str, pool_name: str):
    request = PoolInspectionRequest(
        project_name=project_name,
        pool_name=pool_name,
    )
    return request.project_name, request.pool_name


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Pool inspection

    Manual notebook view of dr-llm Docker projects and the pool names currently
    visible inside each running project database.
    """)
    return


@app.cell(hide_code=True)
def _(create_pool_form, create_project_form):
    _ = (create_project_form.value, create_pool_form.value)
    build_project_display()
    return


@app.cell(hide_code=True)
def _(create_project_form):
    mo.stop(create_project_form.value is None)
    project_name = parse_create_project_inputs(**create_project_form.value)
    check_project_creation_guardrails(project_name, cooldown_seconds=60)
    create_project(project_name)
    return


@app.cell(hide_code=True)
def _(create_pool_form):
    mo.stop(create_pool_form.value is None)
    pool_create_project_name, pool_create_name, pool_create_axes = parse_create_pool_inputs(
        **create_pool_form.value
    )
    check_pool_creation_guardrails(
        pool_create_project_name,
        pool_create_name,
        pool_create_axes,
        max_pools_per_project=5,
    )
    create_pool(pool_create_project_name, pool_create_name, pool_create_axes)
    get_pool_info(pool_create_project_name, pool_create_name)
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
def _(get_pool_info_form):
    mo.stop(get_pool_info_form.value is None)
    pool_info_project, pool_info_name = parse_get_pool_info_inputs(
        **get_pool_info_form.value
    )
    get_pool_info(pool_info_project, pool_info_name)
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import logging

    import marimo as mo
    from dr_llm.pool.admin_service import (
        assess_pool_creation,
        create_pool as create_pool_service,
        inspect_pool,
    )
    from dr_llm.pool.models import CreatePoolRequest, PoolInspectionRequest
    from dr_llm.project.models import CreateProjectRequest
    from dr_llm.project.project_service import (
        assess_project_creation,
        create_project as create_project_service,
        inspect_projects,
    )
    from dr_llm.ui import PoolSimpleStatsPieCard, bootstrap_tailwind, wrap_cards

    IGNORE_DEMO_PROJECTS = True
    logger = logging.getLogger(__name__)


@app.cell
def _():
    bootstrap_tailwind()
    return


@app.cell(column=1, hide_code=True)
def _(create_pool_form, create_project_form):
    _ = (create_project_form.value, create_pool_form.value)
    project_summaries = inspect_projects()
    return (project_summaries,)


@app.cell(hide_code=True)
def _(project_summaries):
    mo.vstack(
        [
            mo.md("## Docker Projects State"),
            mo.ui.table([summary.to_row() for summary in project_summaries]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(project_summaries):
    project_sections = {}
    ignore_demo_banner = (
        mo.md("*Ignoring projects with `demo` in the name*")
        if IGNORE_DEMO_PROJECTS
        else None
    )

    for summary in project_summaries:
        if summary.pool_inspection.status.value != "discovered":
            continue
        if not summary.pool_inspection.pool_names:
            continue
        normalized_project_name = summary.project.name.strip().lower()
        if IGNORE_DEMO_PROJECTS and "demo" in normalized_project_name:
            continue

        cards = []
        failed_pool_names: list[tuple[str, str]] = []
        for pool_name in summary.pool_inspection.pool_names:
            try:
                pool_inspection = inspect_pool(
                    PoolInspectionRequest(
                        project_name=summary.project.name,
                        pool_name=pool_name,
                    )
                )
            except Exception as exc:
                logger.exception(
                    "Failed to inspect pool %s for project %s",
                    pool_name,
                    summary.project.name,
                )
                failed_pool_names.append((pool_name, "inspection failed"))
                continue

            cards.append(
                PoolSimpleStatsPieCard(
                    pool=pool_inspection,
                    width="20rem",
                )
            )

        if not cards and not failed_pool_names:
            continue

        section_items = [mo.md(f"### {summary.project.name}")]
        if cards:
            section_items.append(wrap_cards(cards))
        if failed_pool_names:
            section_items.append(
                mo.md(
                    "Could not inspect on load: "
                    + ", ".join(
                        f"`{pool_name}` ({error})"
                        for pool_name, error in failed_pool_names
                    )
                )
            )

        project_sections[summary.project.name] = mo.vstack(section_items, gap=1)

    section_items = [mo.md("## Running Project Pools")]
    if ignore_demo_banner is not None:
        section_items.append(ignore_demo_banner)
    section = (
        mo.vstack(
            [*section_items, mo.accordion(project_sections)],
            gap=1.5,
        )
        if project_sections
        else mo.vstack(
            [
                *section_items,
                mo.md("No running projects with discovered pools."),
            ],
            gap=1,
        )
    )

    section
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

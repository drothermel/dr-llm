import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from marimo_utils import add_marimo_display

    from dr_llm.pool.admin_service import inspect_pool
    from dr_llm.pool.models import PoolInspection, PoolInspectionRequest

    add_marimo_display()(PoolInspection)


@app.cell
def _():
    PROJECT_NAME = "code_comp_v0"
    POOL_NAME = "encoder_pool_t1"
    return POOL_NAME, PROJECT_NAME


@app.cell(column=1, hide_code=True)
def _(POOL_NAME, PROJECT_NAME):
    pool_status = inspect_pool(
        PoolInspectionRequest(
            project_name=PROJECT_NAME,
            pool_name=POOL_NAME,
        )
    )
    pool_status
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()

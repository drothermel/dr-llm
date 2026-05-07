import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from dr_llm.pool.admin_service import inspect_pool
    from dr_llm.pool.models import PoolInspectionRequest


@app.cell
def _():
    provider = "code_comp_v0"
    pool = "encoder_pool_t1"

    pool_status = inspect_pool(
        PoolInspectionRequest(
            project_name=provider,
            pool_name=pool,
        )
    )
    pool_status_df = pool_status.to_df()
    return (pool_status_df,)


@app.cell(column=1)
def _(pool_status_df):
    pool_status_df
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()

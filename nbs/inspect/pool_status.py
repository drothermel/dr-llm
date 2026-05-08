import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from dr_llm.pool.reader import PoolReader


@app.cell
def _():
    project_name = "code_comp_v0"
    pool_name = "encoder_pool_t1"
    return pool_name, project_name


@app.cell(hide_code=True)
def _(pool_name, project_name):
    with PoolReader.open(project_name, pool_name) as reader:
        _pool_status = reader.inspect()
        _pool_status_df = _pool_status.to_df()
        pool_data_df = reader.pool_data_df()

    mo.vstack(
        [
            mo.md(f"###`[{project_name}]` **Pool:** `{pool_name}`"),
            mo.accordion(
                {
                    "Pool Status": _pool_status_df.iloc[0]
                    .rename_axis("field")
                    .rename("value")
                }
            ),
        ]
    )
    return (pool_data_df,)


@app.cell(column=1, hide_code=True)
def _(pool_data_df, pool_name):
    mo.vstack(
        [
            mo.md(f"### `{pool_name}` Pool Data"),
            pool_data_df,
        ]
    )
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()

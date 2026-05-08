import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from dr_llm.pool.reader import PoolReader


@app.cell
def _():
    project_widget = mo.ui.text(
        label="Project",
        value="code_comp_v0",
    )
    pool_widget = mo.ui.text(
        label="Pool",
        value="encoder_pool_t1",
    )

    mo.vstack([project_widget, pool_widget])
    return pool_widget, project_widget


@app.cell(hide_code=True)
def _(pool_widget, project_widget):
    _project_name = project_widget.value
    _pool_name = pool_widget.value
    with PoolReader.open(_project_name, _pool_name) as reader:
        _pool_status = reader.inspect()
        _pool_status_df = _pool_status.to_df()
        pool_data_df = reader.pool_data_df()

    mo.vstack(
        [
            mo.md(f"###`[{_project_name}]` **Pool:** `{_pool_name}`"),
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


@app.cell(column=1)
def _():
    keep_cols = [
        # ids
        "prompt_template_id",
        "data_sample_id",
        "llm_config_id",
        "sample_idx",
        # stats
        "latency_ms",
        "total_cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "outcome",
        # x, y', y
        "data_sample__source_code",
        "prompt_text",
        "result_text",
    ]
    return (keep_cols,)


@app.cell(hide_code=True)
def _(keep_cols, pool_data_df, pool_widget):
    mo.vstack(
        [
            mo.md(f"### `{pool_widget.value}` Pool Data"),
            pool_data_df[keep_cols],
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

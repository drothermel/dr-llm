import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import re
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    NOTEBOOK_PATH = Path(__file__).resolve()
    REPO_ROOT = NOTEBOOK_PATH.parents[2]
    DEFAULT_PER_TASK_DIR = Path(
        "/Users/daniellerothermel/drotherm/data/code-comp/"
        "dr-llm-humaneval-pool-dumps/20260621_manual/per_elem"
    )
    TASK_ID_COLUMN = "human_eval_task_id"
    CODE_COLUMN = "raw_code_output"
    LOAD_COLUMNS = [
        "attempt_id",
        "project_name",
        "pool_name",
        "pool_sample_id",
        "sample_idx",
        "run_id",
        "created_at",
        TASK_ID_COLUMN,
        "data_sample_id",
        "source_sample_id",
        "prompt_template_id",
        "dec_prompt_template_id",
        "llm_config_id",
        "dec_llm_config_id",
        "model",
        "provider",
        "finish_reason",
        "attempt_count",
        CODE_COLUMN,
        "decoder_input_description",
        "decoder_input_description_source",
    ]


@app.cell
def _():
    TASK_FILE_RE = r"^human_eval-(\d+)-decode\.parquet$"
    return (TASK_FILE_RE,)


@app.cell
def _(TASK_FILE_RE):
    def task_index(path: Path) -> int:
        match = re.fullmatch(TASK_FILE_RE, path.name)
        if match is None:
            return 10_000
        return int(match.group(1))

    return (task_index,)


@app.cell
def _(task_index):
    def task_files(directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            directory.glob("human_eval-*-decode.parquet"), key=task_index
        )

    return (task_files,)


@app.function
def option_label(path: Path) -> str:
    return path.stem.replace("human_eval-", "HumanEval/").replace(
        "-decode", ""
    )


@app.function
def code_line_count(value: str) -> int:
    if not value:
        return 0
    return value.count("\n") + 1


@app.function
def load_decoder_generations(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=LOAD_COLUMNS)
    generations = frame.loc[:, LOAD_COLUMNS].copy()
    generations["generation_char_count"] = generations[
        CODE_COLUMN
    ].str.len()
    generations["generation_line_count"] = generations[CODE_COLUMN].map(
        code_line_count
    )
    return generations


@app.function
def summarize_exact_repeats(generations: pd.DataFrame) -> pd.DataFrame:
    summary = (
        generations.groupby(CODE_COLUMN, sort=False, dropna=False)
        .agg(
            repeat_count=("attempt_id", "size"),
            human_eval_task_id=(TASK_ID_COLUMN, "first"),
            generation_char_count=("generation_char_count", "first"),
            generation_line_count=("generation_line_count", "first"),
            first_project_name=("project_name", "first"),
            first_pool_name=("pool_name", "first"),
            first_model=("model", "first"),
        )
        .reset_index()
    )
    repeated = summary.loc[summary["repeat_count"] > 1].copy()
    repeated = repeated.sort_values(
        ["repeat_count", "generation_char_count"],
        ascending=[False, True],
    )
    repeated.insert(0, "rank", range(1, len(repeated) + 1))
    return repeated


@app.function
def repeat_metrics(
    generations: pd.DataFrame, repeats: pd.DataFrame
) -> pd.DataFrame:
    total_rows = len(generations)
    unique_outputs = generations[CODE_COLUMN].nunique(dropna=False)
    repeated_output_count = len(repeats)
    rows_in_repeats = int(repeats["repeat_count"].sum())
    return pd.DataFrame(
        [
            {
                "rows": total_rows,
                "unique_outputs": unique_outputs,
                "repeated_outputs": repeated_output_count,
                "rows_in_repeated_outputs": rows_in_repeats,
                "exact_duplicate_extra_rows": total_rows - unique_outputs,
            }
        ]
    )


@app.function
def repeat_histogram(repeats: pd.DataFrame) -> alt.Chart:
    top_repeats = repeats.head(100).copy()
    top_repeats["output_rank"] = top_repeats["rank"].astype(str)
    return (
        alt.Chart(top_repeats)
        .mark_bar()
        .encode(
            x=alt.X(
                "output_rank:N",
                title="Exact output rank",
                sort=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=True),
            ),
            y=alt.Y("repeat_count:Q", title="Repeat count"),
            tooltip=[
                "rank:Q",
                "repeat_count:Q",
                "generation_char_count:Q",
                "generation_line_count:Q",
                "first_pool_name:N",
                "first_model:N",
            ],
        )
        .properties(height=260)
    )


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## HumanEval Exact Dedupe
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Load one per-task Parquet file and inspect exact duplicate decoder
    generations within that HumanEval task.
    """)
    return


@app.cell(hide_code=True)
def _(task_files):
    per_task_paths = task_files(DEFAULT_PER_TASK_DIR)
    per_task_options = {option_label(path): path for path in per_task_paths}
    default_task = next(iter(per_task_options), None)
    task_dropdown = mo.ui.dropdown(
        options=per_task_options,
        value=default_task,
        allow_select_none=not per_task_options,
        searchable=True,
        label="HumanEval task",
        full_width=True,
    )
    load_button = mo.ui.run_button(
        label="Load task",
        kind="success",
        full_width=True,
        disabled=not per_task_options,
    )
    artifact_status = mo.md("")
    if not per_task_options:
        artifact_status = mo.callout(
            f"No per-task Parquet files found under `{DEFAULT_PER_TASK_DIR}`.",
            kind="warn",
        )
    controls = mo.vstack([artifact_status, task_dropdown, load_button], gap=1)
    controls
    return load_button, task_dropdown


@app.cell(hide_code=True)
def _(load_button, task_dropdown):
    selected_path = task_dropdown.value
    decoder_generations = pd.DataFrame()
    load_status = mo.callout(
        "Select a task and click Load task.", kind="neutral"
    )
    if load_button.value:
        if selected_path is None:
            load_status = mo.callout(
                "No per-task Parquet file is selected.", kind="warn"
            )
        else:
            decoder_generations = load_decoder_generations(selected_path)
            load_status = mo.callout(
                f"Loaded `{selected_path.name}` with "
                f"{len(decoder_generations):,} decoder generations.",
                kind="success",
            )
    load_status
    return (decoder_generations,)


@app.cell(hide_code=True)
def _(decoder_generations):
    repeat_summary = pd.DataFrame()
    metrics = pd.DataFrame()
    if not decoder_generations.empty:
        repeat_summary = summarize_exact_repeats(decoder_generations)
        metrics = repeat_metrics(decoder_generations, repeat_summary)
    metrics
    return (repeat_summary,)


@app.cell(hide_code=True)
def _(repeat_summary):
    chart_output = mo.callout(
        "No exact repeats found for the loaded task.", kind="neutral"
    )
    if not repeat_summary.empty:
        chart_output = mo.vstack(
            [
                mo.md("### Top repeated exact outputs"),
                mo.ui.altair_chart(repeat_histogram(repeat_summary)),
            ],
            gap=1,
        )
    chart_output
    return


@app.cell(hide_code=True)
def _(repeat_summary):
    repeat_summary
    return


@app.cell(hide_code=True)
def _(decoder_generations):
    decoder_generations
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    from contextlib import contextmanager
    import json
    import math
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from typing import Any

    import marimo as mo
    import pandas as pd
    from marimo_utils.ui import (
        BarChart,
        BarItem,
        BoxPlotCard,
        Card,
        ChartColor,
        DataItem,
        HeatmapChart,
        HistogramCard,
        PieChart,
        PieSlice,
        QuantileFences,
        FrequencyBarCard,
        ViolinPlotCard,
        compute_gini,
        skew_label,
    )
    from dr_llm.pool import (
        DbConfig,
        DbRuntime,
        PoolInspectionRequest,
        PoolReader,
        inspect_pool,
        sample_response_stats,
    )
    from dr_llm.project import (
        inspect_projects,
        maybe_get_project,
    )
    from dr_llm.ui import PoolSimpleStatsPieCard, bootstrap_tailwind

    TARGET_PROJECT_NAMES = (
        "nl_latents",
        "code_comp_v0",
    )


@app.cell
def _():
    bootstrap_tailwind()
    return


@app.cell
def _():
    project_summaries = inspect_projects()
    return (project_summaries,)


@app.cell(hide_code=True)
def _():
    def compact_json(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(
            value,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        )

    def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=list(columns))

    def sort_frame(
        frame: pd.DataFrame,
        *,
        by: Sequence[str],
        ascending: Sequence[bool] | bool,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame.sort_values(
            list(by),
            ascending=ascending,
            kind="stable",
        ).reset_index(drop=True)

    @contextmanager
    def pool_reader_context(
        *,
        project_name: str,
        pool_name: str,
    ):
        project = maybe_get_project(project_name)
        if project is None or project.dsn is None:
            raise ValueError(f"Project {project_name!r} is not available")

        runtime = DbRuntime(
            DbConfig(
                dsn=project.dsn,
                application_name="pool_inspect_notebook",
            )
        )
        try:
            reader = PoolReader.open(pool_name, runtime=runtime)
            schema = reader.schema
            key_columns = list(schema.key_column_names)
            yield runtime, schema, reader, key_columns
        finally:
            runtime.close()

    def load_pool_inspection(
        *,
        project_name: str,
        pool_name: str,
    ) -> Any:
        return inspect_pool(
            PoolInspectionRequest(
                project_name=project_name,
                pool_name=pool_name,
            )
        )

    def load_sample_frame(
        *,
        reader: PoolReader,
        key_columns: Sequence[str],
    ) -> pd.DataFrame:
        samples = reader.samples_list()
        sample_rows = [
            {
                "sample_id": sample.sample_id,
                **{
                    column_name: sample.key_values.get(column_name)
                    for column_name in key_columns
                },
                "sample_idx": sample.sample_idx,
                "run_id": sample.run_id,
                "is_complete": sample.is_complete,
                "finish_reason": sample.finish_reason,
                "attempt_count": sample.attempt_count,
                "created_at": sample.created_at,
                "request_json": compact_json(sample.request),
                "metadata_json": compact_json(sample.metadata),
            }
            for sample in samples
        ]
        sample_frame = pd.DataFrame.from_records(sample_rows)
        sample_columns = [
            "sample_id",
            *key_columns,
            "sample_idx",
            "run_id",
            "is_complete",
            "finish_reason",
            "attempt_count",
            "created_at",
            "request_json",
            "metadata_json",
        ]
        if sample_frame.empty:
            return empty_frame(sample_columns)

        sample_frame = sample_frame.loc[:, sample_columns]
        return sort_frame(
            sample_frame,
            by=[*key_columns, "sample_idx", "created_at"],
            ascending=[*([True] * len(key_columns)), True, True],
        )

    def load_response_stats_frame(
        *,
        reader: PoolReader,
        key_columns: Sequence[str],
    ) -> pd.DataFrame:
        samples = reader.samples_list(completion="complete")
        rows = []
        for sample in samples:
            stats = sample_response_stats(sample)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    **{col: sample.key_values.get(col) for col in key_columns},
                    "sample_idx": sample.sample_idx,
                    "latency_ms": stats.latency_ms,
                    "total_cost_usd": stats.total_cost_usd,
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "reasoning_tokens": stats.reasoning_tokens,
                    "total_tokens": stats.total_tokens,
                    "attempt_count": stats.attempt_count,
                    "finish_reason": stats.finish_reason,
                    "created_at": sample.created_at,
                }
            )
        stats_columns = [
            "sample_id",
            *key_columns,
            "sample_idx",
            "latency_ms",
            "total_cost_usd",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "attempt_count",
            "finish_reason",
            "created_at",
        ]
        frame = pd.DataFrame.from_records(rows)
        if frame.empty:
            return empty_frame(stats_columns)
        return frame.loc[:, stats_columns]

    def load_error_frame(
        *,
        reader: PoolReader,
        key_columns: Sequence[str],
    ) -> pd.DataFrame:
        samples = reader.samples_list(completion="error")
        rows = [
            {
                "sample_id": sample.sample_id,
                **{col: sample.key_values.get(col) for col in key_columns},
                "sample_idx": sample.sample_idx,
                "attempt_count": sample.attempt_count,
                "finish_reason": sample.finish_reason,
                "created_at": sample.created_at,
                "metadata_json": compact_json(sample.metadata),
            }
            for sample in samples
        ]
        error_columns = [
            "sample_id",
            *key_columns,
            "sample_idx",
            "attempt_count",
            "finish_reason",
            "created_at",
            "metadata_json",
        ]
        frame = pd.DataFrame.from_records(rows)
        if frame.empty:
            return empty_frame(error_columns)
        return sort_frame(
            frame.loc[:, error_columns],
            by=["created_at", *key_columns, "sample_idx"],
            ascending=[False, *([True] * len(key_columns)), True],
        )

    CacheGetter = Callable[[], dict[str, Any]]
    CacheSetter = Callable[[Callable[[dict[str, Any]], dict[str, Any]]], None]

    def _cache_put(set_cache: CacheSetter, **updates: Any) -> None:
        set_cache(lambda c: {**c, **updates})

    def ensure_pool_inspection(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> Any:
        cache = get_cache()
        if "pool_inspection" in cache:
            return cache["pool_inspection"]
        inspection = load_pool_inspection(
            project_name=project_name,
            pool_name=pool_name,
        )
        _cache_put(set_cache, pool_inspection=inspection)
        return inspection

    def ensure_sample_frame(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> tuple[pd.DataFrame, list[str]]:
        cache = get_cache()
        if "sample_frame" in cache and "key_columns" in cache:
            return cache["sample_frame"], cache["key_columns"]
        with pool_reader_context(
            project_name=project_name,
            pool_name=pool_name,
        ) as (_, _, reader, key_columns):
            sample_frame = load_sample_frame(
                reader=reader,
                key_columns=key_columns,
            )
            kc = list(key_columns)
        _cache_put(set_cache, sample_frame=sample_frame, key_columns=kc)
        return sample_frame, kc

    def ensure_response_stats_frame(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> tuple[pd.DataFrame, list[str]]:
        cache = get_cache()
        if "response_stats_frame" in cache and "key_columns" in cache:
            return cache["response_stats_frame"], cache["key_columns"]
        with pool_reader_context(
            project_name=project_name,
            pool_name=pool_name,
        ) as (_, _, reader, key_columns):
            frame = load_response_stats_frame(
                reader=reader,
                key_columns=key_columns,
            )
            kc = list(key_columns)
        _cache_put(set_cache, response_stats_frame=frame, key_columns=kc)
        return frame, kc

    def ensure_error_frame(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> tuple[pd.DataFrame, list[str]]:
        cache = get_cache()
        if "error_frame" in cache and "key_columns" in cache:
            return cache["error_frame"], cache["key_columns"]
        with pool_reader_context(
            project_name=project_name,
            pool_name=pool_name,
        ) as (_, _, reader, key_columns):
            frame = load_error_frame(
                reader=reader,
                key_columns=key_columns,
            )
            kc = list(key_columns)
        _cache_put(set_cache, error_frame=frame, key_columns=kc)
        return frame, kc

    return (
        ensure_error_frame,
        ensure_pool_inspection,
        ensure_response_stats_frame,
        ensure_sample_frame,
    )


@app.function(hide_code=True)
def render_section(
    title: str,
    run_button: object,
    body: object | None = None,
) -> mo.Html:
    items: list[object] = [mo.md(f"### {title}"), run_button]
    if body is None:
        items.append(mo.md("Press `Run` to load this section."))
    else:
        items.append(body)
    return mo.vstack(items)


@app.function(hide_code=True)
def fmt_int(n: Any) -> str:
    if n is None:
        return "—"
    if isinstance(n, float) and math.isnan(n):
        return "—"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


@app.function(hide_code=True)
def fmt_float(x: Any, fmt: str = ".2f") -> str:
    if x is None:
        return "—"
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(value):
        return "—"
    return format(value, fmt)


@app.function(hide_code=True)
def fmt_cost(x: Any) -> str:
    if x is None:
        return "—"
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(value):
        return "—"
    return f"${value:,.4f}"


@app.function(hide_code=True)
def fmt_ms(x: Any) -> str:
    if x is None:
        return "—"
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(value):
        return "—"
    return f"{value:,.0f} ms"


@app.function(hide_code=True)
def fmt_ts(ts: Any) -> str:
    if ts is None:
        return "—"
    if isinstance(ts, pd.Timestamp):
        if pd.isna(ts):
            return "—"
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime):
        return ts.astimezone().strftime("%Y-%m-%d %H:%M")
    return str(ts)


@app.function(hide_code=True)
def truncate(s: str, n: int = 60) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


@app.function(hide_code=True)
def value_counts(series: pd.Series) -> list[tuple[str, int]]:
    if series is None or len(series) == 0:
        return []
    cleaned = series.dropna()
    if cleaned.empty:
        return []
    vc = cleaned.value_counts()
    return [(str(label), int(count)) for label, count in vc.items()]


@app.function(hide_code=True)
def ts_to_unix(frame: pd.DataFrame, column: str) -> list[float]:
    if frame is None or frame.empty or column not in frame.columns:
        return []
    series = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
    return [float(ts.timestamp()) for ts in series]


@app.function(hide_code=True)
def stat_card(
    title: str,
    items: Sequence[tuple[str, str]],
    *,
    description: str | None = None,
    width: str = "w-72",
) -> Card:
    rendered_items = [DataItem(label=k, value=v).render() for k, v in items]
    return Card(
        title=title,
        description=description,
        content=mo.vstack(rendered_items, gap=0.25),
        width=width,
    )


@app.cell(column=1, hide_code=True)
def _(project_summaries):
    def pool_rows_from_summaries(
        summaries: Sequence[Any],
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for summary in summaries:
            if summary.pool_inspection.status.value != "discovered":
                continue
            if not summary.pool_inspection.pool_names:
                continue
            for pool_name in summary.pool_inspection.pool_names:
                pool_key = json.dumps(
                    {
                        "project_name": summary.project.name,
                        "pool_name": pool_name,
                    },
                    sort_keys=True,
                )
                rows.append(
                    {
                        "project_name": summary.project.name,
                        "pool_name": pool_name,
                        "pool_key": pool_key,
                        "pool_label": f"{summary.project.name} / {pool_name}",
                    }
                )
        return rows

    target_project_names = {
        project_name.strip().lower() for project_name in TARGET_PROJECT_NAMES
    }
    pool_rows = [
        row
        for row in pool_rows_from_summaries(project_summaries)
        if row["project_name"].strip().lower() in target_project_names
    ]
    return (pool_rows,)


@app.cell(hide_code=True)
def _(pool_selector):
    selected_pool = None
    if pool_selector is not None and pool_selector.value is not None:
        selected_pool = json.loads(pool_selector.value)
    return (selected_pool,)


@app.cell(hide_code=True)
def _(pool_rows):
    pool_selector = None
    target_projects = ", ".join(f"`{name}`" for name in TARGET_PROJECT_NAMES)
    if not pool_rows:
        output = mo.vstack(
            [
                mo.md("## Detailed Pool Inspection"),
                mo.md(f"Configured projects: {target_projects}"),
                mo.md("No discovered pools are available for the configured projects."),
            ],
            gap=0.75,
        )
    else:
        pool_options = {row["pool_label"]: row["pool_key"] for row in pool_rows}
        default_label = pool_rows[0]["pool_label"]
        pool_selector = mo.ui.dropdown(
            options=pool_options,
            value=default_label,
            label="Selected pool",
            searchable=True,
            full_width=True,
        )
        output = mo.vstack(
            [
                mo.md("## Detailed Pool Inspection"),
                mo.md(f"Configured projects: {target_projects}"),
                mo.md("One dataframe per question for the currently selected pool."),
                pool_selector,
            ],
            gap=0.75,
        )
    output
    return (pool_selector,)


@app.cell(hide_code=True)
def _():
    get_section_results, set_section_results = mo.state({})
    return get_section_results, set_section_results


@app.cell(hide_code=True)
def _():
    get_raw_frames, set_raw_frames = mo.state({})
    return get_raw_frames, set_raw_frames


@app.cell(hide_code=True)
def _(selected_pool, set_raw_frames, set_section_results):
    _ = selected_pool
    set_section_results({})
    set_raw_frames({})
    return


@app.cell(hide_code=True)
def _():
    health_run_button = mo.ui.run_button(label="Run", kind="success")
    coverage_run_button = mo.ui.run_button(label="Run", kind="success")
    sample_run_button = mo.ui.run_button(label="Run", kind="success")
    error_run_button = mo.ui.run_button(label="Run", kind="success")
    response_stats_run_button = mo.ui.run_button(label="Run", kind="success")
    trend_run_button = mo.ui.run_button(label="Run", kind="success")
    return (
        coverage_run_button,
        error_run_button,
        health_run_button,
        response_stats_run_button,
        sample_run_button,
        trend_run_button,
    )


@app.cell(hide_code=True)
def _(selected_pool):
    mo.stop(selected_pool is None)
    mo.md(f"""
    **Selected:** `{selected_pool["project_name"]} / {selected_pool["pool_name"]}`
    """)
    return


@app.cell(hide_code=True)
def _(
    ensure_pool_inspection,
    ensure_response_stats_frame,
    get_raw_frames,
    get_section_results,
    health_run_button,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def scaled_series(frame: pd.DataFrame, column: str, scale: float) -> pd.Series:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if scale != 1.0:
            values = values * scale
        return values

    def range_line(series: pd.Series, template: str) -> str:
        return template.format(lo=float(series.min()), hi=float(series.max()))

    def build_health_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        pool_inspection = ensure_pool_inspection(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        response_stats_frame, key_columns = ensure_response_stats_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "pool_label": f"{project_name} / {pool_name}",
            "pool_inspection": pool_inspection,
            "key_columns": key_columns,
            "response_stats_frame": response_stats_frame,
        }

    def make_pool_health_pie_card(inspection: Any) -> PoolSimpleStatsPieCard:
        return PoolSimpleStatsPieCard(
            pool=inspection,
            width="20rem",
        )

    def make_pool_finish_reason_card(
        response_stats_frame: pd.DataFrame,
    ) -> Card | None:
        if "finish_reason" not in response_stats_frame.columns:
            return None
        finish_counts = value_counts(
            response_stats_frame["finish_reason"].astype("string").str.lower()
        )
        return Card(
            title="Finish reasons",
            description="Distribution of finish_reason",
            content=PieChart(
                slices=[PieSlice(label=lab, value=cnt) for lab, cnt in finish_counts],
                height=220,
                show_legend=True,
            ),
            width="w-80",
        )

    def make_pool_cost_distribution_plot(
        response_stats_frame: pd.DataFrame,
        column: str = "total_cost_usd",
    ) -> BoxPlotCard | None:
        if column not in response_stats_frame.columns:
            return None
        cost_series = scaled_series(response_stats_frame, column, 1.0)
        cost_range = range_line(cost_series, "**True Range:** ${lo:,.4f} - ${hi:,.4f}")
        return BoxPlotCard(
            column=column,
            data=response_stats_frame,
            label="Cost",
            title="Cost distribution",
            description=f"p1 · q1 · median · q3 · p99 ($)\n{cost_range}",
            tick_format="$.4f",
            y_label="Cost (USD)",
        )

    def make_pool_cost_shape_plot(
        response_stats_frame: pd.DataFrame,
        column: str = "total_cost_usd",
    ) -> ViolinPlotCard | None:
        if column not in response_stats_frame.columns:
            return None
        cost_series = scaled_series(response_stats_frame, column, 1.0)
        cost_range = range_line(cost_series, "**True Range:** ${lo:,.4f} - ${hi:,.4f}")
        shape_primary = "KDE on p1-p99 bulk; <=2k sampled points"
        return ViolinPlotCard(
            column=column,
            data=response_stats_frame,
            label="Cost",
            title="Cost shape",
            description=f"{shape_primary}\n{cost_range}",
            clip_fences=QuantileFences.P1_P99,
            tick_format="$.4f",
            y_label="Cost (USD)",
        )

    def make_pool_latency_distribution_plot(
        response_stats_frame: pd.DataFrame,
        column: str = "latency_ms",
    ) -> BoxPlotCard | None:
        if column not in response_stats_frame.columns:
            return None
        latency_scale = 0.001
        latency_series = scaled_series(response_stats_frame, column, latency_scale)
        latency_range = range_line(
            latency_series, "**True Range:** {lo:,.2f} - {hi:,.2f}s"
        )
        return BoxPlotCard(
            column=column,
            data=response_stats_frame,
            label="Latency",
            title="Latency distribution",
            description=f"p1 · q1 · median · q3 · p99 (s)\n{latency_range}",
            value_scale=latency_scale,
            tick_format=".2f",
            y_label="Latency (s)",
        )

    def make_pool_latency_shape_plot(
        response_stats_frame: pd.DataFrame,
        column: str = "latency_ms",
    ) -> ViolinPlotCard | None:
        if column not in response_stats_frame.columns:
            return None
        latency_scale = 0.001
        latency_series = scaled_series(response_stats_frame, column, latency_scale)
        latency_range = range_line(
            latency_series, "**True Range:** {lo:,.2f} - {hi:,.2f}s"
        )
        shape_primary = "KDE on p1-p99 bulk; <=2k sampled points"
        return ViolinPlotCard(
            column=column,
            data=response_stats_frame,
            label="Latency",
            title="Latency shape",
            description=f"{shape_primary}\n{latency_range}",
            clip_fences=QuantileFences.P1_P99,
            value_scale=latency_scale,
            tick_format=".2f",
            y_label="Latency (s)",
        )

    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    health_data = _section_results.get("health")
    if health_run_button.value:
        health_data = build_health_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "health": health_data})

    mo.stop(
        health_data is None,
        render_section("Pool health", health_run_button),
    )

    inspection = health_data["pool_inspection"]
    response_stats_frame: pd.DataFrame = health_data["response_stats_frame"]

    pool_pie = make_pool_health_pie_card(inspection)
    finish_reasons = make_pool_finish_reason_card(response_stats_frame)
    cost_distribution = make_pool_cost_distribution_plot(response_stats_frame)
    cost_shape = make_pool_cost_shape_plot(response_stats_frame)
    latency_distribution = make_pool_latency_distribution_plot(response_stats_frame)
    latency_shape = make_pool_latency_shape_plot(response_stats_frame)

    _body = mo.hstack(
        [
            mo.vstack(
                [
                    pool_pie.render(),
                    finish_reasons.render(),
                ],
            ),
            mo.vstack(
                [
                    mo.hstack(
                        [
                            cost_distribution.render(),
                            cost_shape.render(),
                        ],
                    ),
                    mo.hstack(
                        [
                            latency_distribution.render(),
                            latency_shape.render(),
                        ],
                    ),
                ],
            ),
        ],
        align="start",
    )

    render_section("Pool health", health_run_button, _body)
    return


@app.cell(hide_code=True)
def _(
    coverage_run_button,
    ensure_sample_frame,
    get_raw_frames,
    get_section_results,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def build_coverage_frame(
        *,
        sample_frame: pd.DataFrame,
        key_columns: Sequence[str],
    ) -> pd.DataFrame:
        if sample_frame.empty:
            return pd.DataFrame(columns=[*key_columns, "count"])

        coverage_frame = (
            sample_frame.loc[:, key_columns]
            .value_counts(dropna=False)
            .rename("count")
            .reset_index()
        )
        return coverage_frame.sort_values(
            ["count", *key_columns],
            ascending=[False, *([True] * len(key_columns))],
            kind="stable",
        ).reset_index(drop=True)

    def build_coverage_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        sample_frame, key_columns = ensure_sample_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "coverage_frame": build_coverage_frame(
                sample_frame=sample_frame,
                key_columns=key_columns,
            ),
        }

    def per_key_count_series(
        coverage_frame: pd.DataFrame,
        key_column: str,
    ) -> pd.Series:
        return (
            coverage_frame.groupby(key_column, dropna=False)["count"]
            .sum()
            .sort_values(ascending=False)
        )

    def make_labeled_coverage_frame(
        coverage_frame: pd.DataFrame,
        key_columns: Sequence[str],
    ) -> pd.DataFrame:
        labeled_frame = coverage_frame.copy()
        if key_columns:

            def label_row(row: pd.Series) -> str:
                return " · ".join(
                    str(row[col]) if pd.notna(row[col]) else "—" for col in key_columns
                )

            labeled_frame["_label"] = labeled_frame.apply(label_row, axis=1)
        else:
            labeled_frame["_label"] = [f"cell {i}" for i in range(len(labeled_frame))]
        return labeled_frame

    def make_coverage_distribution_card(
        coverage_frame: pd.DataFrame,
    ) -> Card | None:
        if coverage_frame.empty or "count" not in coverage_frame.columns:
            return None

        counts = coverage_frame["count"].astype(int)
        counts_vary = int(counts.max()) != int(counts.min())
        if not counts_vary:
            return None

        top_rows = coverage_frame.sort_values("count", ascending=False).head(15)
        bar_items = [
            BarItem(
                label=truncate(str(row["_label"]), 30),
                value=int(row["count"]),
            )
            for _, row in top_rows.iterrows()
        ]
        return Card(
            title="Per-cell counts",
            description=f"Top {len(bar_items)} cells by sample count",
            content=BarChart(
                items=bar_items,
                height=220,
                orientation="h",
                x_label="Count",
                y_label="Cell",
            ),
            width="w-96",
        )

    def make_coverage_heatmap_card(
        coverage_frame: pd.DataFrame,
        key_columns: Sequence[str],
    ) -> Card | None:
        if len(key_columns) != 2:
            return None

        x_col, y_col = key_columns
        x_vals = [str(v) if pd.notna(v) else "—" for v in coverage_frame[x_col]]
        y_vals = [str(v) if pd.notna(v) else "—" for v in coverage_frame[y_col]]
        x_labels = sorted(set(x_vals))
        y_labels = sorted(set(y_vals))
        if not (0 < len(x_labels) <= 20 and 0 < len(y_labels) <= 20):
            return None

        z: list[list[float]] = [[0.0 for _ in x_labels] for _ in y_labels]
        for x, y, count in zip(
            x_vals, y_vals, coverage_frame["count"].astype(int).tolist()
        ):
            zi = y_labels.index(y)
            zj = x_labels.index(x)
            z[zi][zj] = float(count)
        return Card(
            title="Cross-axis heatmap",
            description=f"{y_col} × {x_col}",
            content=HeatmapChart(
                z=z,
                x_labels=x_labels,
                y_labels=y_labels,
                color=ChartColor.TWO,
                height=260,
                x_label=x_col,
                y_label=y_col,
            ),
            width="w-96",
        )

    def make_per_key_coverage_cards(
        *,
        coverage_frame: pd.DataFrame,
        key_columns: Sequence[str],
    ) -> list[object]:
        if coverage_frame.empty or not key_columns:
            return []
        _cards: list[object] = []
        for index, key_column in enumerate(key_columns):
            per_key = per_key_count_series(coverage_frame, key_column)
            positive = per_key[per_key > 0]
            counts = [int(v) for v in positive.tolist()]
            gini = compute_gini(counts)
            description = (
                f"{len(counts)} ids · {int(sum(counts)):,} samples\n"
                f"Gini {gini:.2f} · {skew_label(gini)}"
            )
            title = f"Coverage: {key_column}"
            if index < 2:
                _cards.append(
                    HistogramCard(
                        data=counts,
                        column=key_column,
                        title=title,
                        description=description,
                        color=ChartColor.TWO,
                        x_label="Samples per id",
                        y_label="Number of ids",
                    )
                )
            else:
                _cards.append(
                    FrequencyBarCard(
                        data=positive,
                        column=key_column,
                        title=title,
                        description=description,
                        color=ChartColor.TWO,
                        top_n=30,
                        x_label="Samples",
                        y_label=key_column,
                        width="w-[500px]",
                    )
                )
        return _cards

    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    coverage_data = _section_results.get("coverage")
    if coverage_run_button.value:
        coverage_data = build_coverage_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "coverage": coverage_data})

    mo.stop(
        coverage_data is None,
        render_section(
            "How is this pool distributed across key cells?",
            coverage_run_button,
        ),
    )

    coverage_frame: pd.DataFrame = coverage_data["coverage_frame"]
    key_columns: list[str] = list(coverage_data["key_columns"])
    labeled_coverage_frame = make_labeled_coverage_frame(
        coverage_frame,
        key_columns,
    )
    distribution_card = make_coverage_distribution_card(labeled_coverage_frame)
    heatmap_card = make_coverage_heatmap_card(
        labeled_coverage_frame,
        key_columns,
    )
    per_key_cards = make_per_key_coverage_cards(
        coverage_frame=labeled_coverage_frame,
        key_columns=key_columns,
    )
    rendered_cards = [
        card.render()
        for card in [distribution_card, heatmap_card, *per_key_cards]
        if card is not None
    ]
    body_items: list[object] = []
    if rendered_cards:
        body_items.append(
            mo.hstack(
                rendered_cards,
                wrap=True,
                align="start",
                justify="start",
            )
        )
    body_items.append(mo.accordion({"Show dataframe": coverage_frame}))
    _body = mo.vstack(body_items)

    render_section(
        "How is this pool distributed across key cells?",
        coverage_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    ensure_sample_frame,
    get_raw_frames,
    get_section_results,
    sample_run_button,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def build_sample_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        sample_frame, key_columns = ensure_sample_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "sample_frame": sample_frame,
        }

    def make_sample_timeline_card(sample_frame: pd.DataFrame) -> HistogramCard:
        created_series = pd.to_datetime(sample_frame["created_at"], errors="coerce")
        earliest = created_series.min()
        latest = created_series.max()
        return HistogramCard(
            data=ts_to_unix(sample_frame, "created_at"),
            column="created_at",
            title="Creation timeline",
            description=(
                "Histogram of sample.created_at (unix seconds). "
                f"\n**Earliest:** {fmt_ts(earliest)}. "
                f"\n**Latest:** {fmt_ts(latest)}."
            ),
            color=ChartColor.TWO,
            binning="continuous",
            nbins=24,
            x_label="Unix seconds",
            y_label="Samples",
        )

    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    sample_data = _section_results.get("sample")
    if sample_run_button.value:
        sample_data = build_sample_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "sample": sample_data})

    mo.stop(
        sample_data is None,
        render_section(
            "What finalized samples are currently in the pool?",
            sample_run_button,
        ),
    )

    _sample_frame: pd.DataFrame = sample_data["sample_frame"]
    _timeline_card = make_sample_timeline_card(_sample_frame)
    _rendered_cards = [_timeline_card.render()]
    _body = mo.vstack(
        [
            mo.hstack(_rendered_cards, wrap=True, align="start"),
            mo.accordion({"Show dataframe": _sample_frame}),
        ]
    )

    render_section(
        "What finalized samples are currently in the pool?",
        sample_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    ensure_error_frame,
    error_run_button,
    get_raw_frames,
    get_section_results,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def build_error_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        error_frame, key_columns = ensure_error_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "error_frame": error_frame,
        }

    def make_error_summary_card(error_frame: pd.DataFrame) -> Card:
        total = len(error_frame)
        if total == 0:
            return stat_card("Error summary", [("Total", "0")])
        max_attempts_value = pd.to_numeric(
            error_frame["attempt_count"], errors="coerce"
        ).max()
        max_attempts = int(max_attempts_value) if pd.notna(max_attempts_value) else 0
        latest = pd.to_datetime(error_frame["created_at"], errors="coerce").max()

        return stat_card(
            "Error summary",
            [
                ("Total errors", fmt_int(total)),
                ("Max attempts", fmt_int(max_attempts)),
                ("Most recent", fmt_ts(latest)),
            ],
        )

    def make_error_attempts_card(
        error_frame: pd.DataFrame,
    ) -> HistogramCard:
        return HistogramCard(
            data=error_frame,
            column="attempt_count",
            title="Attempts before error",
            description="Distribution of attempt_count for errored samples",
            color=ChartColor.ONE,
            x_label="attempt_count",
            y_label="Count",
        )

    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    error_data = _section_results.get("error")
    if error_run_button.value:
        error_data = build_error_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "error": error_data})

    mo.stop(
        error_data is None,
        render_section(
            "What errors have occurred?",
            error_run_button,
        ),
    )

    _error_frame: pd.DataFrame = error_data["error_frame"]
    _summary_card = make_error_summary_card(_error_frame)
    _attempts_card = make_error_attempts_card(_error_frame)
    _rendered_cards = [card.render() for card in [_summary_card, _attempts_card]]
    _body = mo.vstack(
        [
            mo.hstack(_rendered_cards, wrap=True, align="start", justify="start"),
            mo.accordion({"Show dataframe": _error_frame}),
        ]
    )

    render_section(
        "What errors have occurred?",
        error_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    ensure_response_stats_frame,
    get_raw_frames,
    get_section_results,
    response_stats_run_button,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def build_response_stats_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        frame, key_columns = ensure_response_stats_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "response_stats_frame": frame,
        }

    def make_token_mix_card(cs: pd.DataFrame) -> Card:
        prompt_mean = pd.to_numeric(cs["prompt_tokens"], errors="coerce").mean()
        completion_mean = pd.to_numeric(cs["completion_tokens"], errors="coerce").mean()
        reasoning_mean = pd.to_numeric(cs["reasoning_tokens"], errors="coerce").mean()
        return Card(
            title="Token mix (mean)",
            description="Mean tokens per call across prompt / completion / reasoning",
            content=BarChart(
                items=[
                    BarItem(
                        label="prompt",
                        value=float(prompt_mean) if pd.notna(prompt_mean) else 0.0,
                        color=ChartColor.ONE,
                    ),
                    BarItem(
                        label="completion",
                        value=float(completion_mean)
                        if pd.notna(completion_mean)
                        else 0.0,
                        color=ChartColor.TWO,
                    ),
                    BarItem(
                        label="reasoning",
                        value=float(reasoning_mean)
                        if pd.notna(reasoning_mean)
                        else 0.0,
                        color=ChartColor.THREE,
                    ),
                ],
                height=220,
                x_label="Token type",
                y_label="Mean",
            ),
            width="w-80",
        )

    def make_token_distribution_pair(
        cs: pd.DataFrame,
        *,
        column: str,
        label: str,
    ) -> tuple[BoxPlotCard | None, ViolinPlotCard | None]:
        if column not in cs.columns:
            return None, None

        series = pd.to_numeric(cs[column], errors="coerce").dropna()
        if series.empty:
            return None, None

        token_range = (
            f"**True Range:** {float(series.min()):,.0f} - {float(series.max()):,.0f}"
        )
        distribution_card = BoxPlotCard(
            column=column,
            data=cs,
            label=label,
            title=f"{label} distribution",
            description=f"p1 · q1 · median · q3 · p99 (tokens)\n{token_range}",
            tick_format=".0f",
            y_label=f"{label} tokens",
        )
        shape_card = ViolinPlotCard(
            column=column,
            data=cs,
            label=label,
            title=f"{label} shape",
            description=f"KDE on p1-p99 bulk; <=2k sampled points\n{token_range}",
            clip_fences=QuantileFences.P1_P99,
            tick_format=".0f",
            y_label=f"{label} tokens",
        )
        return distribution_card, shape_card

    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    stats_data = _section_results.get("response_stats")
    if response_stats_run_button.value:
        stats_data = build_response_stats_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "response_stats": stats_data})

    mo.stop(
        stats_data is None,
        render_section(
            "What is the token distribution?",
            response_stats_run_button,
        ),
    )

    _stats_frame: pd.DataFrame = stats_data["response_stats_frame"]
    total = len(_stats_frame)
    if total == 0:
        _cards = [stat_card("Token distribution", [("Rows", "0")])]
    else:
        _token_mix_card = make_token_mix_card(_stats_frame)
        _prompt_distribution_card, _prompt_shape_card = make_token_distribution_pair(
            _stats_frame,
            column="prompt_tokens",
            label="Prompt",
        )
        _completion_distribution_card, _completion_shape_card = (
            make_token_distribution_pair(
                _stats_frame,
                column="completion_tokens",
                label="Completion",
            )
        )
        _reasoning_distribution_card, _reasoning_shape_card = (
            make_token_distribution_pair(
                _stats_frame,
                column="reasoning_tokens",
                label="Reasoning",
            )
        )
        _cards = [
            card
            for card in [
                _token_mix_card,
                _prompt_distribution_card,
                _prompt_shape_card,
                _completion_distribution_card,
                _completion_shape_card,
                _reasoning_distribution_card,
                _reasoning_shape_card,
            ]
            if card is not None
        ]

    _rendered_cards = [card.render() for card in _cards]
    _body = mo.vstack(
        [
            _rendered_cards[0],
            mo.hstack(_rendered_cards[1:], wrap=True, align="start", justify="start"),
            mo.accordion({"Show dataframe": _stats_frame}),
        ],
    )

    render_section(
        "What is the token distribution?",
        response_stats_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    ensure_response_stats_frame,
    get_raw_frames,
    get_section_results,
    selected_pool,
    set_raw_frames,
    set_section_results,
    trend_run_button,
):
    def build_trend_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        frame, key_columns = ensure_response_stats_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "response_stats_frame": frame,
        }

    def make_cumulative_cost_card() -> Card:
        return Card(
            title="Cumulative cost",
            description="Would show cumulative USD over time. Plot temporarily disabled while we fix notebook performance for large pools.",
            width="w-96",
        )

    def make_latency_trend_card() -> Card:
        return Card(
            title="Latency over time",
            description="Would show per-call latency_ms over time. Plot temporarily disabled while we fix notebook performance for large pools.",
            width="w-96",
        )

    def make_tokens_trend_card() -> Card:
        return Card(
            title="Tokens over time",
            description="Would show the rolling mean (window=20) of total_tokens over time. Plot temporarily disabled while we fix notebook performance for large pools.",
            width="w-96",
        )

    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    trend_data = _section_results.get("trend")
    if trend_run_button.value:
        trend_data = build_trend_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "trend": trend_data})

    mo.stop(
        trend_data is None,
        render_section(
            "What are the cost and latency trends over time?",
            trend_run_button,
        ),
    )

    _trend_frame: pd.DataFrame = trend_data["response_stats_frame"]
    if _trend_frame.empty or "created_at" not in _trend_frame.columns:
        _cards = [stat_card("Trends", [("Calls", "0")])]
    else:
        _cumulative_cost_card = make_cumulative_cost_card()
        _latency_trend_card = make_latency_trend_card()
        _tokens_card = make_tokens_trend_card()
        _cards = [_cumulative_cost_card, _latency_trend_card, _tokens_card]

    _rendered_cards = [card.render() for card in _cards]
    _body = mo.vstack(
        [
            mo.hstack(_rendered_cards, wrap=True, align="start", justify="start"),
            mo.accordion({"Show dataframe": _trend_frame}),
        ]
    )

    render_section(
        "What are the cost and latency trends over time?",
        trend_run_button,
        _body,
    )
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    leave space
    """)
    return


if __name__ == "__main__":
    app.run()

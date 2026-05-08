import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    from contextlib import contextmanager
    import hashlib
    import json
    import math
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from typing import Any

    import marimo as mo
    import pandas as pd
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        Table,
        Text,
        select,
    )

    from marimo_utils.ui import (
        Badge,
        BadgeVariant,
        BarChart,
        BarItem,
        BoxPlotCard,
        Card,
        ChartColor,
        DataItem,
        HeatmapChart,
        HistogramCard,
        LabeledList,
        LineChart,
        LineSeries,
        PieChart,
        PieSlice,
        QuantileFences,
        ScatterChart,
        ScatterSeries,
        FrequencyBarCard,
        ViolinPlotCard,
        compute_gini,
        skew_label,
    )
    from dr_llm.pool.admin_service import inspect_pool
    from dr_llm.pool.db.runtime import DbConfig, DbRuntime
    from dr_llm.pool.models import PoolInspectionRequest
    from dr_llm.pool.pending.pending_status import PendingStatus
    from dr_llm.pool.reader import (
        PoolReader,
        _load_schema_from_db as load_schema_from_db,
    )
    from dr_llm.project.project_service import (
        maybe_get_project,
        inspect_projects,
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


    def metadata_category(key: str) -> str:
        if key.startswith("_"):
            return "internal"
        if "/" in key:
            return key.split("/", 1)[0]
        return "unprefixed"


    def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=list(columns))


    def build_claims_table(table_name: str) -> Table:
        return Table(
            table_name,
            MetaData(),
            Column("claim_id", Text),
            Column("run_id", Text),
            Column("request_id", Text),
            Column("consumer_tag", Text),
            Column("sample_id", Text),
            Column("claim_idx", Integer),
            Column("claimed_at", DateTime(timezone=True)),
        )


    def build_call_stats_table(table_name: str) -> Table:
        return Table(
            table_name,
            MetaData(),
            Column("sample_id", Text),
            Column("latency_ms", Integer),
            Column("total_cost_usd", Float),
            Column("prompt_tokens", Integer),
            Column("completion_tokens", Integer),
            Column("reasoning_tokens", Integer),
            Column("total_tokens", Integer),
            Column("attempt_count", Integer),
            Column("finish_reason", Text),
            Column("created_at", DateTime(timezone=True)),
        )


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
            schema = load_schema_from_db(runtime, pool_name)
            reader = PoolReader.from_runtime(runtime, schema=schema)
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
                "source_run_id": sample.source_run_id,
                "created_at": sample.created_at,
                "payload_json": compact_json(sample.payload),
                "metadata_json": compact_json(sample.metadata),
            }
            for sample in samples
        ]
        sample_frame = pd.DataFrame.from_records(sample_rows)
        sample_columns = [
            "sample_id",
            *key_columns,
            "sample_idx",
            "source_run_id",
            "created_at",
            "payload_json",
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


    def load_pending_frames(
        *,
        reader: PoolReader,
        key_columns: Sequence[str],
    ) -> dict[str, pd.DataFrame]:
        all_pending = reader.pending_list(
            status=[
                PendingStatus.pending,
                PendingStatus.leased,
                PendingStatus.promoted,
                PendingStatus.failed,
            ]
        )

        pending_rows = [
            {
                "pending_id": pending.pending_id,
                **{
                    column_name: pending.key_values.get(column_name)
                    for column_name in key_columns
                },
                "sample_idx": pending.sample_idx,
                "priority": pending.priority,
                "status": pending.status.value,
                "worker_id": pending.worker_id,
                "lease_expires_at": pending.lease_expires_at,
                "attempt_count": pending.attempt_count,
                "source_run_id": pending.source_run_id,
                "created_at": pending.created_at,
                "payload_json": compact_json(pending.payload),
                "metadata_json": compact_json(pending.metadata),
                "fail_reason": pending.metadata.get("fail_reason", ""),
                "llm_config_json": compact_json(pending.payload.get("llm_config")),
                "prompt_json": compact_json(pending.payload.get("prompt")),
            }
            for pending in all_pending
        ]
        pending_columns = [
            "pending_id",
            *key_columns,
            "sample_idx",
            "priority",
            "status",
            "worker_id",
            "lease_expires_at",
            "attempt_count",
            "source_run_id",
            "created_at",
            "payload_json",
            "metadata_json",
        ]
        failure_columns = [
            "pending_id",
            *key_columns,
            "sample_idx",
            "attempt_count",
            "created_at",
            "fail_reason",
            "payload_json",
            "metadata_json",
        ]
        provenance_columns = [
            *key_columns,
            "sample_idx",
            "status",
            "priority",
            "attempt_count",
            "source_run_id",
            "created_at",
            "llm_config_json",
            "prompt_json",
            "payload_json",
            "metadata_json",
        ]

        pending_frame = pd.DataFrame.from_records(pending_rows)
        if pending_frame.empty:
            return {
                "pending_frame": empty_frame(pending_columns),
                "failure_frame": empty_frame(failure_columns),
                "provenance_frame": empty_frame(provenance_columns),
            }

        pending_frame = pending_frame.loc[:, pending_columns]
        pending_frame = pending_frame.loc[
            pending_frame["status"].isin(
                [PendingStatus.pending.value, PendingStatus.leased.value]
            )
        ]
        if pending_frame.empty:
            pending_frame = empty_frame(pending_columns)
        else:
            pending_frame = sort_frame(
                pending_frame,
                by=[
                    "status",
                    "priority",
                    "created_at",
                    *key_columns,
                    "sample_idx",
                ],
                ascending=[
                    True,
                    False,
                    True,
                    *([True] * len(key_columns)),
                    True,
                ],
            )

        failure_frame = pd.DataFrame.from_records(pending_rows)
        failure_frame = failure_frame.loc[
            failure_frame["status"] == PendingStatus.failed.value
        ]
        if failure_frame.empty:
            failure_frame = empty_frame(failure_columns)
        else:
            failure_frame = failure_frame.loc[:, failure_columns]
            failure_frame = sort_frame(
                failure_frame,
                by=["created_at", *key_columns, "sample_idx"],
                ascending=[False, *([True] * len(key_columns)), True],
            )

        provenance_frame = pd.DataFrame.from_records(pending_rows)
        provenance_frame = provenance_frame.loc[:, provenance_columns]
        provenance_frame = sort_frame(
            provenance_frame,
            by=[*key_columns, "sample_idx", "created_at"],
            ascending=[*([True] * len(key_columns)), True, True],
        )

        return {
            "pending_frame": pending_frame,
            "failure_frame": failure_frame,
            "provenance_frame": provenance_frame,
        }


    def load_metadata_frame(*, reader: PoolReader) -> pd.DataFrame:
        metadata_entries = reader.metadata_prefix("")
        metadata_rows = [
            {
                "key": key,
                "category": metadata_category(key),
                "value_json": compact_json(value),
            }
            for key, value in metadata_entries.items()
        ]
        metadata_frame = pd.DataFrame.from_records(metadata_rows)
        metadata_columns = ["key", "category", "value_json"]
        if metadata_frame.empty:
            return empty_frame(metadata_columns)

        metadata_frame = metadata_frame.loc[:, metadata_columns]
        return sort_frame(
            metadata_frame,
            by=["category", "key"],
            ascending=[True, True],
        )


    def load_claims_and_call_stats_frames(
        *,
        runtime: DbRuntime,
        schema: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        claims_table = build_claims_table(schema.claims_table)
        call_stats_table = build_call_stats_table(schema.call_stats_table)
        with runtime.connect() as conn:
            claim_rows = (
                conn.execute(
                    select(claims_table).order_by(
                        claims_table.c.claimed_at.desc(),
                        claims_table.c.claim_idx.asc(),
                    )
                )
                .mappings()
                .all()
            )
            call_stats_rows = (
                conn.execute(
                    select(call_stats_table).order_by(
                        call_stats_table.c.created_at.desc(),
                        call_stats_table.c.sample_id.asc(),
                    )
                )
                .mappings()
                .all()
            )

        claims_frame = pd.DataFrame.from_records([dict(row) for row in claim_rows])
        claim_columns = [
            "claim_id",
            "run_id",
            "request_id",
            "consumer_tag",
            "sample_id",
            "claim_idx",
            "claimed_at",
        ]
        if claims_frame.empty:
            claims_frame = empty_frame(claim_columns)
        else:
            claims_frame = claims_frame.loc[:, claim_columns]

        call_stats_frame = pd.DataFrame.from_records(
            [dict(row) for row in call_stats_rows]
        )
        call_stats_columns = [
            "sample_id",
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
        if call_stats_frame.empty:
            call_stats_frame = empty_frame(call_stats_columns)
        else:
            call_stats_frame = call_stats_frame.loc[:, call_stats_columns]

        return claims_frame, call_stats_frame


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


    def ensure_pending_frames(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        cache = get_cache()
        if "pending_frames" in cache and "key_columns" in cache:
            return cache["pending_frames"], cache["key_columns"]
        with pool_reader_context(
            project_name=project_name,
            pool_name=pool_name,
        ) as (_, _, reader, key_columns):
            pending_frames = load_pending_frames(
                reader=reader,
                key_columns=key_columns,
            )
            kc = list(key_columns)
        _cache_put(set_cache, pending_frames=pending_frames, key_columns=kc)
        return pending_frames, kc


    def ensure_metadata_frame(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> tuple[pd.DataFrame, list[str]]:
        cache = get_cache()
        if "metadata_frame" in cache and "key_columns" in cache:
            return cache["metadata_frame"], cache["key_columns"]
        with pool_reader_context(
            project_name=project_name,
            pool_name=pool_name,
        ) as (_, _, reader, key_columns):
            metadata_frame = load_metadata_frame(reader=reader)
            kc = list(key_columns)
        _cache_put(set_cache, metadata_frame=metadata_frame, key_columns=kc)
        return metadata_frame, kc


    def ensure_claims_and_call_stats(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        cache = get_cache()
        if (
            "claims_frame" in cache
            and "call_stats_frame" in cache
            and "key_columns" in cache
        ):
            return (
                cache["claims_frame"],
                cache["call_stats_frame"],
                cache["key_columns"],
            )
        with pool_reader_context(
            project_name=project_name,
            pool_name=pool_name,
        ) as (runtime, schema, _, key_columns):
            claims_frame, call_stats_frame = load_claims_and_call_stats_frames(
                runtime=runtime,
                schema=schema,
            )
            kc = list(key_columns)
        _cache_put(
            set_cache,
            claims_frame=claims_frame,
            call_stats_frame=call_stats_frame,
            key_columns=kc,
        )
        return claims_frame, call_stats_frame, kc

    return (
        ensure_claims_and_call_stats,
        ensure_metadata_frame,
        ensure_pending_frames,
        ensure_pool_inspection,
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
                mo.md(
                    "No discovered pools are available for the configured projects."
                ),
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
                mo.md(
                    "One dataframe per question for the currently selected pool."
                ),
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
    pending_run_button = mo.ui.run_button(label="Run", kind="success")
    failure_run_button = mo.ui.run_button(label="Run", kind="success")
    provenance_run_button = mo.ui.run_button(label="Run", kind="success")
    metadata_run_button = mo.ui.run_button(label="Run", kind="success")
    call_stats_run_button = mo.ui.run_button(label="Run", kind="success")
    trend_run_button = mo.ui.run_button(label="Run", kind="success")
    throughput_run_button = mo.ui.run_button(label="Run", kind="success")
    return (
        call_stats_run_button,
        coverage_run_button,
        failure_run_button,
        health_run_button,
        metadata_run_button,
        pending_run_button,
        provenance_run_button,
        sample_run_button,
        throughput_run_button,
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
    ensure_claims_and_call_stats,
    ensure_pool_inspection,
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
        _, call_stats_frame, key_columns = ensure_claims_and_call_stats(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "pool_label": f"{project_name} / {pool_name}",
            "pool_inspection": pool_inspection,
            "key_columns": key_columns,
            "call_stats_frame": call_stats_frame,
        }


    def make_pool_health_pie_card(inspection: Any) -> PoolSimpleStatsPieCard:
        return PoolSimpleStatsPieCard(
            pool=inspection,
            width="20rem",
        )


    def make_pool_finish_reason_card(
        call_stats_frame: pd.DataFrame,
    ) -> Card | None:
        if "finish_reason" not in call_stats_frame.columns:
            return None
        finish_counts = value_counts(
            call_stats_frame["finish_reason"].astype("string").str.lower()
        )
        return Card(
            title="Finish reasons",
            description="Distribution of finish_reason",
            content=PieChart(
                slices=[
                    PieSlice(label=lab, value=cnt) for lab, cnt in finish_counts
                ],
                height=220,
                show_legend=True,
            ),
            width="w-80",
        )


    def make_pool_cost_distribution_plot(
        call_stats_frame: pd.DataFrame,
        column: str = "total_cost_usd",
    ) -> BoxPlotCard | None:
        if column not in call_stats_frame.columns:
            return None
        cost_series = scaled_series(call_stats_frame, column, 1.0)
        cost_range = range_line(
            cost_series, "**True Range:** ${lo:,.4f} - ${hi:,.4f}"
        )
        return BoxPlotCard(
            column=column,
            data=call_stats_frame,
            label="Cost",
            title="Cost distribution",
            description=f"p1 · q1 · median · q3 · p99 ($)\n{cost_range}",
            tick_format="$.4f",
            y_label="Cost (USD)",
        )


    def make_pool_cost_shape_plot(
        call_stats_frame: pd.DataFrame,
        column: str = "total_cost_usd",
    ) -> ViolinPlotCard | None:
        if column not in call_stats_frame.columns:
            return None
        cost_series = scaled_series(call_stats_frame, column, 1.0)
        cost_range = range_line(
            cost_series, "**True Range:** ${lo:,.4f} - ${hi:,.4f}"
        )
        shape_primary = "KDE on p1-p99 bulk; <=2k sampled points"
        return ViolinPlotCard(
            column=column,
            data=call_stats_frame,
            label="Cost",
            title="Cost shape",
            description=f"{shape_primary}\n{cost_range}",
            clip_fences=QuantileFences.P1_P99,
            tick_format="$.4f",
            y_label="Cost (USD)",
        )


    def make_pool_latency_distribution_plot(
        call_stats_frame: pd.DataFrame,
        column: str = "latency_ms",
    ) -> BoxPlotCard | None:
        if column not in call_stats_frame.columns:
            return None
        latency_scale = 0.001
        latency_series = scaled_series(call_stats_frame, column, latency_scale)
        latency_range = range_line(
            latency_series, "**True Range:** {lo:,.2f} - {hi:,.2f}s"
        )
        return BoxPlotCard(
            column=column,
            data=call_stats_frame,
            label="Latency",
            title="Latency distribution",
            description=f"p1 · q1 · median · q3 · p99 (s)\n{latency_range}",
            value_scale=latency_scale,
            tick_format=".2f",
            y_label="Latency (s)",
        )


    def make_pool_latency_shape_plot(
        call_stats_frame: pd.DataFrame,
        column: str = "latency_ms",
    ) -> ViolinPlotCard | None:
        if column not in call_stats_frame.columns:
            return None
        latency_scale = 0.001
        latency_series = scaled_series(call_stats_frame, column, latency_scale)
        latency_range = range_line(
            latency_series, "**True Range:** {lo:,.2f} - {hi:,.2f}s"
        )
        shape_primary = "KDE on p1-p99 bulk; <=2k sampled points"
        return ViolinPlotCard(
            column=column,
            data=call_stats_frame,
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
    call_stats_frame: pd.DataFrame = health_data["call_stats_frame"]

    pool_pie = make_pool_health_pie_card(inspection)
    finish_reasons = make_pool_finish_reason_card(call_stats_frame)
    cost_distribution = make_pool_cost_distribution_plot(call_stats_frame)
    cost_shape = make_pool_cost_shape_plot(call_stats_frame)
    latency_distribution = make_pool_latency_distribution_plot(call_stats_frame)
    latency_shape = make_pool_latency_shape_plot(call_stats_frame)

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
                    str(row[col]) if pd.notna(row[col]) else "—"
                    for col in key_columns
                )

            labeled_frame["_label"] = labeled_frame.apply(label_row, axis=1)
        else:
            labeled_frame["_label"] = [
                f"cell {i}" for i in range(len(labeled_frame))
            ]
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
        created_series = pd.to_datetime(
            sample_frame["created_at"], errors="coerce"
        )
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
    ensure_pending_frames,
    get_raw_frames,
    get_section_results,
    pending_run_button,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def build_pending_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        pending_frames, key_columns = ensure_pending_frames(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "pending_frame": pending_frames["pending_frame"],
        }


    def make_pending_summary_card(pending_frame: pd.DataFrame) -> Card:
        total = len(pending_frame)
        if total == 0:
            return stat_card("Pending summary", [("Total", "0")])
        pending_count = int(
            (pending_frame["status"] == PendingStatus.pending.value).sum()
        )
        leased_count = int(
            (pending_frame["status"] == PendingStatus.leased.value).sum()
        )
        unique_workers = int(pending_frame["worker_id"].dropna().nunique())

        now = pd.Timestamp.utcnow()
        expires = pd.to_datetime(
            pending_frame["lease_expires_at"], utc=True, errors="coerce"
        )
        stale = int(((expires < now) & expires.notna()).sum())

        return stat_card(
            "Pending summary",
            [
                ("Total open", fmt_int(total)),
                ("Pending", fmt_int(pending_count)),
                ("Leased", fmt_int(leased_count)),
                ("Unique workers", fmt_int(unique_workers)),
                ("Stale leases", fmt_int(stale)),
            ],
        )


    def make_pending_status_card(pending_frame: pd.DataFrame) -> Card:
        pending_count = int(
            (pending_frame["status"] == PendingStatus.pending.value).sum()
        )
        leased_count = int(
            (pending_frame["status"] == PendingStatus.leased.value).sum()
        )
        return Card(
            title="Status split",
            description="Pending vs leased",
            content=PieChart(
                slices=[
                    PieSlice(label="pending", value=pending_count),
                    PieSlice(label="leased", value=leased_count),
                ],
                height=220,
                show_legend=True,
            ),
            width="w-80",
        )


    def make_pending_priority_card(
        pending_frame: pd.DataFrame,
    ) -> HistogramCard:
        return HistogramCard(
            data=pending_frame,
            column="priority",
            title="Priority distribution",
            description="Counts by priority value",
            color=ChartColor.FOUR,
            x_label="Priority",
            y_label="Count",
        )


    def make_pending_retries_card(pending_frame: pd.DataFrame) -> Card:
        attempt_counts = value_counts(pending_frame["attempt_count"])
        return Card(
            title="Retry pressure",
            description="Pending items by attempt_count",
            content=BarChart(
                items=[
                    BarItem(label=str(lab), value=cnt)
                    for lab, cnt in attempt_counts
                ],
                height=220,
                x_label="attempt_count",
                y_label="Items",
            ),
            width="w-80",
        )


    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    pending_data = _section_results.get("pending")
    if pending_run_button.value:
        pending_data = build_pending_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "pending": pending_data})

    mo.stop(
        pending_data is None,
        render_section(
            "What pending work exists right now?",
            pending_run_button,
        ),
    )

    _pending_frame: pd.DataFrame = pending_data["pending_frame"]
    _summary_card = make_pending_summary_card(_pending_frame)
    _status_card = make_pending_status_card(_pending_frame)
    _priority_card = make_pending_priority_card(_pending_frame)
    _retries_card = make_pending_retries_card(_pending_frame)
    _rendered_cards = [
        card.render()
        for card in [
            _summary_card,
            _status_card,
            _priority_card,
            _retries_card,
        ]
    ]
    _body = mo.vstack(
        [
            mo.hstack(_rendered_cards, wrap=True, align="start"),
            mo.accordion({"Show dataframe": _pending_frame}),
        ]
    )

    render_section(
        "What pending work exists right now?",
        pending_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    ensure_pending_frames,
    failure_run_button,
    get_raw_frames,
    get_section_results,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def build_failure_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        pending_frames, key_columns = ensure_pending_frames(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "failure_frame": pending_frames["failure_frame"],
        }


    def make_failure_summary_card(failure_frame: pd.DataFrame) -> Card:
        total = len(failure_frame)
        if total == 0:
            return stat_card("Failure summary", [("Total", "0")])
        unique_reasons = int(failure_frame["fail_reason"].dropna().nunique())
        max_attempts_value = pd.to_numeric(
            failure_frame["attempt_count"], errors="coerce"
        ).max()
        max_attempts = (
            int(max_attempts_value) if pd.notna(max_attempts_value) else 0
        )
        latest = pd.to_datetime(failure_frame["created_at"], errors="coerce").max()

        return stat_card(
            "Failure summary",
            [
                ("Total failed", fmt_int(total)),
                ("Unique reasons", fmt_int(unique_reasons)),
                ("Max attempts", fmt_int(max_attempts)),
                ("Most recent", fmt_ts(latest)),
            ],
        )


    def make_failure_reasons_card(failure_frame: pd.DataFrame) -> Card:
        reason_counts = value_counts(failure_frame["fail_reason"])[:10]
        return Card(
            title="Top failure reasons",
            description=f"Top {len(reason_counts)} fail_reason values",
            content=BarChart(
                items=[
                    BarItem(label=truncate(lab, 30), value=cnt)
                    for lab, cnt in reason_counts
                ],
                height=220,
                orientation="h",
                x_label="Failures",
                y_label="Reason",
            ),
            width="w-96",
        )


    def make_failure_attempts_card(
        failure_frame: pd.DataFrame,
    ) -> HistogramCard:
        return HistogramCard(
            data=failure_frame,
            column="attempt_count",
            title="Attempts before fail",
            description="Distribution of attempt_count for failed items",
            color=ChartColor.ONE,
            x_label="attempt_count",
            y_label="Count",
        )


    def make_failure_timeline_card(
        failure_frame: pd.DataFrame,
    ) -> HistogramCard:
        return HistogramCard(
            data=ts_to_unix(failure_frame, "created_at"),
            column="created_at",
            title="Failures over time",
            description="Histogram of created_at (unix seconds)",
            color=ChartColor.ONE,
            binning="continuous",
            nbins=24,
            x_label="Unix seconds",
            y_label="Failures",
        )


    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    failure_data = _section_results.get("failure")
    if failure_run_button.value:
        failure_data = build_failure_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "failure": failure_data})

    mo.stop(
        failure_data is None,
        render_section(
            "What failures have happened?",
            failure_run_button,
        ),
    )

    _failure_frame: pd.DataFrame = failure_data["failure_frame"]
    _summary_card = make_failure_summary_card(_failure_frame)
    _reasons_card = make_failure_reasons_card(_failure_frame)
    _attempts_card = make_failure_attempts_card(_failure_frame)
    _timeline_card = make_failure_timeline_card(_failure_frame)
    _rendered_cards = [
        card.render()
        for card in [
            _summary_card,
            _reasons_card,
            _attempts_card,
            _timeline_card,
        ]
    ]
    _body = mo.vstack(
        [
            mo.hstack(_rendered_cards, wrap=True, align="start", justify="start"),
            mo.accordion({"Show dataframe": _failure_frame}),
        ]
    )

    render_section(
        "What failures have happened?",
        failure_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    ensure_pending_frames,
    get_raw_frames,
    get_section_results,
    provenance_run_button,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def hash_short(s: str, n: int = 6) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


    def build_provenance_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        pending_frames, key_columns = ensure_pending_frames(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "provenance_frame": pending_frames["provenance_frame"],
        }


    def make_provenance_summary_card(provenance_frame: pd.DataFrame) -> Card:
        total = len(provenance_frame)
        if total == 0:
            return stat_card("Provenance", [("Rows", "0")])
        unique_runs = int(provenance_frame["source_run_id"].dropna().nunique())
        unique_configs = int(
            provenance_frame["llm_config_json"].dropna().nunique()
        )
        unique_prompts = int(provenance_frame["prompt_json"].dropna().nunique())

        return stat_card(
            "Provenance summary",
            [
                ("Rows", fmt_int(total)),
                ("Source runs", fmt_int(unique_runs)),
                ("LLM configs", fmt_int(unique_configs)),
                ("Prompt variants", fmt_int(unique_prompts)),
            ],
        )


    def make_provenance_runs_card(provenance_frame: pd.DataFrame) -> Card:
        run_counts = value_counts(provenance_frame["source_run_id"])[:10]
        return Card(
            title="Samples per source run",
            description=f"Top {len(run_counts)} runs",
            content=BarChart(
                items=[
                    BarItem(label=truncate(lab, 24), value=cnt)
                    for lab, cnt in run_counts
                ],
                height=220,
                orientation="h",
                x_label="Rows",
                y_label="Run",
            ),
            width="w-96",
        )


    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    provenance_data = _section_results.get("provenance")
    if provenance_run_button.value:
        provenance_data = build_provenance_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(
            lambda results: {**results, "provenance": provenance_data}
        )

    mo.stop(
        provenance_data is None,
        render_section(
            "What seed or fill provenance defines this pool?",
            provenance_run_button,
        ),
    )

    _provenance_frame: pd.DataFrame = provenance_data["provenance_frame"]
    _summary_card = make_provenance_summary_card(_provenance_frame)
    _runs_card = make_provenance_runs_card(_provenance_frame)
    _rendered_cards = [card.render() for card in [_summary_card, _runs_card]]
    _body = mo.vstack(
        [
            mo.hstack(_rendered_cards, wrap=True, align="start", justify="start"),
            mo.accordion({"Show dataframe": _provenance_frame}),
        ]
    )

    render_section(
        "What seed or fill provenance defines this pool?",
        provenance_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    ensure_metadata_frame,
    get_raw_frames,
    get_section_results,
    metadata_run_button,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def build_metadata_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        metadata_frame, key_columns = ensure_metadata_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "metadata_frame": metadata_frame,
        }


    def make_metadata_summary_card(metadata_frame: pd.DataFrame) -> Card:
        total = len(metadata_frame)
        if total == 0:
            return stat_card("Metadata", [("Keys", "0")])
        categories = metadata_frame["category"].dropna().astype(str)
        unique_categories = int(categories.nunique())
        internal_count = int((categories == "internal").sum())
        user_count = total - internal_count

        return stat_card(
            "Metadata summary",
            [
                ("Keys", fmt_int(total)),
                ("Categories", fmt_int(unique_categories)),
                ("Internal", fmt_int(internal_count)),
                ("User-prefixed", fmt_int(user_count)),
            ],
        )


    def make_metadata_category_card(metadata_frame: pd.DataFrame) -> Card:
        category_counts = value_counts(metadata_frame["category"])
        return Card(
            title="Category split",
            description="Metadata keys grouped by category",
            content=PieChart(
                slices=[
                    PieSlice(label=lab, value=cnt) for lab, cnt in category_counts
                ],
                height=220,
            ),
            width="w-80",
        )


    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    metadata_data = _section_results.get("metadata")
    if metadata_run_button.value:
        metadata_data = build_metadata_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(lambda results: {**results, "metadata": metadata_data})

    mo.stop(
        metadata_data is None,
        render_section(
            "What metadata is attached to this pool?",
            metadata_run_button,
        ),
    )

    _metadata_frame: pd.DataFrame = metadata_data["metadata_frame"]
    _summary_card = make_metadata_summary_card(_metadata_frame)
    _category_card = make_metadata_category_card(_metadata_frame)
    _rendered_cards = [card.render() for card in [_summary_card, _category_card]]
    _body = mo.vstack(
        [
            mo.hstack(_rendered_cards, wrap=True, align="start"),
            mo.accordion({"Show dataframe": _metadata_frame}),
        ],
        align="start",
        justify="start",
    )

    render_section(
        "What metadata is attached to this pool?",
        metadata_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    call_stats_run_button,
    ensure_claims_and_call_stats,
    get_raw_frames,
    get_section_results,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
    def build_call_stats_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        _, _call_stats_frame, key_columns = ensure_claims_and_call_stats(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "call_stats_frame": _call_stats_frame,
        }


    def make_token_mix_card(cs: pd.DataFrame) -> Card:
        prompt_mean = pd.to_numeric(cs["prompt_tokens"], errors="coerce").mean()
        completion_mean = pd.to_numeric(
            cs["completion_tokens"], errors="coerce"
        ).mean()
        reasoning_mean = pd.to_numeric(
            cs["reasoning_tokens"], errors="coerce"
        ).mean()
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

        token_range = f"**True Range:** {float(series.min()):,.0f} - {float(series.max()):,.0f}"
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
    call_stats_data = _section_results.get("call_stats")
    if call_stats_run_button.value:
        call_stats_data = build_call_stats_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(
            lambda results: {**results, "call_stats": call_stats_data}
        )

    mo.stop(
        call_stats_data is None,
        render_section(
            "What is the token distribution?",
            call_stats_run_button,
        ),
    )

    _call_stats_frame: pd.DataFrame = call_stats_data["call_stats_frame"]
    total = len(_call_stats_frame)
    if total == 0:
        _cards = [stat_card("Token distribution", [("Rows", "0")])]
    else:
        _token_mix_card = make_token_mix_card(_call_stats_frame)
        _prompt_distribution_card, _prompt_shape_card = (
            make_token_distribution_pair(
                _call_stats_frame,
                column="prompt_tokens",
                label="Prompt",
            )
        )
        _completion_distribution_card, _completion_shape_card = (
            make_token_distribution_pair(
                _call_stats_frame,
                column="completion_tokens",
                label="Completion",
            )
        )
        _reasoning_distribution_card, _reasoning_shape_card = (
            make_token_distribution_pair(
                _call_stats_frame,
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
            mo.hstack(
                _rendered_cards[1:], wrap=True, align="start", justify="start"
            ),
            mo.accordion({"Show dataframe": _call_stats_frame}),
        ],
    )

    render_section(
        "What is the token distribution?",
        call_stats_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    ensure_claims_and_call_stats,
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
        _, _call_stats_frame, key_columns = ensure_claims_and_call_stats(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "call_stats_frame": _call_stats_frame,
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

    _trend_frame: pd.DataFrame = trend_data["call_stats_frame"]
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


@app.cell(hide_code=True)
def _(
    ensure_claims_and_call_stats,
    ensure_sample_frame,
    get_raw_frames,
    get_section_results,
    selected_pool,
    set_raw_frames,
    set_section_results,
    throughput_run_button,
):
    def build_throughput_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache,
        set_cache,
    ) -> dict[str, Any]:
        _sample_frame, key_columns = ensure_sample_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        _claims_frame, _, _ = ensure_claims_and_call_stats(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "sample_frame": _sample_frame,
            "claims_frame": _claims_frame,
        }


    def make_samples_throughput_card(
        sample_frame: pd.DataFrame,
    ) -> HistogramCard:
        sample_ts = ts_to_unix(sample_frame, "created_at")
        return HistogramCard(
            data=sample_ts,
            column="created_at",
            title="Samples per bucket",
            description=f"Histogram of {len(sample_ts)} sample timestamps",
            color=ChartColor.TWO,
            binning="continuous",
            nbins=24,
            x_label="Unix seconds",
            y_label="Samples",
        )


    def make_claims_throughput_card(
        claims_frame: pd.DataFrame,
    ) -> HistogramCard:
        claim_ts = ts_to_unix(claims_frame, "claimed_at")
        return HistogramCard(
            data=claim_ts,
            column="claimed_at",
            title="Claims per bucket",
            description=f"Histogram of {len(claim_ts)} claim timestamps",
            color=ChartColor.FOUR,
            binning="continuous",
            nbins=24,
            x_label="Unix seconds",
            y_label="Claims",
        )


    def make_workers_throughput_card(claims_frame: pd.DataFrame) -> Card:
        worker_counts: list[tuple[str, int]] = []
        if not claims_frame.empty and "consumer_tag" in claims_frame.columns:
            worker_counts = value_counts(claims_frame["consumer_tag"])[:10]
        return Card(
            title="Workers by activity",
            description=f"Top {len(worker_counts)} consumer_tag values",
            content=BarChart(
                items=[
                    BarItem(label=truncate(lab, 24), value=cnt)
                    for lab, cnt in worker_counts
                ],
                height=220,
                orientation="h",
                x_label="Claims",
                y_label="Worker",
            ),
            width="w-96",
        )


    mo.stop(selected_pool is None)

    _section_results = get_section_results()
    throughput_data = _section_results.get("throughput")
    if throughput_run_button.value:
        throughput_data = build_throughput_data(
            project_name=selected_pool["project_name"],
            pool_name=selected_pool["pool_name"],
            get_cache=get_raw_frames,
            set_cache=set_raw_frames,
        )
        set_section_results(
            lambda results: {**results, "throughput": throughput_data}
        )

    mo.stop(
        throughput_data is None,
        render_section(
            "What does recent activity and throughput look like?",
            throughput_run_button,
        ),
    )

    _sample_frame: pd.DataFrame = throughput_data["sample_frame"]
    _claims_frame: pd.DataFrame = throughput_data["claims_frame"]
    _samples_card = make_samples_throughput_card(_sample_frame)
    _claims_card = make_claims_throughput_card(_claims_frame)
    _workers_card = make_workers_throughput_card(_claims_frame)
    _rendered_cards = [
        card.render() for card in [_samples_card, _claims_card, _workers_card]
    ]
    _body = mo.vstack(
        [
            mo.hstack(_rendered_cards, wrap=True, align="start", justify="start"),
            mo.accordion({"Show dataframe": _claims_frame}),
        ]
    )

    render_section(
        "What does recent activity and throughput look like?",
        throughput_run_button,
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

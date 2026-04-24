import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import hashlib
    from typing import Any
    import json
    import math
    from collections.abc import Sequence
    from datetime import datetime

    import pandas as pd
    import marimo as mo
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

    from marimo_utils import add_marimo_display
    from marimo_utils.ui._rendering import html_block
    from marimo_utils.ui import (
        Badge,
        BadgeVariant,
        BarChart,
        BarItem,
        Card,
        ChartColor,
        DataItem,
        DateStamp,
        HeatmapChart,
        HistogramChart,
        LabeledList,
        LineChart,
        LineSeries,
        PieChart,
        PieSlice,
        ScatterChart,
        ScatterSeries,
    )
    from dr_llm.pool.admin_service import (
        assess_pool_creation,
        create_pool as create_pool_service,
        inspect_pool,
    )
    from dr_llm.pool.db.runtime import DbConfig, DbRuntime
    from dr_llm.pool.models import CreatePoolRequest, PoolInspectionRequest
    from dr_llm.pool.pending.pending_status import PendingStatus
    from dr_llm.pool.reader import (
        PoolReader,
        _load_schema_from_db as load_schema_from_db,
    )
    from dr_llm.project.models import CreateProjectRequest
    from dr_llm.project.project_info import ProjectInfo
    from dr_llm.project.project_service import (
        assess_project_creation,
        create_project as create_project_service,
        maybe_get_project,
        inspect_projects,
    )
    from dr_llm.style import PiePoolCard, Style, bootstrap_tailwind

    IGNORE_DEMO_PROJECTS = True

    add_marimo_display()(ProjectInfo)


@app.cell
def _():
    bootstrap_tailwind()
    return


@app.cell
def _():
    def is_demo_project(project_name: str) -> bool:
        return "demo" in project_name.strip().lower()


    def pool_rows_from_summaries(
        project_summaries: Sequence[Any],
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for summary in project_summaries:
            if summary.pool_inspection.status.value != "discovered":
                continue
            if not summary.pool_inspection.pool_names:
                continue
            if IGNORE_DEMO_PROJECTS and is_demo_project(summary.project.name):
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


    def render_question_frame(question: str, frame: pd.DataFrame) -> mo.Html:
        return mo.vstack(
            [mo.md(f"### {question}"), frame],
            gap=0.75,
        )


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


    def build_pool_drilldown_frames(
        *,
        project_name: str,
        pool_name: str,
    ) -> dict[str, Any]:
        project = maybe_get_project(project_name)
        if project is None or project.dsn is None:
            raise ValueError(f"Project {project_name!r} is not available")

        pool_inspection = inspect_pool(
            PoolInspectionRequest(
                project_name=project_name,
                pool_name=pool_name,
            )
        )

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

            samples = reader.samples_list()
            all_pending = reader.pending_list(
                status=[
                    PendingStatus.pending,
                    PendingStatus.leased,
                    PendingStatus.promoted,
                    PendingStatus.failed,
                ]
            )
            metadata_entries = reader.metadata_prefix("")

            sample_rows = [
                {
                    "sample_id": sample.sample_id,
                    **{
                        column_name: sample.key_values.get(column_name)
                        for column_name in key_columns
                    },
                    "sample_idx": sample.sample_idx,
                    "status": sample.status.value,
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
                "status",
                "source_run_id",
                "created_at",
                "payload_json",
                "metadata_json",
            ]
            if sample_frame.empty:
                sample_frame = empty_frame(sample_columns)
            else:
                sample_frame = sample_frame.loc[:, sample_columns]
                sample_frame = sort_frame(
                    sample_frame,
                    by=[*key_columns, "sample_idx", "created_at"],
                    ascending=[*([True] * len(key_columns)), True, True],
                )

            if sample_frame.empty:
                coverage_frame = empty_frame([*key_columns, "count"])
            else:
                coverage_frame = (
                    sample_frame.loc[:, key_columns]
                    .value_counts(dropna=False)
                    .rename("count")
                    .reset_index()
                )
                coverage_frame = sort_frame(
                    coverage_frame,
                    by=["count", *key_columns],
                    ascending=[False, *([True] * len(key_columns))],
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
                    "llm_config_json": compact_json(
                        pending.payload.get("llm_config")
                    ),
                    "prompt_json": compact_json(pending.payload.get("prompt")),
                }
                for pending in all_pending
            ]
            pending_frame = pd.DataFrame.from_records(pending_rows)
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
            if pending_frame.empty:
                pending_frame = empty_frame(pending_columns)
                failure_frame = empty_frame(
                    [
                        "pending_id",
                        *key_columns,
                        "sample_idx",
                        "attempt_count",
                        "created_at",
                        "fail_reason",
                        "payload_json",
                        "metadata_json",
                    ]
                )
                provenance_frame = empty_frame(
                    [
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
                )
            else:
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
                provenance_frame = provenance_frame.loc[:, provenance_columns]
                provenance_frame = sort_frame(
                    provenance_frame,
                    by=[*key_columns, "sample_idx", "created_at"],
                    ascending=[*([True] * len(key_columns)), True, True],
                )

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
                metadata_frame = empty_frame(metadata_columns)
            else:
                metadata_frame = metadata_frame.loc[:, metadata_columns]
                metadata_frame = sort_frame(
                    metadata_frame,
                    by=["category", "key"],
                    ascending=[True, True],
                )

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

            claims_frame = pd.DataFrame.from_records(
                [dict(row) for row in claim_rows]
            )
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

            return {
                "project_name": project_name,
                "pool_name": pool_name,
                "pool_label": f"{project_name} / {pool_name}",
                "pool_inspection": pool_inspection,
                "key_columns": key_columns,
                "coverage_frame": coverage_frame,
                "sample_frame": sample_frame,
                "pending_frame": pending_frame,
                "failure_frame": failure_frame,
                "provenance_frame": provenance_frame,
                "metadata_frame": metadata_frame,
                "call_stats_frame": call_stats_frame,
                "claims_frame": claims_frame,
            }
        finally:
            runtime.close()


    def tailwind_grid(
        items: Sequence[Any],
        cols: str = "grid-cols-1 sm:grid-cols-2 xl:grid-cols-3",
    ) -> object:
        parts: list[str] = []
        for item in items:
            if item is None:
                continue
            rendered = item.render() if hasattr(item, "render") else item
            parts.append(str(rendered))
        inner = "".join(parts)
        return html_block(f'<div class="grid {cols} gap-4 mb-4">{inner}</div>')


    def render_card_section(
        question: str,
        cards: Sequence[Any],
        frame: pd.DataFrame | None = None,
        *,
        cols: str | None = None,
    ) -> mo.Html:
        grid = tailwind_grid(cards, cols=cols) if cols else tailwind_grid(cards)
        items: list[object] = [mo.md(f"### {question}"), grid]
        if frame is not None:
            items.append(mo.accordion({"Show dataframe": frame}))
        return mo.vstack(items, gap=0.5)


    def _fmt_int(n: Any) -> str:
        if n is None:
            return "—"
        if isinstance(n, float) and math.isnan(n):
            return "—"
        try:
            return f"{int(n):,}"
        except (TypeError, ValueError):
            return str(n)


    def _fmt_float(x: Any, fmt: str = ".2f") -> str:
        if x is None:
            return "—"
        try:
            value = float(x)
        except (TypeError, ValueError):
            return str(x)
        if math.isnan(value):
            return "—"
        return format(value, fmt)


    def _fmt_cost(x: Any) -> str:
        if x is None:
            return "—"
        try:
            value = float(x)
        except (TypeError, ValueError):
            return str(x)
        if math.isnan(value):
            return "—"
        return f"${value:,.4f}"


    def _fmt_ms(x: Any) -> str:
        if x is None:
            return "—"
        try:
            value = float(x)
        except (TypeError, ValueError):
            return str(x)
        if math.isnan(value):
            return "—"
        return f"{value:,.0f} ms"


    def _fmt_ts(ts: Any) -> str:
        if ts is None:
            return "—"
        if isinstance(ts, pd.Timestamp):
            if pd.isna(ts):
                return "—"
            ts = ts.to_pydatetime()
        if isinstance(ts, datetime):
            return ts.astimezone().strftime("%Y-%m-%d %H:%M")
        return str(ts)


    def _hash_short(s: str, n: int = 6) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


    def _truncate(s: str, n: int = 60) -> str:
        if len(s) <= n:
            return s
        return s[: n - 1] + "…"


    def _percentile(values: Sequence[float], p: float) -> float | None:
        cleaned = [
            v
            for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        if not cleaned:
            return None
        return float(pd.Series(cleaned).quantile(p))


    def _value_counts(series: pd.Series) -> list[tuple[str, int]]:
        if series is None or len(series) == 0:
            return []
        cleaned = series.dropna()
        if cleaned.empty:
            return []
        vc = cleaned.value_counts()
        return [(str(label), int(count)) for label, count in vc.items()]


    def _ts_to_unix(frame: pd.DataFrame, column: str) -> list[float]:
        if frame is None or frame.empty or column not in frame.columns:
            return []
        series = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
        return [float(ts.timestamp()) for ts in series]


    def _stat_card(
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


    def build_health_cards(data: dict[str, Any]) -> list[object]:
        inspection = data["pool_inspection"]
        key_columns: list[str] = list(data["key_columns"])
        sample_frame: pd.DataFrame = data["sample_frame"]
        claims_frame: pd.DataFrame = data["claims_frame"]
        call_stats_frame: pd.DataFrame = data["call_stats_frame"]

        status_variant = {
            "complete": BadgeVariant.SECONDARY,
            "in_progress": BadgeVariant.DEFAULT,
            "empty": BadgeVariant.OUTLINE,
        }.get(inspection.status.value, BadgeVariant.OUTLINE)

        identity_children: list[object] = [
            Badge(label=inspection.status.value, variant=status_variant).render(),
        ]
        if inspection.created_at is not None:
            identity_children.append(
                DateStamp(value=inspection.created_at).render()
            )
        if key_columns:
            identity_children.append(
                LabeledList(
                    label="Axes",
                    items=[
                        Badge(label=col, variant=BadgeVariant.SECONDARY)
                        for col in key_columns
                    ],
                ).render()
            )

        identity_card = Card(
            title=inspection.name,
            description=inspection.project_name,
            content=mo.vstack(identity_children, gap=0.5),
            width="w-72",
        )

        totals_card = _stat_card(
            "Totals",
            [
                ("Samples", _fmt_int(inspection.sample_count)),
                ("Pending", _fmt_int(inspection.pending_counts.pending)),
                ("Leased", _fmt_int(inspection.pending_counts.leased)),
                ("Failed", _fmt_int(inspection.pending_counts.failed)),
            ],
        )

        latest_sample_ts = None
        if not sample_frame.empty and "created_at" in sample_frame.columns:
            latest_sample_ts = pd.to_datetime(
                sample_frame["created_at"], errors="coerce"
            ).max()
        latest_claim_ts = None
        if not claims_frame.empty and "claimed_at" in claims_frame.columns:
            latest_claim_ts = pd.to_datetime(
                claims_frame["claimed_at"], errors="coerce"
            ).max()
        active_workers = 0
        if not claims_frame.empty and "consumer_tag" in claims_frame.columns:
            active_workers = int(claims_frame["consumer_tag"].dropna().nunique())

        activity_card = _stat_card(
            "Last activity",
            [
                ("Latest sample", _fmt_ts(latest_sample_ts)),
                ("Latest claim", _fmt_ts(latest_claim_ts)),
                ("Workers seen", _fmt_int(active_workers)),
                ("Claim rows", _fmt_int(len(claims_frame))),
            ],
        )

        total_cost = 0.0
        mean_cost: float | None = None
        median_latency: float | None = None
        if not call_stats_frame.empty:
            if "total_cost_usd" in call_stats_frame.columns:
                costs = pd.to_numeric(
                    call_stats_frame["total_cost_usd"], errors="coerce"
                )
                total_cost = float(costs.sum())
                mean_cost = float(costs.mean()) if len(costs.dropna()) else None
            if "latency_ms" in call_stats_frame.columns:
                median_latency = _percentile(
                    pd.to_numeric(call_stats_frame["latency_ms"], errors="coerce")
                    .dropna()
                    .tolist(),
                    0.5,
                )

        cost_card = _stat_card(
            "Cost so far",
            [
                ("Total cost", _fmt_cost(total_cost)),
                ("Mean / call", _fmt_cost(mean_cost)),
                ("Median latency", _fmt_ms(median_latency)),
                ("Call rows", _fmt_int(len(call_stats_frame))),
            ],
        )

        return [identity_card, totals_card, activity_card, cost_card]


    def build_coverage_cards(data: dict[str, Any]) -> list[object]:
        coverage_frame: pd.DataFrame = data["coverage_frame"]
        key_columns: list[str] = list(data["key_columns"])

        if coverage_frame.empty or "count" not in coverage_frame.columns:
            return [
                _stat_card(
                    "Coverage summary",
                    [("Populated cells", "0"), ("Total samples", "0")],
                )
            ]

        counts = coverage_frame["count"].astype(int)
        total_samples = int(counts.sum())
        populated_cells = int((counts > 0).sum())

        coverage_frame = coverage_frame.copy()
        if key_columns:

            def _label_row(row: pd.Series) -> str:
                return " · ".join(
                    str(row[col]) if pd.notna(row[col]) else "—"
                    for col in key_columns
                )

            coverage_frame["_label"] = coverage_frame.apply(_label_row, axis=1)
            top_row = coverage_frame.sort_values("count", ascending=False).iloc[0]
            top_label = str(top_row["_label"])
            top_count = int(top_row["count"])
        else:
            coverage_frame["_label"] = [
                f"cell {i}" for i in range(len(coverage_frame))
            ]
            top_label, top_count = "—", 0

        summary = _stat_card(
            "Coverage summary",
            [
                ("Populated cells", _fmt_int(populated_cells)),
                ("Total samples", _fmt_int(total_samples)),
                ("Max / cell", _fmt_int(int(counts.max()))),
                ("Min / cell", _fmt_int(int(counts.min()))),
                ("Median / cell", _fmt_float(float(counts.median()), ".1f")),
                ("Top cell", _truncate(f"{top_label} ({top_count})", 40)),
            ],
        )

        top_rows = coverage_frame.sort_values("count", ascending=False).head(15)
        bar_items = [
            BarItem(
                label=_truncate(str(row["_label"]), 30), value=int(row["count"])
            )
            for _, row in top_rows.iterrows()
        ]
        distribution_card = Card(
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

        heatmap_card = None
        if len(key_columns) == 2:
            x_col, y_col = key_columns
            x_vals = [
                str(v) if pd.notna(v) else "—" for v in coverage_frame[x_col]
            ]
            y_vals = [
                str(v) if pd.notna(v) else "—" for v in coverage_frame[y_col]
            ]
            x_labels = sorted(set(x_vals))
            y_labels = sorted(set(y_vals))
            if 0 < len(x_labels) <= 20 and 0 < len(y_labels) <= 20:
                z: list[list[float]] = [[0.0 for _ in x_labels] for _ in y_labels]
                for x, y, count in zip(
                    x_vals, y_vals, coverage_frame["count"].astype(int).tolist()
                ):
                    zi = y_labels.index(y)
                    zj = x_labels.index(x)
                    z[zi][zj] = float(count)
                heatmap_card = Card(
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

        return [summary, distribution_card, heatmap_card]


    def build_sample_cards(data: dict[str, Any]) -> list[object]:
        sample_frame: pd.DataFrame = data["sample_frame"]
        total = len(sample_frame)
        if total == 0:
            return [_stat_card("Sample summary", [("Total", "0")])]

        unique_runs = int(sample_frame["source_run_id"].dropna().nunique())
        created_series = pd.to_datetime(
            sample_frame["created_at"], errors="coerce"
        )
        earliest = created_series.min()
        latest = created_series.max()

        summary = _stat_card(
            "Sample summary",
            [
                ("Total", _fmt_int(total)),
                ("Unique runs", _fmt_int(unique_runs)),
                ("Earliest", _fmt_ts(earliest)),
                ("Latest", _fmt_ts(latest)),
            ],
        )

        status_counts = _value_counts(sample_frame["status"])
        status_card = Card(
            title="Status split",
            description="Distribution of sample.status",
            content=PieChart(
                slices=[
                    PieSlice(label=lab, value=cnt) for lab, cnt in status_counts
                ],
                height=220,
                show_legend=True,
            ),
            width="w-80",
        )

        ts_values = _ts_to_unix(sample_frame, "created_at")
        timeline_card = Card(
            title="Creation timeline",
            description="Histogram of sample.created_at (unix seconds)",
            content=HistogramChart(
                values=ts_values,
                color=ChartColor.TWO,
                nbins=24,
                height=220,
                x_label="Unix seconds",
                y_label="Samples",
            ),
            width="w-80",
        )

        run_counts = _value_counts(sample_frame["source_run_id"])[:10]
        runs_card = Card(
            title="Top source runs",
            description=f"Top {len(run_counts)} runs by sample count",
            content=BarChart(
                items=[
                    BarItem(label=_truncate(lab, 24), value=cnt)
                    for lab, cnt in run_counts
                ],
                height=220,
                orientation="h",
                x_label="Samples",
                y_label="Run",
            ),
            width="w-96",
        )

        return [summary, status_card, timeline_card, runs_card]


    def build_pending_cards(data: dict[str, Any]) -> list[object]:
        pending_frame: pd.DataFrame = data["pending_frame"]
        total = len(pending_frame)
        if total == 0:
            return [_stat_card("Pending summary", [("Total", "0")])]

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

        summary = _stat_card(
            "Pending summary",
            [
                ("Total open", _fmt_int(total)),
                ("Pending", _fmt_int(pending_count)),
                ("Leased", _fmt_int(leased_count)),
                ("Unique workers", _fmt_int(unique_workers)),
                ("Stale leases", _fmt_int(stale)),
            ],
        )

        status_card = Card(
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

        priority_card = Card(
            title="Priority distribution",
            description="Counts by priority value",
            content=HistogramChart(
                values=[
                    float(p)
                    for p in pd.to_numeric(
                        pending_frame["priority"], errors="coerce"
                    )
                    .dropna()
                    .tolist()
                ],
                color=ChartColor.FOUR,
                height=220,
                x_label="Priority",
                y_label="Count",
            ),
            width="w-80",
        )

        attempt_counts = _value_counts(pending_frame["attempt_count"])
        retries_card = Card(
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

        return [summary, status_card, priority_card, retries_card]


    def build_failure_cards(data: dict[str, Any]) -> list[object]:
        failure_frame: pd.DataFrame = data["failure_frame"]
        total = len(failure_frame)
        if total == 0:
            return [_stat_card("Failure summary", [("Total", "0")])]

        unique_reasons = int(failure_frame["fail_reason"].dropna().nunique())
        max_attempts_value = pd.to_numeric(
            failure_frame["attempt_count"], errors="coerce"
        ).max()
        max_attempts = (
            int(max_attempts_value) if pd.notna(max_attempts_value) else 0
        )
        latest = pd.to_datetime(failure_frame["created_at"], errors="coerce").max()

        summary = _stat_card(
            "Failure summary",
            [
                ("Total failed", _fmt_int(total)),
                ("Unique reasons", _fmt_int(unique_reasons)),
                ("Max attempts", _fmt_int(max_attempts)),
                ("Most recent", _fmt_ts(latest)),
            ],
        )

        reason_counts = _value_counts(failure_frame["fail_reason"])[:10]
        reasons_card = Card(
            title="Top failure reasons",
            description=f"Top {len(reason_counts)} fail_reason values",
            content=BarChart(
                items=[
                    BarItem(label=_truncate(lab, 30), value=cnt)
                    for lab, cnt in reason_counts
                ],
                height=220,
                orientation="h",
                x_label="Failures",
                y_label="Reason",
            ),
            width="w-96",
        )

        attempts_card = Card(
            title="Attempts before fail",
            description="Distribution of attempt_count for failed items",
            content=HistogramChart(
                values=[
                    float(v)
                    for v in pd.to_numeric(
                        failure_frame["attempt_count"], errors="coerce"
                    )
                    .dropna()
                    .tolist()
                ],
                color=ChartColor.ONE,
                height=220,
                x_label="attempt_count",
                y_label="Count",
            ),
            width="w-80",
        )

        timeline_card = Card(
            title="Failures over time",
            description="Histogram of created_at (unix seconds)",
            content=HistogramChart(
                values=_ts_to_unix(failure_frame, "created_at"),
                color=ChartColor.ONE,
                nbins=24,
                height=220,
                x_label="Unix seconds",
                y_label="Failures",
            ),
            width="w-80",
        )

        return [summary, reasons_card, attempts_card, timeline_card]


    def build_provenance_cards(data: dict[str, Any]) -> list[object]:
        provenance_frame: pd.DataFrame = data["provenance_frame"]
        total = len(provenance_frame)
        if total == 0:
            return [_stat_card("Provenance", [("Rows", "0")])]

        unique_runs = int(provenance_frame["source_run_id"].dropna().nunique())
        unique_configs = int(
            provenance_frame["llm_config_json"].dropna().nunique()
        )
        unique_prompts = int(provenance_frame["prompt_json"].dropna().nunique())

        summary = _stat_card(
            "Provenance summary",
            [
                ("Rows", _fmt_int(total)),
                ("Source runs", _fmt_int(unique_runs)),
                ("LLM configs", _fmt_int(unique_configs)),
                ("Prompt variants", _fmt_int(unique_prompts)),
            ],
        )

        run_counts = _value_counts(provenance_frame["source_run_id"])[:10]
        runs_card = Card(
            title="Samples per source run",
            description=f"Top {len(run_counts)} runs",
            content=BarChart(
                items=[
                    BarItem(label=_truncate(lab, 24), value=cnt)
                    for lab, cnt in run_counts
                ],
                height=220,
                orientation="h",
                x_label="Rows",
                y_label="Run",
            ),
            width="w-96",
        )

        config_fingerprints = sorted(
            {
                _hash_short(c)
                for c in provenance_frame["llm_config_json"]
                .dropna()
                .astype(str)
                .tolist()
                if c
            }
        )
        config_card = Card(
            title="LLM config diversity",
            description="Short hashes of distinct llm_config payloads",
            content=LabeledList(
                label="Configs",
                items=(
                    [
                        Badge(label=fp, variant=BadgeVariant.SECONDARY)
                        for fp in config_fingerprints[:30]
                    ]
                    if config_fingerprints
                    else [Badge(label="(none)", variant=BadgeVariant.OUTLINE)]
                ),
            ).render(),
            width="w-80",
        )

        prompt_fingerprints = sorted(
            {
                _hash_short(p)
                for p in provenance_frame["prompt_json"]
                .dropna()
                .astype(str)
                .tolist()
                if p
            }
        )
        prompt_card = Card(
            title="Prompt diversity",
            description="Short hashes of distinct prompt payloads",
            content=LabeledList(
                label="Prompts",
                items=(
                    [
                        Badge(label=fp, variant=BadgeVariant.SECONDARY)
                        for fp in prompt_fingerprints[:30]
                    ]
                    if prompt_fingerprints
                    else [Badge(label="(none)", variant=BadgeVariant.OUTLINE)]
                ),
            ).render(),
            width="w-80",
        )

        return [summary, runs_card, config_card, prompt_card]


    def build_metadata_cards(data: dict[str, Any]) -> list[object]:
        metadata_frame: pd.DataFrame = data["metadata_frame"]
        total = len(metadata_frame)
        if total == 0:
            return [_stat_card("Metadata", [("Keys", "0")])]

        categories = metadata_frame["category"].dropna().astype(str)
        unique_categories = int(categories.nunique())
        internal_count = int((categories == "internal").sum())
        user_count = total - internal_count

        summary = _stat_card(
            "Metadata summary",
            [
                ("Keys", _fmt_int(total)),
                ("Categories", _fmt_int(unique_categories)),
                ("Internal", _fmt_int(internal_count)),
                ("User-prefixed", _fmt_int(user_count)),
            ],
        )

        category_counts = _value_counts(metadata_frame["category"])
        category_card = Card(
            title="Category split",
            description="Metadata keys grouped by category",
            content=PieChart(
                slices=[
                    PieSlice(label=lab, value=cnt) for lab, cnt in category_counts
                ],
                height=220,
                show_legend=True,
            ),
            width="w-80",
        )

        key_items = [
            DataItem(
                label=_truncate(str(row["key"]), 28),
                value=_truncate(str(row["value_json"]), 40),
            ).render()
            for _, row in metadata_frame.head(30).iterrows()
        ]
        keys_card = Card(
            title="Keys & values",
            description=f"First {min(30, total)} entries",
            content=mo.vstack(key_items, gap=0.15),
            width="w-96",
        )

        return [summary, category_card, keys_card]


    def build_call_stats_cards(data: dict[str, Any]) -> list[object]:
        cs: pd.DataFrame = data["call_stats_frame"]
        total = len(cs)
        if total == 0:
            return [_stat_card("Call stats", [("Rows", "0")])]

        latency_vals = (
            pd.to_numeric(cs["latency_ms"], errors="coerce").dropna().tolist()
        )
        cost_vals = (
            pd.to_numeric(cs["total_cost_usd"], errors="coerce").dropna().tolist()
        )
        total_tokens_vals = (
            pd.to_numeric(cs["total_tokens"], errors="coerce").dropna().tolist()
        )

        total_cost = float(sum(cost_vals)) if cost_vals else 0.0
        mean_latency = (
            float(sum(latency_vals) / len(latency_vals)) if latency_vals else None
        )
        p50 = _percentile(latency_vals, 0.5)
        p95 = _percentile(latency_vals, 0.95)
        mean_total_tokens = (
            float(sum(total_tokens_vals) / len(total_tokens_vals))
            if total_tokens_vals
            else None
        )

        summary = _stat_card(
            "Generation totals",
            [
                ("Calls", _fmt_int(total)),
                ("Total cost", _fmt_cost(total_cost)),
                ("Mean latency", _fmt_ms(mean_latency)),
                ("p50 latency", _fmt_ms(p50)),
                ("p95 latency", _fmt_ms(p95)),
                ("Mean tokens", _fmt_float(mean_total_tokens, ",.0f")),
            ],
        )

        latency_card = Card(
            title="Latency distribution",
            description="Per-call latency_ms",
            content=HistogramChart(
                values=[float(v) for v in latency_vals],
                color=ChartColor.TWO,
                nbins=30,
                height=220,
                x_label="latency_ms",
                y_label="Calls",
            ),
            width="w-80",
        )

        cost_card = Card(
            title="Cost distribution",
            description="Per-call total_cost_usd",
            content=HistogramChart(
                values=[float(v) for v in cost_vals],
                color=ChartColor.FOUR,
                nbins=30,
                height=220,
                x_label="USD",
                y_label="Calls",
            ),
            width="w-80",
        )

        prompt_mean = pd.to_numeric(cs["prompt_tokens"], errors="coerce").mean()
        completion_mean = pd.to_numeric(
            cs["completion_tokens"], errors="coerce"
        ).mean()
        reasoning_mean = pd.to_numeric(
            cs["reasoning_tokens"], errors="coerce"
        ).mean()
        token_card = Card(
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

        finish_counts = _value_counts(cs["finish_reason"])
        finish_card = Card(
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

        return [summary, latency_card, cost_card, token_card, finish_card]


    def build_trend_cards(data: dict[str, Any]) -> list[object]:
        cs: pd.DataFrame = data["call_stats_frame"]
        if cs.empty or "created_at" not in cs.columns:
            return [_stat_card("Trends", [("Calls", "0")])]

        sorted_cs = cs.copy()
        sorted_cs["_ts"] = pd.to_datetime(
            sorted_cs["created_at"], utc=True, errors="coerce"
        )
        sorted_cs = sorted_cs.dropna(subset=["_ts"]).sort_values("_ts")
        if sorted_cs.empty:
            return [_stat_card("Trends", [("Calls", "0")])]

        x = [float(ts.timestamp()) for ts in sorted_cs["_ts"]]
        cost_series = pd.to_numeric(
            sorted_cs["total_cost_usd"], errors="coerce"
        ).fillna(0.0)
        cumulative_cost = cost_series.cumsum().tolist()
        latency = (
            pd.to_numeric(sorted_cs["latency_ms"], errors="coerce")
            .fillna(0.0)
            .tolist()
        )
        tokens = pd.to_numeric(sorted_cs["total_tokens"], errors="coerce").fillna(
            0.0
        )
        window = min(20, max(1, len(tokens)))
        rolling_tokens = (
            tokens.rolling(window=window, min_periods=1).mean().tolist()
        )

        cumulative_cost_card = Card(
            title="Cumulative cost",
            description="USD over time",
            content=LineChart(
                series=[LineSeries(label="cost", x=x, y=cumulative_cost)],
                height=220,
                x_label="Unix seconds",
                y_label="USD",
            ),
            width="w-96",
        )

        latency_trend_card = Card(
            title="Latency over time",
            description="Per-call latency_ms",
            content=ScatterChart(
                series=[ScatterSeries(label="latency_ms", x=x, y=latency)],
                height=220,
                x_label="Unix seconds",
                y_label="latency_ms",
            ),
            width="w-96",
        )

        tokens_card = Card(
            title="Tokens over time",
            description="Rolling mean (window=20) of total_tokens",
            content=LineChart(
                series=[LineSeries(label="total_tokens", x=x, y=rolling_tokens)],
                height=220,
                x_label="Unix seconds",
                y_label="total_tokens",
            ),
            width="w-96",
        )

        return [cumulative_cost_card, latency_trend_card, tokens_card]


    def build_throughput_cards(data: dict[str, Any]) -> list[object]:
        sample_frame: pd.DataFrame = data["sample_frame"]
        claims_frame: pd.DataFrame = data["claims_frame"]

        sample_ts = _ts_to_unix(sample_frame, "created_at")
        claim_ts = _ts_to_unix(claims_frame, "claimed_at")

        samples_card = Card(
            title="Samples per bucket",
            description=f"Histogram of {len(sample_ts)} sample timestamps",
            content=HistogramChart(
                values=sample_ts,
                color=ChartColor.TWO,
                nbins=24,
                height=220,
                x_label="Unix seconds",
                y_label="Samples",
            ),
            width="w-80",
        )

        claims_card = Card(
            title="Claims per bucket",
            description=f"Histogram of {len(claim_ts)} claim timestamps",
            content=HistogramChart(
                values=claim_ts,
                color=ChartColor.FOUR,
                nbins=24,
                height=220,
                x_label="Unix seconds",
                y_label="Claims",
            ),
            width="w-80",
        )

        worker_counts: list[tuple[str, int]] = []
        if not claims_frame.empty and "consumer_tag" in claims_frame.columns:
            worker_counts = _value_counts(claims_frame["consumer_tag"])[:10]
        workers_card = Card(
            title="Workers by activity",
            description=f"Top {len(worker_counts)} consumer_tag values",
            content=BarChart(
                items=[
                    BarItem(label=_truncate(lab, 24), value=cnt)
                    for lab, cnt in worker_counts
                ],
                height=220,
                orientation="h",
                x_label="Claims",
                y_label="Worker",
            ),
            width="w-96",
        )

        return [samples_card, claims_card, workers_card]

    return (
        build_call_stats_cards,
        build_coverage_cards,
        build_failure_cards,
        build_health_cards,
        build_metadata_cards,
        build_pending_cards,
        build_pool_drilldown_frames,
        build_provenance_cards,
        build_sample_cards,
        build_throughput_cards,
        build_trend_cards,
        pool_rows_from_summaries,
        render_card_section,
        tailwind_grid,
    )


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
    card_style = Style.default()
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
                failed_pool_names.append(
                    (pool_name, f"{type(exc).__name__}: {exc}")
                )
                continue

            cards.append(
                PiePoolCard(
                    pool=pool_inspection,
                    style=card_style,
                    width="20rem",
                ).render()
            )

        if not cards and not failed_pool_names:
            continue

        section_items = [mo.md(f"### {summary.project.name}")]
        if cards:
            section_items.append(
                mo.hstack(cards, wrap=True, justify="start", gap=1)
            )
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


@app.cell(column=3, hide_code=True)
def _(pool_rows_from_summaries, project_summaries):
    pool_rows = pool_rows_from_summaries(project_summaries)
    return (pool_rows,)


@app.cell(hide_code=True)
def _(build_pool_drilldown_frames, pool_selector):
    mo.stop(pool_selector is None or pool_selector.value is None)

    selected_pool = json.loads(pool_selector.value)
    selected_pool_data = build_pool_drilldown_frames(
        project_name=selected_pool["project_name"],
        pool_name=selected_pool["pool_name"],
    )
    return (selected_pool_data,)


@app.cell(hide_code=True)
def _(pool_rows):
    pool_selector = None
    if not pool_rows:
        output = mo.vstack(
            [
                mo.md("## Pool Drilldowns"),
                mo.md("No discovered non-demo pools are available for drilldown."),
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
                mo.md("## Pool Drilldowns"),
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
def _(selected_pool_data):
    mo.md(f"""
    **Selected:** `{selected_pool_data["pool_label"]}`
    """)
    return


@app.cell(hide_code=True)
def _(build_health_cards, selected_pool_data, tailwind_grid):
    mo.vstack(
        [
            mo.md("### Pool health"),
            tailwind_grid(
                build_health_cards(selected_pool_data),
                cols="grid-cols-1 sm:grid-cols-2 lg:grid-cols-4",
            ),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(build_coverage_cards, render_card_section, selected_pool_data):
    render_card_section(
        "How is this pool distributed across key cells?",
        build_coverage_cards(selected_pool_data),
        selected_pool_data["coverage_frame"],
    )
    return


@app.cell(hide_code=True)
def _(build_sample_cards, render_card_section, selected_pool_data):
    render_card_section(
        "What finalized samples are currently in the pool?",
        build_sample_cards(selected_pool_data),
        selected_pool_data["sample_frame"],
    )
    return


@app.cell(hide_code=True)
def _(build_pending_cards, render_card_section, selected_pool_data):
    render_card_section(
        "What pending work exists right now?",
        build_pending_cards(selected_pool_data),
        selected_pool_data["pending_frame"],
    )
    return


@app.cell(hide_code=True)
def _(build_failure_cards, render_card_section, selected_pool_data):
    render_card_section(
        "What failures have happened?",
        build_failure_cards(selected_pool_data),
        selected_pool_data["failure_frame"],
    )
    return


@app.cell(hide_code=True)
def _(build_provenance_cards, render_card_section, selected_pool_data):
    render_card_section(
        "What seed or fill provenance defines this pool?",
        build_provenance_cards(selected_pool_data),
        selected_pool_data["provenance_frame"],
    )
    return


@app.cell(hide_code=True)
def _(build_metadata_cards, render_card_section, selected_pool_data):
    render_card_section(
        "What metadata is attached to this pool?",
        build_metadata_cards(selected_pool_data),
        selected_pool_data["metadata_frame"],
    )
    return


@app.cell(hide_code=True)
def _(build_call_stats_cards, render_card_section, selected_pool_data):
    render_card_section(
        "What are the per-sample generation stats?",
        build_call_stats_cards(selected_pool_data),
        selected_pool_data["call_stats_frame"],
    )
    return


@app.cell(hide_code=True)
def _(build_trend_cards, render_card_section, selected_pool_data):
    render_card_section(
        "What are the cost and latency trends over time?",
        build_trend_cards(selected_pool_data),
        selected_pool_data["call_stats_frame"],
    )
    return


@app.cell(hide_code=True)
def _(build_throughput_cards, render_card_section, selected_pool_data):
    render_card_section(
        "What does recent activity and throughput look like?",
        build_throughput_cards(selected_pool_data),
        selected_pool_data["claims_frame"],
    )
    return


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


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

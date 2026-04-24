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
        Card,
        ChartColor,
        DataItem,
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
    from dr_llm.style import PoolSimpleStatsPieCard, bootstrap_tailwind, wrap_cards

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
    def pool_rows_from_summaries(
        project_summaries: Sequence[Any],
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for summary in project_summaries:
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
            return empty_frame(sample_columns)

        sample_frame = sample_frame.loc[:, sample_columns]
        return sort_frame(
            sample_frame,
            by=[*key_columns, "sample_idx", "created_at"],
            ascending=[*([True] * len(key_columns)), True, True],
        )

    def build_coverage_frame(
        *,
        sample_frame: pd.DataFrame,
        key_columns: Sequence[str],
    ) -> pd.DataFrame:
        if sample_frame.empty:
            return empty_frame([*key_columns, "count"])

        coverage_frame = (
            sample_frame.loc[:, key_columns]
            .value_counts(dropna=False)
            .rename("count")
            .reset_index()
        )
        return sort_frame(
            coverage_frame,
            by=["count", *key_columns],
            ascending=[False, *([True] * len(key_columns))],
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

    def build_health_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
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

    def build_coverage_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
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

    def build_sample_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
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

    def build_pending_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
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

    def build_failure_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
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

    def build_provenance_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
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

    def build_metadata_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
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

    def build_call_stats_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> dict[str, Any]:
        _, call_stats_frame, key_columns = ensure_claims_and_call_stats(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "call_stats_frame": call_stats_frame,
        }

    def build_trend_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> dict[str, Any]:
        return build_call_stats_data(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )

    def build_throughput_data(
        *,
        project_name: str,
        pool_name: str,
        get_cache: CacheGetter,
        set_cache: CacheSetter,
    ) -> dict[str, Any]:
        sample_frame, key_columns = ensure_sample_frame(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        claims_frame, _, _ = ensure_claims_and_call_stats(
            project_name=project_name,
            pool_name=pool_name,
            get_cache=get_cache,
            set_cache=set_cache,
        )
        return {
            "key_columns": key_columns,
            "sample_frame": sample_frame,
            "claims_frame": claims_frame,
        }

    def render_card_section(
        question: str,
        cards: Sequence[Any],
        frame: pd.DataFrame | None = None,
        *,
        include_title: bool = True,
    ) -> mo.Html:
        items: list[object] = [wrap_cards(cards)]
        if include_title:
            items.insert(0, mo.md(f"### {question}"))
        if frame is not None:
            items.append(mo.accordion({"Show dataframe": frame}))
        return mo.vstack(items, gap=0.5)

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
        call_stats_frame: pd.DataFrame = data["call_stats_frame"]

        pie_pool_card = PoolSimpleStatsPieCard(
            pool=inspection,
            width="20rem",
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

        return [pie_pool_card, cost_card]

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
                    str(row[col]) if pd.notna(row[col]) else "—" for col in key_columns
                )

            coverage_frame["_label"] = coverage_frame.apply(_label_row, axis=1)
            top_row = coverage_frame.sort_values("count", ascending=False).iloc[0]
            top_label = str(top_row["_label"])
            top_count = int(top_row["count"])
        else:
            coverage_frame["_label"] = [f"cell {i}" for i in range(len(coverage_frame))]
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
            BarItem(label=_truncate(str(row["_label"]), 30), value=int(row["count"]))
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
            x_vals = [str(v) if pd.notna(v) else "—" for v in coverage_frame[x_col]]
            y_vals = [str(v) if pd.notna(v) else "—" for v in coverage_frame[y_col]]
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
        created_series = pd.to_datetime(sample_frame["created_at"], errors="coerce")
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
                slices=[PieSlice(label=lab, value=cnt) for lab, cnt in status_counts],
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
                    for p in pd.to_numeric(pending_frame["priority"], errors="coerce")
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
                    BarItem(label=str(lab), value=cnt) for lab, cnt in attempt_counts
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
        max_attempts = int(max_attempts_value) if pd.notna(max_attempts_value) else 0
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
        unique_configs = int(provenance_frame["llm_config_json"].dropna().nunique())
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
                for p in provenance_frame["prompt_json"].dropna().astype(str).tolist()
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
                slices=[PieSlice(label=lab, value=cnt) for lab, cnt in category_counts],
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
        completion_mean = pd.to_numeric(cs["completion_tokens"], errors="coerce").mean()
        reasoning_mean = pd.to_numeric(cs["reasoning_tokens"], errors="coerce").mean()
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
                slices=[PieSlice(label=lab, value=cnt) for lab, cnt in finish_counts],
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
            pd.to_numeric(sorted_cs["latency_ms"], errors="coerce").fillna(0.0).tolist()
        )
        tokens = pd.to_numeric(sorted_cs["total_tokens"], errors="coerce").fillna(0.0)
        window = min(20, max(1, len(tokens)))
        rolling_tokens = tokens.rolling(window=window, min_periods=1).mean().tolist()

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
        build_call_stats_data,
        build_coverage_cards,
        build_coverage_data,
        build_failure_cards,
        build_failure_data,
        build_health_cards,
        build_health_data,
        build_metadata_cards,
        build_metadata_data,
        build_pending_cards,
        build_pending_data,
        build_provenance_cards,
        build_provenance_data,
        build_sample_cards,
        build_sample_data,
        build_throughput_cards,
        build_throughput_data,
        build_trend_cards,
        build_trend_data,
        pool_rows_from_summaries,
        render_card_section,
        render_section,
    )


@app.cell(column=1, hide_code=True)
def _(pool_rows_from_summaries, project_summaries):
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
    build_health_cards,
    build_health_data,
    get_raw_frames,
    get_section_results,
    health_run_button,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
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

    _body = None
    if health_data is not None:
        _body = wrap_cards(build_health_cards(health_data))

    render_section("Pool health", health_run_button, _body)
    return


@app.cell(hide_code=True)
def _(
    build_coverage_cards,
    build_coverage_data,
    coverage_run_button,
    get_raw_frames,
    get_section_results,
    render_card_section,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
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

    _body = None
    if coverage_data is not None:
        _body = render_card_section(
            "How is this pool distributed across key cells?",
            build_coverage_cards(coverage_data),
            coverage_data["coverage_frame"],
            include_title=False,
        )

    render_section(
        "How is this pool distributed across key cells?",
        coverage_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    build_sample_cards,
    build_sample_data,
    get_raw_frames,
    get_section_results,
    render_card_section,
    render_section,
    sample_run_button,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
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

    _body = None
    if sample_data is not None:
        _body = render_card_section(
            "What finalized samples are currently in the pool?",
            build_sample_cards(sample_data),
            sample_data["sample_frame"],
            include_title=False,
        )

    render_section(
        "What finalized samples are currently in the pool?",
        sample_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    build_pending_cards,
    build_pending_data,
    get_raw_frames,
    get_section_results,
    pending_run_button,
    render_card_section,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
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

    _body = None
    if pending_data is not None:
        _body = render_card_section(
            "What pending work exists right now?",
            build_pending_cards(pending_data),
            pending_data["pending_frame"],
            include_title=False,
        )

    render_section(
        "What pending work exists right now?",
        pending_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    build_failure_cards,
    build_failure_data,
    failure_run_button,
    get_raw_frames,
    get_section_results,
    render_card_section,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
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

    _body = None
    if failure_data is not None:
        _body = render_card_section(
            "What failures have happened?",
            build_failure_cards(failure_data),
            failure_data["failure_frame"],
            include_title=False,
        )

    render_section(
        "What failures have happened?",
        failure_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    build_provenance_cards,
    build_provenance_data,
    get_raw_frames,
    get_section_results,
    provenance_run_button,
    render_card_section,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
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
        set_section_results(lambda results: {**results, "provenance": provenance_data})

    _body = None
    if provenance_data is not None:
        _body = render_card_section(
            "What seed or fill provenance defines this pool?",
            build_provenance_cards(provenance_data),
            provenance_data["provenance_frame"],
            include_title=False,
        )

    render_section(
        "What seed or fill provenance defines this pool?",
        provenance_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    build_metadata_cards,
    build_metadata_data,
    get_raw_frames,
    get_section_results,
    metadata_run_button,
    render_card_section,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
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

    _body = None
    if metadata_data is not None:
        _body = render_card_section(
            "What metadata is attached to this pool?",
            build_metadata_cards(metadata_data),
            metadata_data["metadata_frame"],
            include_title=False,
        )

    render_section(
        "What metadata is attached to this pool?",
        metadata_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    build_call_stats_cards,
    build_call_stats_data,
    call_stats_run_button,
    get_raw_frames,
    get_section_results,
    render_card_section,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
):
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
        set_section_results(lambda results: {**results, "call_stats": call_stats_data})

    _body = None
    if call_stats_data is not None:
        _body = render_card_section(
            "What are the per-sample generation stats?",
            build_call_stats_cards(call_stats_data),
            call_stats_data["call_stats_frame"],
            include_title=False,
        )

    render_section(
        "What are the per-sample generation stats?",
        call_stats_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    build_trend_cards,
    build_trend_data,
    get_raw_frames,
    get_section_results,
    render_card_section,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
    trend_run_button,
):
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

    _body = None
    if trend_data is not None:
        _body = render_card_section(
            "What are the cost and latency trends over time?",
            build_trend_cards(trend_data),
            trend_data["call_stats_frame"],
            include_title=False,
        )

    render_section(
        "What are the cost and latency trends over time?",
        trend_run_button,
        _body,
    )
    return


@app.cell(hide_code=True)
def _(
    build_throughput_cards,
    build_throughput_data,
    get_raw_frames,
    get_section_results,
    render_card_section,
    render_section,
    selected_pool,
    set_raw_frames,
    set_section_results,
    throughput_run_button,
):
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
        set_section_results(lambda results: {**results, "throughput": throughput_data})

    _body = None
    if throughput_data is not None:
        _body = render_card_section(
            "What does recent activity and throughput look like?",
            build_throughput_cards(throughput_data),
            throughput_data["claims_frame"],
            include_title=False,
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

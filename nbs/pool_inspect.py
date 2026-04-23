import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")

with app.setup:
    import json
    from collections.abc import Sequence

    import pandas as pd
    import marimo as mo
    from sqlalchemy import Column, DateTime, Float, Integer, MetaData, Table, Text, select

    from marimo_utils import add_marimo_display
    from dr_llm.pool.admin_service import (
        assess_pool_creation,
        create_pool as create_pool_service,
        inspect_pool,
    )
    from dr_llm.pool.db.runtime import DbConfig, DbRuntime
    from dr_llm.pool.models import CreatePoolRequest, PoolInspectionRequest
    from dr_llm.pool.pending.pending_status import PendingStatus
    from dr_llm.pool.reader import PoolReader, _load_schema_from_db as load_schema_from_db
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
        project_summaries: Sequence[object],
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
                pool_key = f"{summary.project.name}:{pool_name}"
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
            [
                mo.md(f"### {question}"),
                frame,
            ],
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
    ) -> dict[str, object]:
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
            key_columns = schema.key_column_names

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
                    "llm_config_json": compact_json(pending.payload.get("llm_config")),
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
                        by=["status", "priority", "created_at", *key_columns, "sample_idx"],
                        ascending=[True, False, True, *([True] * len(key_columns)), True],
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
                claim_rows = conn.execute(
                    select(claims_table).order_by(
                        claims_table.c.claimed_at.desc(),
                        claims_table.c.claim_idx.asc(),
                    )
                ).mappings().all()
                call_stats_rows = conn.execute(
                    select(call_stats_table).order_by(
                        call_stats_table.c.created_at.desc(),
                        call_stats_table.c.sample_id.asc(),
                    )
                ).mappings().all()

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

            return {
                "project_name": project_name,
                "pool_name": pool_name,
                "pool_label": f"{project_name} / {pool_name}",
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

    return (
        build_pool_drilldown_frames,
        pool_rows_from_summaries,
        render_question_frame,
    )


@app.cell(column=1)
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
    project_sections = []
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

        project_sections.append(mo.vstack(section_items, gap=1))

    section_items = [mo.md("## Running Project Pools")]
    if ignore_demo_banner is not None:
        section_items.append(ignore_demo_banner)
    section = (
        mo.vstack(
            [*section_items, *project_sections],
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


@app.cell
def _(pool_rows_from_summaries, project_summaries):
    pool_rows = pool_rows_from_summaries(project_summaries)
    return (pool_rows,)


@app.cell(column=3, hide_code=True)
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
        pool_options = {
            row["pool_label"]: row["pool_key"] for row in pool_rows
        }
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


@app.cell
def _(build_pool_drilldown_frames, pool_selector):
    mo.stop(pool_selector is None or pool_selector.value is None)

    selected_project_name, selected_pool_name = pool_selector.value.split(":", 1)
    selected_pool_data = build_pool_drilldown_frames(
        project_name=selected_project_name,
        pool_name=selected_pool_name,
    )
    return (selected_pool_data,)


@app.cell(hide_code=True)
def _(selected_pool_data):
    mo.md(f"**Selected:** `{selected_pool_data['pool_label']}`")
    return


@app.cell(hide_code=True)
def _(render_question_frame, selected_pool_data):
    render_question_frame(
        "How is this pool distributed across key cells?",
        selected_pool_data["coverage_frame"],
    )
    return


@app.cell(hide_code=True)
def _(render_question_frame, selected_pool_data):
    render_question_frame(
        "What finalized samples are currently in the pool?",
        selected_pool_data["sample_frame"],
    )
    return


@app.cell(hide_code=True)
def _(render_question_frame, selected_pool_data):
    render_question_frame(
        "What pending work exists right now?",
        selected_pool_data["pending_frame"],
    )
    return


@app.cell(hide_code=True)
def _(render_question_frame, selected_pool_data):
    render_question_frame(
        "What failures have happened?",
        selected_pool_data["failure_frame"],
    )
    return


@app.cell(hide_code=True)
def _(render_question_frame, selected_pool_data):
    render_question_frame(
        "What seed or fill provenance defines this pool?",
        selected_pool_data["provenance_frame"],
    )
    return


@app.cell(hide_code=True)
def _(render_question_frame, selected_pool_data):
    render_question_frame(
        "What metadata is attached to this pool?",
        selected_pool_data["metadata_frame"],
    )
    return


@app.cell(hide_code=True)
def _(render_question_frame, selected_pool_data):
    render_question_frame(
        "What are the per-sample generation stats?",
        selected_pool_data["call_stats_frame"],
    )
    return


@app.cell(hide_code=True)
def _(render_question_frame, selected_pool_data):
    render_question_frame(
        "How has this pool been consumed?",
        selected_pool_data["claims_frame"],
    )
    return


if __name__ == "__main__":
    app.run()

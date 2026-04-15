import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import re
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import text

    from dr_llm.pool.db.runtime import DbConfig, DbRuntime
    from dr_llm.pool.db.schema import PoolSchema
    from dr_llm.pool.pool_store import PoolStore, SCHEMA_METADATA_KEY
    from dr_llm.pool.reader import PoolReader, _load_schema_from_db as load_schema_from_db
    from dr_llm.project.project_service import (
        create_project as create_project_service,
        list_projects,
    )

    NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
    POOL_TABLE_RE = re.compile(r"^pool_(.+)_samples$")
    POOL_DISCOVERY_SQL = text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' "
        r"AND table_name LIKE 'pool\_%\_samples' "
        "ORDER BY table_name"
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **How this works**

    - `build_project_display()` assembles the visible project summary table.
    - `list_projects()` asks Docker for dr-llm project containers.
    - `check_project_creation_guardrails(...)` blocks project creation when the
      name is invalid, the project already exists, or another project was created
      too recently.
    - `create_project(...)` creates a real Docker-backed project and returns it.
    - `discover_pools(...)` scans `information_schema.tables` for
      `pool_*_samples` tables in each running project database.
    - `get_pool_info(...)` opens a pool via `PoolReader`, reads progress, and
      queries the `_schema` metadata row timestamp as a pool creation proxy.
    - `check_pool_creation_guardrails(...)` blocks pool creation when the project is not
      running, the project already has too many pools, another pool is in progress,
      or a pool was created too recently.
    - `create_pool(...)` runs `ensure_schema()` for real and returns the store.
    - The project and pool creation UIs use marimo `form`s, so the notebook only
      receives values after you submit them.
    - If Docker or DB access fails, the exception bubbles up naturally.
    """)
    return


@app.function
def discover_pools(dsn: str) -> list[str]:
    runtime = DbRuntime(DbConfig(dsn=dsn))
    try:
        with runtime.connect() as conn:
            rows = conn.execute(POOL_DISCOVERY_SQL).fetchall()
        return [
            match.group(1)
            for (table_name,) in rows
            if (match := POOL_TABLE_RE.match(table_name))
        ]
    finally:
        runtime.close()


@app.function
def get_pool_info(project, pool_name: str) -> dict[str, object]:
    if project.dsn is None:
        raise ValueError(f"Project {project.name!r} has no DSN; start it first.")

    runtime = DbRuntime(DbConfig(dsn=project.dsn))
    try:
        schema = load_schema_from_db(runtime, pool_name)
        reader = PoolReader.from_runtime(runtime, schema=schema)
        progress = reader.progress()

        metadata_created_at_sql = text(
            f"SELECT created_at FROM {schema.metadata_table} "
            "WHERE pool_name = :pool_name AND key = :key"
        )
        with runtime.connect() as conn:
            created_at = conn.execute(
                metadata_created_at_sql,
                {"pool_name": schema.name, "key": SCHEMA_METADATA_KEY},
            ).scalar_one_or_none()
    finally:
        runtime.close()

    pending_counts = {
        **progress.pending_counts.model_dump(),
        "total": progress.pending_counts.total,
        "in_flight": progress.pending_counts.in_flight,
    }
    if progress.samples_total == 0 and progress.pending_counts.total == 0:
        status = "empty"
    elif progress.pending_counts.in_flight > 0:
        status = "in_progress"
    else:
        status = "complete"

    return {
        "name": schema.name,
        "schema": schema,
        "created_at": created_at,
        "sample_count": progress.samples_total,
        "pending_counts": pending_counts,
        "status": status,
    }


@app.function
def normalize_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@app.function
def check_pool_creation_guardrails(
    project,
    *,
    max_pools_per_project: int = 5,
    cooldown_seconds: int = 60,
) -> list[dict[str, object]]:
    if not project.running:
        raise ValueError(
            f"Project {project.name!r} must be running before creating a pool."
        )
    if project.dsn is None:
        raise ValueError(f"Project {project.name!r} has no DSN; start it first.")

    pool_names = discover_pools(project.dsn)
    if len(pool_names) >= max_pools_per_project:
        raise ValueError(
            f"Project {project.name!r} already has {len(pool_names)} pools; "
            f"max_pools_per_project={max_pools_per_project}."
        )

    pool_infos = [get_pool_info(project, pool_name) for pool_name in pool_names]

    in_progress_pools = [
        info["name"] for info in pool_infos if info["status"] == "in_progress"
    ]
    if in_progress_pools:
        raise ValueError(
            "Cannot create a new pool while other pools are in progress: "
            + ", ".join(str(name) for name in in_progress_pools)
        )

    cutoff = datetime.now(UTC) - timedelta(seconds=cooldown_seconds)
    recent_pools = [
        info["name"]
        for info in pool_infos
        if (created_at := normalize_utc(info["created_at"])) is not None
        and created_at >= cutoff
    ]
    if recent_pools:
        raise ValueError(
            "Cannot create a new pool yet; recent pools are still within the cooldown window: "
            + ", ".join(str(name) for name in recent_pools)
        )

    return pool_infos


@app.function
def build_project_display():
    projects = list_projects()
    project_rows = []
    for project in projects:
        pools = discover_pools(project.dsn) if project.dsn else []
        project_rows.append(
            {
                "project": project.name,
                "status": str(project.status),
                "port": project.port,
                "dsn": project.dsn,
                "pool_count": len(pools),
                "pools": ", ".join(pools) if pools else "(none)",
            }
        )

    if not project_rows:
        return None

    running_count = sum(row["status"] == "running" for row in project_rows)
    total_pools = sum(row["pool_count"] for row in project_rows)
    return mo.vstack(
        [
            mo.md(
                f"**Projects:** {len(project_rows)}  \
**Running:** {running_count}  \
**Pools discovered:** {total_pools}"
            ),
            mo.ui.table(project_rows),
        ]
    )


@app.function
def create_pool(project, pool_name: str, key_axes: list[str]) -> PoolStore:
    if project.dsn is None:
        raise ValueError(f"Project {project.name!r} has no DSN; start it first.")

    schema = PoolSchema.from_axis_names(pool_name, key_axes)
    runtime = DbRuntime(DbConfig(dsn=project.dsn))
    store = PoolStore(schema, runtime)
    store.ensure_schema()

    print(f"[created] project: {project.name}")
    print(f"[created] dsn: {project.dsn}")
    print(f"[created] pool: {schema.name}")
    print(f"[created] key axes: {schema.key_column_names}")
    print("[created] tables:")
    print(f"  - {schema.samples_table}")
    print(f"  - {schema.claims_table}")
    print(f"  - {schema.pending_table}")
    print(f"  - {schema.metadata_table}")
    print(f"  - {schema.call_stats_table}")
    print("[created] cleanup: call store.close() when you are done with it")

    return store


@app.function
def parse_create_pool_inputs(
    project_name: str,
    pool_name: str,
    axes_csv: str,
):
    normalized_project_name = project_name.strip()
    normalized_pool_name = pool_name.strip()
    key_axes = [axis.strip() for axis in axes_csv.split(",") if axis.strip()]

    if not normalized_project_name:
        raise ValueError("project_name is required")
    if not normalized_pool_name:
        raise ValueError("pool_name is required")
    if not key_axes:
        raise ValueError("At least one key axis is required")

    project = next(
        (
            candidate
            for candidate in list_projects()
            if candidate.name == normalized_project_name
        ),
        None,
    )
    if project is None:
        raise ValueError(f"Project {normalized_project_name!r} not found")

    return project, normalized_pool_name, key_axes


@app.function
def parse_create_project_inputs(project_name: str) -> str:
    normalized_project_name = project_name.strip()
    if not normalized_project_name:
        raise ValueError("project_name is required")
    return normalized_project_name


@app.function
def check_project_creation_guardrails(
    project_name: str,
    *,
    cooldown_seconds: int = 60,
) -> list[object]:
    if not NAME_RE.match(project_name):
        raise ValueError(
            "project_name must be lowercase alphanumeric with underscores, "
            f"starting with a letter; got {project_name!r}"
        )

    projects = list_projects()
    if any(project.name == project_name for project in projects):
        raise ValueError(f"Project {project_name!r} already exists.")

    cutoff = datetime.now(UTC) - timedelta(seconds=cooldown_seconds)
    recent_projects = [
        project.name
        for project in projects
        if (created_at := normalize_utc(project.created_at)) is not None
        and created_at >= cutoff
    ]
    if recent_projects:
        raise ValueError(
            "Cannot create a new project yet; recent projects are still within the cooldown window: "
            + ", ".join(recent_projects)
        )

    return projects


@app.function
def create_project(project_name: str):
    project = create_project_service(project_name)
    print(f"[created] project: {project.name}")
    print(f"[created] status: {project.status}")
    print(f"[created] port: {project.port}")
    print(f"[created] dsn: {project.dsn}")
    return project


@app.function
def parse_get_pool_info_inputs(project_name: str, pool_name: str):
    normalized_project_name = project_name.strip()
    normalized_pool_name = pool_name.strip()

    if not normalized_project_name:
        raise ValueError("project_name is required")
    if not normalized_pool_name:
        raise ValueError("pool_name is required")

    resolved_project = next(
        (
            candidate
            for candidate in list_projects()
            if candidate.name == normalized_project_name
        ),
        None,
    )
    if resolved_project is None:
        raise ValueError(f"Project {normalized_project_name!r} not found")

    return resolved_project, normalized_pool_name


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Pool inspection

    Manual notebook view of dr-llm Docker projects and the pool names currently
    visible inside each running project database.
    """)
    return


@app.cell(hide_code=True)
def _(create_pool_form, create_project_form):
    _ = (create_project_form.value, create_pool_form.value)
    build_project_display()
    return


@app.cell(hide_code=True)
def _(create_project_form):
    mo.stop(create_project_form.value is None)
    project_name = parse_create_project_inputs(**create_project_form.value)
    check_project_creation_guardrails(project_name, cooldown_seconds=60)
    create_project(project_name)
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
    project, pool_name, key_axes = parse_create_pool_inputs(**create_pool_form.value)
    check_pool_creation_guardrails(project, max_pools_per_project=5)
    store = create_pool(project, pool_name, key_axes)
    try:
        get_pool_info(project, pool_name)
    finally:
        store.close()
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
def _():
    get_pool_info_form = (
        mo.md(
            """
            **Selections for Pool Info**

            {project_name}

            {pool_name}
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
        )
        .form(
            submit_button_label="Get pool info",
        )
    )

    get_pool_info_form
    return (get_pool_info_form,)


@app.cell(hide_code=True)
def _(get_pool_info_form):
    mo.stop(get_pool_info_form.value is None)
    pool_info_project, pool_info_name = parse_get_pool_info_inputs(
        **get_pool_info_form.value
    )
    get_pool_info(pool_info_project, pool_info_name)
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()

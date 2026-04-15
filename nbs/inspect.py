import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import re

    from sqlalchemy import text

    from dr_llm.pool.db.runtime import DbConfig, DbRuntime
    from dr_llm.pool.db.schema import PoolSchema
    from dr_llm.pool.pool_store import PoolStore
    from dr_llm.project.project_service import list_projects

    POOL_TABLE_RE = re.compile(r"^pool_(.+)_samples$")
    POOL_DISCOVERY_SQL = text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' "
        r"AND table_name LIKE 'pool\_%\_samples' "
        "ORDER BY table_name"
    )


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

    print(f"[dry-run] project: {project.name}")
    print(f"[dry-run] dsn: {project.dsn}")
    print(f"[dry-run] pool: {schema.name}")
    print(f"[dry-run] key axes: {schema.key_column_names}")
    print("[dry-run] would create tables:")
    print(f"  - {schema.samples_table}")
    print(f"  - {schema.claims_table}")
    print(f"  - {schema.pending_table}")
    print(f"  - {schema.metadata_table}")
    print(f"  - {schema.call_stats_table}")
    print("[dry-run] would call store.ensure_schema()")
    print("[dry-run] would persist the schema into the metadata table under '_schema'")

    return store


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Pool inspection

    Manual notebook view of dr-llm Docker projects and the pool names currently
    visible inside each running project database.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **How this works**

    - `build_project_display()` assembles the visible project summary table.
    - `list_projects()` asks Docker for dr-llm project containers.
    - `discover_pools(...)` scans `information_schema.tables` for
      `pool_*_samples` tables in each running project database.
    - `create_pool(...)` is currently a dry run that builds the schema and
      store, prints what `ensure_schema()` would create, and returns the store.
    - If Docker or DB access fails, the exception bubbles up naturally.
    """)
    return


@app.cell(hide_code=True)
def _():
    build_project_display()
    return


@app.cell
def _():
    # ADD A CREATE POOL RUN BUTTON HERE
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()

# Neon Pool Snapshots

This project can use a two-level Postgres setup:

- local Docker Postgres projects are the write-heavy source of truth
- Neon databases are refreshed read snapshots for travel, notebooks, and
  cross-machine access

Do not run pool-filling workers against Neon unless you intentionally want Neon
to become the writer. Pool leases and completion state are easier to reason
about when there is one writer.

## One-Time Setup

Create one Neon project and one database for each local `dr-llm` project. Use
the direct Neon connection string for Postgres sync operations.

Store the admin URL outside git:

```bash
export DR_LLM_POSTGRES_SYNC_ADMIN_URL='postgresql://.../neondb?sslmode=require'
```

The admin URL can create, rename, and drop databases. Use a separate read-only
URL on travel machines when possible.

## Portable Backups

For a cloud-portable dump that omits local ownership and privilege statements:

```bash
uv run python -m dr_llm project backup PROJECT --portable
```

The backup is written under `~/.dr-llm/backups/PROJECT/`.

## Refresh Neon From Local

Run sync from the machine that has the local Docker project:

```bash
uv run python -m dr_llm project sync-postgres PROJECT
```

The command:

1. plans timestamped temporary and previous database names
2. creates a temporary Neon database
3. dumps the local Docker project with `pg_dump --no-owner --no-privileges`
4. restores into the temporary database with `psql`, using a temporary
   `pgpass` file that is removed after the restore attempt
5. validates public table lists, `pool_catalog` count, and exact table row
   counts
6. renames the old Neon database to a `_prev_...` name
7. renames the validated temporary database to `PROJECT`

By default, the previous database is kept as a rollback point. Drop it after a
successful sync with:

```bash
uv run python -m dr_llm project sync-postgres PROJECT --drop-previous
```

Use `--target-database NAME` when the remote database name differs from the
local project name.

If the final rename from the validated temporary database to the target database
fails after an existing target was moved aside, the sync attempts to rename the
previous database back to the target name. If that rollback also fails, the
command reports a sync swap failure so the remote databases can be inspected
manually before another attempt.

The implementation is provider-neutral Postgres sync. Neon is one compatible
target because its direct connection string supports the required database
create, restore, rename, and drop operations. For a local repeatable check of
the same primitive, run:

```bash
uv run python scripts/demo-project-sync-postgres.py
```

## Read From Another Machine

Install the project and set a Neon reader DSN:

```bash
export DR_LLM_DATABASE_URL='postgresql://.../PROJECT?sslmode=require'
uv run python -m dr_llm pool list-dsn
uv run python -m dr_llm pool inspect-dsn POOL_NAME
```

Python code can use the same DSN with the normal pool readers:

```python
from dr_llm.pool import PoolReader
from dr_llm.pool.db import DbConfig, DbRuntime

runtime = DbRuntime(DbConfig(dsn="postgresql://.../PROJECT?sslmode=require"))
try:
    reader = PoolReader.open("POOL_NAME", runtime=runtime)
    print(reader.progress())
finally:
    runtime.close()
```

Keep Neon credentials out of notebooks, shell history where practical, and git.
If an admin URL is pasted into a shared place, rotate the role password in Neon.

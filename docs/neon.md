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
the direct Neon connection string for restore and sync operations.

Store the admin URL outside git:

```bash
export DR_LLM_NEON_ADMIN_URL='postgresql://.../neondb?sslmode=require'
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
uv run python -m dr_llm project sync-neon PROJECT
```

The command:

1. creates a temporary Neon database
2. dumps the local Docker project with `pg_dump --no-owner --no-privileges`
3. restores into the temporary database
4. validates public table lists, `pool_catalog` count, and exact table row
   counts
5. renames the old Neon database to a `_prev_...` name
6. renames the validated temporary database to `PROJECT`

By default, the previous database is kept as a rollback point. Drop it after a
successful sync with:

```bash
uv run python -m dr_llm project sync-neon PROJECT --drop-previous
```

Use `--target-database NAME` when the remote database name differs from the
local project name.

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

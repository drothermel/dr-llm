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

## Published Tables

Neon is now refreshed from a committed publish whitelist instead of a raw dump
of every local project table. The whitelist lives in
`src/dr_llm/project/data/neon_publish.yml`.

The first published pool is:

- source pool: `nl_latents`
- processor: `nl_latents_samples_v1`
- summary table: `published_nl_latents_samples`
- manifest table: `published_pool_summaries`

The `nl_latents_samples_v1` processor reads `pool_nl_latents_samples` and joins
prompt config metadata from `nl_latents_pool_catalog_entries`. It writes one row
per sample with the original dimensions plus display-ready fields:

- identifiers and dimensions: sample, config, task, family, difficulty, split,
  language, budget, data version, run, call, status, and timestamps
- readable model fields: `enc_model_label`, `dec_model_label`
- prompt config fields: raw prompt config JSON, block ids, block names, and a
  slash-separated `prompt_config_label`
- result fields: `result_state`, `passed`, failure category, normalized failure
  category, budget checks, character counts, and validation summary JSON
- display fields: input code, encoder prompt instructions, full encoder prompt,
  decoder system/task, decoded code, and error detail
- timings: encoder, decoder, and validation seconds when present

To rebuild the published tables locally without syncing Neon:

```bash
uv run python -m dr_llm project start PROJECT
uv run python -m dr_llm project publish-neon PROJECT
uv run python -m dr_llm project stop PROJECT
```

## Refresh Neon From Local

Run sync from the machine that has the local Docker project. The local project
must be running because sync first rebuilds the published summary tables from
the local pool data.

```bash
uv run python -m dr_llm project start PROJECT
uv run python -m dr_llm project sync-postgres PROJECT
uv run python -m dr_llm project stop PROJECT
```

The CLI loads `.env` from the current working directory, so
`DR_LLM_POSTGRES_SYNC_ADMIN_URL` can live there. An explicitly exported shell
value or `--admin-url` still takes precedence.

The command:

1. plans timestamped temporary and previous database names
2. rebuilds the local published summary tables from the whitelist
3. creates a temporary Neon database
4. dumps only the whitelisted published tables with
   `pg_dump --no-owner --no-privileges --table public.TABLE`
5. restores into the temporary database with `psql`, using a temporary
   `pgpass` file that is removed after the restore attempt
6. validates that the temporary database contains exactly the published tables
   and matching row counts
7. renames the old Neon database to a `_prev_...` name
8. renames the validated temporary database to `PROJECT`

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

## Temporary Migration

During rollout, treat the current Neon database as disposable read-side state.
The first sync with this publish flow should create a clean database containing
only:

- `published_pool_summaries`
- `published_nl_latents_samples`

Recommended rollout:

1. Start the local source project:
   `uv run python -m dr_llm project start nl_latents`
2. Rebuild the published tables:
   `uv run python -m dr_llm project publish-neon nl_latents`
3. Inspect row counts in the command output. For `nl_latents`, the source and
   summary counts should match because the v1 summary is one row per sample.
4. Sync to Neon with a temporary target name first:
   `uv run python -m dr_llm project sync-postgres nl_latents --target-database nl_latents_next`
5. Point the viewer at the temporary database with
   `DR_LLM_DATABASE_URL='postgresql://.../nl_latents_next?sslmode=require'`
   and verify `/nl-latents`.
6. Run the same sync against the canonical target database. Keep the generated
   `_prev_...` database until the viewer has been checked.
7. After verification, drop the previous Neon database manually or rerun with
   `--drop-previous`.
8. Stop the local source project when finished:
   `uv run python -m dr_llm project stop nl_latents`

Do not manually delete local pool tables. The cleanup is on the Neon side: the
new sync path should stop copying raw pool/runtime tables to the remote
database.

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

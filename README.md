# dr-llm

Provider-agnostic LLM primitives: call any model, browse catalogs, run batch experiments with typed sample pools.

Domain-neutral by design — shared across repos like `nl_latents` and `unitbench`.

## Two Flows

**Flow 1 — Standalone (no database):**
Call providers, sync model catalogs, browse available models. File-based catalog cache, zero infrastructure.

**Flow 2 — Pool (Postgres-backed):**
Schema-driven sample pools with no-replacement acquisition, pending sample lifecycle, and per-project isolated databases via Docker.

## Install

```bash
uv add dr-llm
```

For the optional marimo notebook in [`nbs/pool_inspect.py`](nbs/pool_inspect.py), install the
notebook extra:

```bash
uv add "dr-llm[notebooks]"
```

## Quick Start

### 1. Query a provider

```bash
uv run dr-llm query \
  --provider openai \
  --model gpt-4.1 \
  --message "Hello, what's 2+2?"
```

No database needed.

### 2. List providers

```bash
uv run dr-llm providers         # human-readable table
uv run dr-llm providers --json  # machine-readable
```

### 3. Sync and browse model catalogs

```bash
uv run dr-llm models sync --provider openai
uv run dr-llm models list --provider openai
uv run dr-llm models show --provider openai --model gpt-4.1
```

Catalog data is cached locally at `~/.dr_llm/catalog_cache/`. No database required.
Human-readable and JSON model listings also include the repo's curated blacklist,
and OpenRouter listings are filtered through the local reasoning-policy allowlist.

## Available Providers

| Provider | Type | Requirements |
|---|---|---|
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `openrouter` | OpenRouter API | `OPENROUTER_API_KEY` |
| `minimax` | MiniMax Anthropic-compatible API | `MINIMAX_API_KEY` |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY` |
| `google` | Google Gemini API | `GOOGLE_API_KEY` |
| `glm` | GLM (ZAI) API | `ZAI_API_KEY` |
| `codex` | Codex CLI (headless) | `codex` executable |
| `claude-code` | Claude Code CLI (headless) | `claude` executable |
| `kimi-code` | Kimi Code API (Anthropic-compatible) | `KIMI_API_KEY` |

Headless providers shell out to CLI tools. `minimax` and `kimi-code` are direct Anthropic-compatible `/messages` API providers. Headless input shapes do not expose `temperature`, `top_p`, or `max_tokens`. `kimi-code` rejects `temperature` and `top_p`, but still requires `max_tokens`.

Some providers use static model lists for `models sync` (no `/models` endpoint). The CLI notes when a list may be out of date and links to docs.

## Python API

`OpenAILlmRequest` / `OpenAILlmConfig` are the concrete request/config shapes for `provider="openai"`. `ApiLlmRequest` / `ApiLlmConfig` are the concrete shapes for the remaining sampling-capable API providers. `KimiCodeLlmRequest` / `KimiCodeLlmConfig` are the concrete shapes for `kimi-code`. `HeadlessLlmRequest` / `HeadlessLlmConfig` are the concrete shapes for CLI-backed providers. `LlmRequest` and `LlmConfig` remain available as unions, and `parse_llm_request(...)` / `parse_llm_config(...)` validate raw payloads into the correct concrete model by `provider`.

For generic sampling-capable API providers, omitted sampling controls default to `temperature=1.0` and `top_p=0.95`. OpenAI omits those fields unless you set them explicitly. `kimi-code` and headless providers reject those fields entirely.

### Calling a provider

```python
from dr_llm.llm import OpenAILlmRequest, build_default_registry
from dr_llm.llm.messages import Message

registry = build_default_registry()
adapter = registry.get("openai")

response = adapter.generate(
    OpenAILlmRequest(
        provider="openai",
        model="gpt-4.1",
        messages=[Message(role="user", content="hello")],
    )
)
print(response.text)
```

### Filling a pool with LLM calls (requires Docker)

The recommended way to populate a pool: declare each variant axis (LLM
configs, prompts, datasets, …), pass them to `seed_llm_grid`, and let
parallel workers make the actual provider calls. `seed_llm_grid` walks
the cross product, builds per-cell payloads in the shape
`make_llm_process_fn` consumes, deduplicates and upserts per-axis
metadata, and bulk-inserts the pending rows in one round-trip. Docker
is used to auto-manage a Postgres project.

```python
from dr_llm import DbConfig, PoolSchema, PoolStore
from dr_llm.llm import build_default_registry
from dr_llm.llm.config import ApiLlmConfig, LlmConfig, OpenAILlmConfig
from dr_llm.llm.messages import Message
from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.llm_pool_adapter import make_llm_process_fn, seed_llm_grid
from dr_llm.pool.pending.backend import PoolPendingBackend, PoolPendingBackendConfig
from dr_llm.pool.pending.grid import Axis, AxisMember, GridCell
from dr_llm.pool.pending.progress import drain, format_pool_progress_line
from dr_llm.project.project_service import create_project
from dr_llm.workers import WorkerConfig, start_workers

# 1. Create a Docker-managed Postgres project
project = create_project("my_eval")

# 2. Build a schema whose key columns match the axis names
schema = PoolSchema.from_axis_names("my_eval", ["llm_config", "prompt"])
runtime = DbRuntime(DbConfig(dsn=project.dsn))
store = PoolStore(schema, runtime)
store.ensure_schema()

# 3. Declare each axis as a list of AxisMembers
llm_config_axis = Axis[LlmConfig](
    name="llm_config",
    members=[
        AxisMember(
            id="gpt-4.1-mini",
            value=OpenAILlmConfig(
                provider="openai",
                model="gpt-4.1-mini",
                max_tokens=64,
            ),
        ),
        AxisMember(
            id="gemini-flash",
            value=ApiLlmConfig(
                provider="google",
                model="gemini-2.5-flash",
                max_tokens=64,
            ),
        ),
    ],
)
prompt_axis = Axis[list[Message]](
    name="prompt",
    members=[
        AxisMember(
            id="haiku",
            value=[Message(role="user", content="Write a haiku about programming.")],
        ),
        AxisMember(
            id="math",
            value=[Message(role="user", content="What is 17 * 23?")],
        ),
    ],
)

# 4. Seed the cross product. seed_llm_grid handles payload shaping,
#    sample_idx expansion, axis-metadata upserts, and bulk insert.
def build_request(cell: GridCell) -> tuple[list[Message], LlmConfig]:
    return cell.values["prompt"], cell.values["llm_config"]

seed_result = seed_llm_grid(
    store,
    axes=[llm_config_axis, prompt_axis],
    build_request=build_request,
    n=2,  # 2 configs × 2 prompts × n=2 = 8 pending rows
)
print(f"Seeded {seed_result.inserted} pending rows")

# 5. Start workers — they call the real providers
registry = build_default_registry()
controller = start_workers(
    PoolPendingBackend(store, config=PoolPendingBackendConfig(max_retries=1)),
    process_fn=make_llm_process_fn(registry),
    config=WorkerConfig(num_workers=4, thread_name_prefix="pool-fill"),
)

# 6. Drain to idle, printing one line per visible state change
try:
    drain(controller, on_change=lambda snap: print(format_pool_progress_line(snap)))
finally:
    controller.stop()
    controller.join()

# 7. Acquire samples (no-replacement within a run)
from dr_llm.pool.models import AcquireQuery
result = store.acquire(AcquireQuery(
    run_id="eval_run_1",
    key_values={"llm_config": "gpt-4.1-mini", "prompt": "math"},
    n=2,
))

# 8. Clean up when done
registry.close()
runtime.close()
```

See `scripts/demo-pool-fill.py` for a complete runnable example.

### Reading an existing pool

Once a pool has been seeded and filled, `PoolReader` gives consumers a
typed read-only handle for inspection without re-wiring `DbRuntime` /
`PoolSchema` / `PoolStore` by hand. The reader composes a private
`PoolStore` and exposes only its read-side methods.

```python
from dr_llm import PoolReader

with PoolReader.open("my_eval", "my_eval") as reader:
    progress = reader.progress()
    print(
        f"{progress.samples_total} promoted, "
        f"{progress.in_flight} in-flight, "
        f"{progress.pending_counts.failed} failed"
    )

    # Typed PoolSample iterator/list with optional key + status filters
    for sample in reader.samples_list(key_filter={"llm_config": "gpt-4.1-mini"}):
        print(sample.sample_id, sample.payload)

    # Scan consumer-owned axis metadata by key prefix
    templates = reader.metadata_prefix("prompt_template/")
```

`PoolReader.open(project, pool)` resolves the project DSN, constructs a
`DbRuntime`, and reads the pool's `PoolSchema` from the metadata table
where `PoolStore.ensure_schema()` persists it under the reserved key
`_schema`. Pools created before this feature shipped raise
`PoolSchemaNotPersistedError` on `open()`; use
`PoolReader.from_runtime(runtime, schema=...)` to inspect them with an
explicit schema, or re-run `ensure_schema()` once to backfill the row.
For a manual backfill, construct the known schema explicitly, for example
`PoolSchema.from_axis_names(pool_name, ["prompt_template_id", "data_sample_id", "llm_config_id"])`,
then run `PoolStore(schema, runtime).ensure_schema()`. Outcome: the pool's
tables and indexes remain unchanged, and the missing `_schema` metadata row is
upserted so future `PoolReader.open(...)` and `inspect_pool(...)` calls work
normally without an explicit schema.

### Migrating existing pools to `call_stats`

New pools get `pool_{name}_call_stats` automatically when `store.ensure_schema()`
runs. Existing pools created before this change need a one-time migration.

```bash
# inspect what the migration would do
uv run python scripts/migrate-call-stats.py --dry-run

# create missing call_stats tables for all pools
uv run python scripts/migrate-call-stats.py

# create + backfill historical stats for one pool
uv run python scripts/migrate-call-stats.py --pool my_eval --backfill
```

Use `--backfill` when historical `pool_{name}_samples.payload_json` rows already
contain response metrics like `latency_ms`, `usage`, `cost`, and
`finish_reason`, and you want those copied into `call_stats`. If you only need
the new table for future promotions, run the script without `--backfill`. Pass
`--dsn` to target a database other than `DR_LLM_DATABASE_URL`.

## CLI Reference

```bash
# Providers
dr-llm providers [--json]

# Model catalog (file-based, no DB needed)
dr-llm models sync [--provider NAME] [--verbose]
dr-llm models list [--provider NAME] [--supports-reasoning] [--model-contains TEXT] [--json]
dr-llm models sync-list [--provider NAME] [--supports-reasoning] [--model-contains TEXT] [--json]
dr-llm models show --provider NAME --model NAME

# Query
dr-llm query --provider NAME --model NAME --message TEXT
dr-llm query --provider openai --model gpt-5-mini --reasoning-json '{"kind":"openai","thinking_level":"high"}' --message TEXT
dr-llm query --provider codex --model gpt-5.1-codex-mini --reasoning-json '{"kind":"codex","thinking_level":"xhigh"}' --message TEXT
dr-llm query --provider google --model gemini-2.5-flash --reasoning-json '{"kind":"google","thinking_level":"budget","budget_tokens":512}' --message TEXT
dr-llm query --provider openrouter --model openai/gpt-oss-20b --reasoning-json '{"kind":"openrouter","effort":"high"}' --message TEXT

# Sampling / token controls
# Generic sampling API providers default omitted sampling controls to temperature=1.0 and top_p=0.95.
# OpenAI omits temperature/top_p unless you set them explicitly.
# OpenAI GPT-5 custom temperature/top_p controls are only supported on gpt-5.2/gpt-5.4 with reasoning off.
# --temperature, --top-p, and --max-tokens are rejected for headless providers (codex, claude-code)
# --temperature and --top-p are also rejected for kimi-code; --max-tokens is required there

# Projects (Docker-managed Postgres)
dr-llm project create NAME
dr-llm project list
dr-llm project use NAME
dr-llm project start|stop NAME
dr-llm pool destroy PROJECT_NAME POOL_NAME --yes-really-delete-everything
dr-llm project backup NAME
dr-llm project restore NAME BACKUP_PATH  # BACKUP_PATH must be .sql.gz
dr-llm project destroy NAME --yes-really-delete-everything
```

### Deleting pools and projects

Deletion now uses one standard primitive: pool deletion.

- `dr-llm pool destroy PROJECT_NAME POOL_NAME --yes-really-delete-everything`
  deletes the fixed pool table set for that pool name: `samples`, `claims`,
  `pending`, `metadata`, and `call_stats`.
- direct pool deletion requires the project to be running and blocks if the
  pool is still in progress
- legacy pools without persisted `_schema` metadata can still be deleted,
  because deletion targets the derived table names directly rather than loading
  `PoolSchema` from metadata

`dr-llm project destroy` is now an orchestrator over pool deletion rather than a
blind Docker destroy.

- if the project is stopped, it is started temporarily for pool discovery and deletion
- discovered pools are deleted with bounded parallelism, but result ordering is
  deterministic and follows pool discovery order rather than completion order
- if any pool deletion fails, project container and volume deletion are skipped
- if the project had to be started temporarily and deletion fails, it is stopped
  again to restore the original state

Both destroy commands now emit structured JSON results. For project deletion,
the payload includes `discovered_pool_names`, ordered `pool_results`,
`temporarily_started`, and `destroyed_project_resources`.

## Configuration

Generation transcript logging (default on, used for LLM call debugging):

| Variable | Default |
|---|---|
| `DR_LLM_GENERATION_LOG_ENABLED` | `true` |
| `DR_LLM_GENERATION_LOG_DIR` | `.dr_llm/generation_logs` |
| `DR_LLM_GENERATION_LOG_ROTATE_BYTES` | `104857600` (100MB) |
| `DR_LLM_GENERATION_LOG_BACKUPS` | `10` |
| `DR_LLM_GENERATION_LOG_REDACT_ENABLED` | `true` |

Provider endpoint defaults:
- GLM: `https://api.z.ai/api/coding/paas/v4`
- MiniMax API: `https://api.minimax.io/anthropic/v1/messages`
- Kimi Code API: `https://api.kimi.com/coding/v1/messages`

## Testing

```bash
uv run ruff format && uv run ruff check --fix .
uv run ty check
uv run pytest tests/ -v -m "not integration"
```

### Integration tests (requires Docker)

```bash
./scripts/run-tests-local.sh
```

`pytest` now defaults to `pytest-xdist`, so `uv run pytest tests/ -v -m "not integration"` runs the safe non-integration suite in parallel. `run-tests-local.sh` forces `-n 0`, auto-creates a temporary Docker Postgres project, runs `pytest -m integration`, and destroys it on exit. Pass extra pytest args for targeted runs: `./scripts/run-tests-local.sh -k test_pool_store`.

## Demo Scripts

### Provider discovery (no DB needed)

```bash
uv run python scripts/demo-providers.py
```

Lists all supported providers, syncs and displays model catalogs for each available one.

### Pool provider demo (requires Docker)

```bash
uv run python scripts/demo-pool-providers.py
```

Creates a project, queries every available provider, stores results in a typed pool, prints a summary table. Run with `--help` for options.

### Pool fill worker demo (requires Docker + API keys)

```bash
uv run python scripts/demo-pool-fill.py
```

Auto-creates a Docker Postgres project, seeds an `(llm_config, prompt)` pool via `seed_llm_grid` from declared `Axis` instances, starts workers that make real LLM calls via `make_llm_process_fn`, drains the queue to idle with `drain`, shows response snippets, and destroys the project on exit. Pass `--dsn` to use an existing database instead. Run with `--help` for options.

### Reasoning and effort demo (live API / CLI checks)

```bash
uv run python scripts/demo_thinking_and_effort.py
```

Exercises the branch's provider-specific reasoning and effort validation against
curated model sets for OpenAI, OpenRouter, Google, Codex, Claude Code, MiniMax,
and Kimi Code. Use `--provider` to limit the run to one provider.

Reasoning configs are validated before dispatch. For example, OpenAI GPT-5
family models use configs like `{"kind":"openai","thinking_level":"high"}`,
Codex reasoning-capable models also accept `{"kind":"codex","thinking_level":"xhigh"}`,
Google 2.5 models accept budget configs like
`{"kind":"google","thinking_level":"budget","budget_tokens":512}`, `minimax`
requires `{"kind":"anthropic","thinking_level":"na"}` together with an explicit
`--effort`, `kimi-code` uses Anthropic-compatible reasoning like
`{"kind":"anthropic","thinking_level":"adaptive"}` together with an explicit
`--effort` and `--max-tokens`, and OpenRouter reasoning-capable models use
`{"kind":"openrouter", ...}` with either `enabled` or `effort` depending on the
repo's curated model policy.

See [`OPEN_ROUTER_REASONING_NOTES.md`](/Users/daniellerothermel/drotherm/repos/dr-llm/OPEN_ROUTER_REASONING_NOTES.md)
for the direct API observations that informed the OpenRouter policy layer.

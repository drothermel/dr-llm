# dr-llm

Provider-agnostic LLM primitives: call any model, browse catalogs, run batch experiments with typed sample pools.

Domain-neutral by design — shared across repos like `nl_latents` and `unitbench`.

## Two Flows

**Flow 1 — Standalone (no database):**
Call providers, sync model catalogs, browse available models. File-based catalog cache, zero infrastructure.

**Flow 2 — Pool (Postgres-backed):**
Schema-driven sample pools with a unified two-table design
(`pool_<name>_samples` + `pool_<name>_leases`), no-replacement acquisition,
and per-project isolated databases via Docker.

## Install

```bash
uv add dr-llm
```

For the optional marimo pool-inspection notebooks in
[`nbs/inspect/`](nbs/inspect), install the notebook extra:

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
`make_llm_process_fn` consumes, and bulk-inserts the unfilled sample
rows in one round-trip. Docker is used to auto-manage a Postgres
project.

```python
import time

from dr_llm import DbConfig, PoolSchema, PoolStore
from dr_llm.llm import build_default_registry
from dr_llm.llm.config import ApiLlmConfig, LlmConfig, OpenAILlmConfig
from dr_llm.llm.messages import Message
from dr_llm.pool.backend import LlmPoolBackend, LlmPoolBackendConfig, make_llm_process_fn
from dr_llm.pool.db.runtime import DbRuntime
from dr_llm.pool.seed_grid import Axis, AxisMember, GridCell, seed_llm_grid
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
    n=2,  # 2 configs × 2 prompts × n=2 = 8 sample rows
)
print(f"Seeded {seed_result.inserted} sample rows")

# 5. Start workers — they call the real providers
registry = build_default_registry()
controller = start_workers(
    LlmPoolBackend(store, config=LlmPoolBackendConfig(max_retries=1)),
    process_fn=make_llm_process_fn(registry),
    config=WorkerConfig(num_workers=4, thread_name_prefix="pool-fill"),
)

# 6. Drain to idle
try:
    while True:
        snapshot = controller.snapshot()
        state = snapshot.backend_state
        if state is not None:
            print(f"incomplete={state.incomplete} complete={state.complete}")
            if state.incomplete == 0:
                break
        time.sleep(0.5)
finally:
    controller.stop()
    controller.join()

# 7. Acquire samples (no-replacement, per-consumer)
#    Sample acquisition lives in dr_llm.sampling. Each consumer gets its
#    own claims table; setup_consumer/teardown_consumer manages it.
from dr_llm.sampling.acquisition import AcquireQuery
from dr_llm.sampling.sampling_store import SamplingStore

sampling = SamplingStore(store.schema, runtime, store._tables)
sampling.setup_consumer("eval_consumer_1")
result = sampling.acquire(
    AcquireQuery(
        run_id="eval_run_1",
        key_values={"llm_config": "gpt-4.1-mini", "prompt": "math"},
        n=2,
    ),
    "eval_consumer_1",
)

# 8. Clean up when done
sampling.teardown_consumer("eval_consumer_1")
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
from dr_llm.pool.db.runtime import DbConfig, DbRuntime
from dr_llm.project.project_service import maybe_get_project

project = maybe_get_project("my_eval")
runtime = DbRuntime(DbConfig(dsn=project.dsn))
try:
    with PoolReader.open("provider_queries", runtime=runtime) as reader:
        progress = reader.progress()
        print(
            f"total={progress.total} "
            f"complete={progress.complete} "
            f"incomplete={progress.incomplete} "
            f"leased={progress.leased} "
            f"error={progress.error}"
        )

        # Typed PoolSample iterator/list with optional key + completion filters
        for sample in reader.samples_list(
            key_filter={"llm_config": "gpt-4.1-mini"},
        ):
            print(sample.sample_id, sample.request, sample.response)
finally:
    runtime.close()
```

`PoolReader.open(project, pool)` resolves the project DSN, constructs a
`DbRuntime`, and reads the pool's `PoolSchema` from the project-global
`pool_catalog` table, where `PoolStore.ensure_schema()` persists it.
Pools created before catalog persistence shipped raise
`PoolSchemaNotPersistedError` on `open()`; use
`PoolReader.from_runtime(runtime, schema=...)` to inspect them with an
explicit schema, or run `dr-llm pool backfill-catalog PROJECT_NAME` once
to derive each pool's schema from its samples table and persist it (see
the migration section below).

### Backfilling `pool_catalog` for legacy projects

Projects created before catalog persistence have working pool tables but
no `pool_catalog` row, so `load_schema()` and anything that depends on it
(`PoolReader.open`, `inspect_pool`, `dr-llm project destroy`) fail.

```bash
dr-llm pool backfill-catalog PROJECT_NAME [--dry-run]
```

For each pool in the project the command derives a `PoolSchema` by
inspecting the existing `pool_<name>_samples` table, creates `pool_catalog`
if it does not already exist, and persists the schema row. The project is
started temporarily if it is stopped and restored to its original state on
exit. Each pool is processed independently, so one pool's failure does not
block the others, and the command is idempotent — already-persisted pools
are reported as `already_persisted`. The same logic is exposed
programmatically as `backfill_project_catalog` in
`dr_llm.pool.admin.migration`.

Pools whose samples table predates this branch's split of `payload_json`
into `request_json`/`response_json` cannot be backfilled — that is a data
migration, not a catalog migration.

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
dr-llm pool destroy-testish PROJECT_NAME --yes-really-delete-everything
dr-llm pool destroy-testish PROJECT_NAME --dry-run
dr-llm pool backfill-catalog PROJECT_NAME [--dry-run]
dr-llm project backup NAME
dr-llm project restore NAME BACKUP_PATH  # BACKUP_PATH must be .sql.gz
dr-llm project destroy NAME --yes-really-delete-everything
```

### Deleting pools and projects

Deletion now uses one standard primitive: pool deletion.

- `dr-llm pool destroy PROJECT_NAME POOL_NAME --yes-really-delete-everything`
  deletes the fixed pool table set for that pool name (`pool_<name>_samples`
  and `pool_<name>_leases`) and removes the pool's row from `pool_catalog`.
- `dr-llm pool destroy-testish PROJECT_NAME --yes-really-delete-everything`
  discovers pools in that project and deletes only the ones whose
  underscore-delimited lowercase name tokens include `test`, `tst`, `smoke`, or `demo`
- `dr-llm pool destroy-testish PROJECT_NAME --dry-run` previews the matched
  pools and returns the same structured result shape without deleting anything
- direct pool deletion requires the project to be running, but leased rows
  do not block deletion
- legacy pools without persisted `pool_catalog` metadata can still be deleted,
  because deletion targets the derived table names directly rather than loading
  `PoolSchema` from `pool_catalog`

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

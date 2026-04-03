# dr-llm

Provider-agnostic LLM primitives: call any model, browse catalogs, run batch experiments with typed sample pools.

Domain-neutral by design — shared across repos like `nl_latents` and `unitbench`.

## Two Flows

**Flow 1 — Standalone (no database):**
Call providers, sync model catalogs, browse available models. File-based catalog cache, zero infrastructure.

**Flow 2 — Pool (Postgres-backed):**
Schema-driven sample pools with no-replacement acquisition, pending sample lifecycle, run/call recording, and per-project isolated databases via Docker.

## Install

```bash
uv add dr-llm
```

## Quick Start

### 1. Query a provider

```bash
uv run dr-llm query \
  --provider openai \
  --model gpt-4.1 \
  --message "Hello, what's 2+2?" \
  --no-record
```

No database needed with `--no-record`.

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

## Available Providers

| Provider | Type | Requirements |
|---|---|---|
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `openrouter` | OpenRouter API | `OPENROUTER_API_KEY` |
| `minimax` | MiniMax API | `MINIMAX_API_KEY` |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY` |
| `google` | Google Gemini API | `GOOGLE_API_KEY` |
| `glm` | GLM (ZAI) API | `ZAI_API_KEY` |
| `codex` | Codex CLI (headless) | `codex` executable |
| `claude-code` | Claude Code CLI (headless) | `claude` executable |
| `claude-code-minimax` | Claude Code via MiniMax | `claude` + `MINIMAX_API_KEY` |
| `claude-code-kimi` | Claude Code via Kimi | `claude` + `KIMI_API_KEY` |

Headless providers shell out to CLI tools. MiniMax/Kimi variants point Claude Code at third-party Anthropic-compatible endpoints.

Some providers use static model lists for `models sync` (no `/models` endpoint). The CLI notes when a list may be out of date and links to docs.

## Python API

### Calling a provider

```python
from dr_llm.providers import build_default_registry
from dr_llm.providers.llm_request import LlmRequest
from dr_llm.providers.models import Message

registry = build_default_registry()
adapter = registry.get("openai")

response = adapter.generate(
    LlmRequest(
        provider="openai",
        model="gpt-4.1",
        messages=[Message(role="user", content="hello")],
    )
)
print(response.text)
```

### Filling a pool with LLM calls (requires Docker)

The recommended way to populate a pool: define `LlmConfig`s and prompts, seed the pending queue, and let parallel workers make the actual provider calls. Docker is used to auto-manage a Postgres project.

```python
from dr_llm.pool import (
    DbConfig, DbRuntime, KeyColumn, PoolSchema, PoolStore,
    make_llm_process_fn, seed_pending, start_workers,
)
from dr_llm.project.project_info import ProjectInfo
from dr_llm.providers import build_default_registry
from dr_llm.providers.llm_config import LlmConfig
from dr_llm.providers.models import Message

# 1. Create a Docker-managed Postgres project
project = ProjectInfo.create_new("my_eval")

# 2. Define pool schema — keys are IDs, not raw values
schema = PoolSchema(
    name="my_eval",
    key_columns=[KeyColumn(name="llm_config"), KeyColumn(name="prompt")],
)
runtime = DbRuntime(DbConfig(dsn=project.dsn))
store = PoolStore(schema, runtime)
store.init_schema()

# 3. Define configs and prompts
llm_configs = {
    "gpt-4.1-mini": LlmConfig(
        provider="openai", model="gpt-4.1-mini",
        max_tokens=64,
    ),
    "gemini-flash": LlmConfig(
        provider="google", model="gemini-2.5-flash",
        max_tokens=64,
    ),
}
prompts = {
    "haiku": [Message(role="user", content="Write a haiku about programming.")],
    "math": [Message(role="user", content="What is 17 * 23?")],
}

# 4. Seed pending queue — cross product of configs × prompts × n
seed_pending(store, key_grid={"llm_config": llm_configs, "prompt": prompts}, n=2)
# Creates 2×2×2 = 8 pending samples, each with the full LlmConfig and
# prompt messages serialized in its payload

# 5. Start workers — they call the real providers
registry = build_default_registry()
process_fn = make_llm_process_fn(registry)
controller = start_workers(store, process_fn=process_fn, num_workers=4)

# 6. Wait for completion
import time
while True:
    snap = controller.snapshot()
    if snap.status_counts.pending == 0 and snap.status_counts.leased == 0:
        break
    time.sleep(1)
controller.stop()
controller.join()

# 7. Acquire samples (no-replacement within a run)
from dr_llm.pool import AcquireQuery
result = store.acquire(AcquireQuery(
    run_id="eval_run_1",
    key_values={"llm_config": "gpt-4.1-mini", "prompt": "math"},
    n=2,
))

# 8. Clean up when done
registry.close()
runtime.close()
project.destroy()
```

See `scripts/demo-pool-fill.py` for a complete runnable example.

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
dr-llm query --provider NAME --model NAME --message TEXT [--no-record]
dr-llm query --provider openai --model gpt-5-mini --reasoning-json '{"kind":"effort","level":"high"}' --message TEXT
dr-llm query --provider google --model gemini-2.5-flash --reasoning-json '{"kind":"google","thinking_budget":512}' --message TEXT

# Runs (requires DB)
dr-llm run start [--run-type TYPE] [--metadata-json JSON]
dr-llm run finish --run-id ID --status success|failed|canceled
dr-llm run list-calls [--run-id ID]

# Projects (Docker-managed Postgres)
dr-llm project create NAME
dr-llm project list
dr-llm project use NAME
dr-llm project start|stop NAME
dr-llm project backup NAME
dr-llm project restore NAME BACKUP_PATH
dr-llm project destroy NAME --yes-really-delete-everything
```

## Configuration

Generation transcript logging (default on):

| Variable | Default |
|---|---|
| `DR_LLM_GENERATION_LOG_ENABLED` | `true` |
| `DR_LLM_GENERATION_LOG_DIR` | `.dr_llm/generation_logs` |
| `DR_LLM_GENERATION_LOG_ROTATE_BYTES` | `104857600` (100MB) |
| `DR_LLM_GENERATION_LOG_BACKUPS` | `10` |
| `DR_LLM_GENERATION_LOG_REDACT_SECRETS` | `true` |

Provider endpoint defaults:
- GLM: `https://api.z.ai/api/coding/paas/v4`
- MiniMax API: `https://api.minimax.io/v1`
- Claude headless MiniMax: `https://api.minimax.io/anthropic`
- Claude headless Kimi: `https://api.kimi.com/coding/`

## Testing

```bash
uv run ruff format && uv run ruff check --fix .
uv run ty check
uv run pytest tests/ -v
```

### Integration tests (requires Docker)

```bash
./scripts/run-tests-local.sh
```

Auto-creates a temporary Docker Postgres project, runs `pytest -m integration`, and destroys it on exit. Pass extra pytest args for targeted runs: `./scripts/run-tests-local.sh -k test_pool_fill`.

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

Auto-creates a Docker Postgres project, seeds a pending queue for an `(llm_config, prompt)` pool using `LlmConfig` and `Message` objects, starts workers that make real LLM calls via `make_llm_process_fn`, prints progress, shows response snippets, and destroys the project on exit. Pass `--dsn` to use an existing database instead. Run with `--help` for options.

Reasoning configs are validated before dispatch. For example, `{"kind":"effort","level":"high"}` is valid for `gpt-5-mini`, while Google 2.5 models require budget-style reasoning such as `{"kind":"google","thinking_budget":512}`.

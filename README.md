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

### Sample pools (requires Postgres)

```python
from dr_llm.pool import (
    ColumnType, KeyColumn, PoolSchema, PoolStore, PoolService,
    AcquireQuery, DbConfig, DbRuntime,
)
from dr_llm.pool.sample_models import PoolSample

# 1. Define pool schema
schema = PoolSchema(
    name="my_pool",
    key_columns=[
        KeyColumn(name="provider"),
        KeyColumn(name="difficulty", type=ColumnType.integer),
    ],
)

# 2. Connect and create tables
runtime = DbRuntime(DbConfig(dsn="postgresql://..."))
store = PoolStore(schema, runtime)
store.init_schema()

# 3. Insert samples
store.insert_samples([
    PoolSample(
        key_values={"provider": "openai", "difficulty": 1},
        sample_idx=0,
        payload={"prompt": "What is 2+2?", "expected": "4"},
    ),
])

# 4. Acquire samples (no-replacement within a run)
result = store.acquire(AcquireQuery(
    run_id="run_001",
    key_values={"provider": "openai", "difficulty": 1},
    n=2,
))

# 5. Pending samples and metadata via sub-stores
store.pending.insert_pending(...)
store.metadata.upsert_metadata("config", {"key": "value"})

# 6. Auto top-up with PoolService
service = PoolService(store)
result = service.acquire_or_generate(
    AcquireQuery(
        run_id="run_002",
        key_values={"provider": "openai", "difficulty": 2},
        n=5,
    ),
    generator_fn=lambda kv, deficit: [
        PoolSample(key_values=kv, payload={"generated": True})
        for _ in range(deficit)
    ],
)
```

## CLI Reference

```bash
# Providers
dr-llm providers [--json]

# Model catalog (file-based, no DB needed)
dr-llm models sync [--provider NAME] [--verbose]
dr-llm models list [--provider NAME] [--supports-reasoning] [--model-contains TEXT] [--json]
dr-llm models show --provider NAME --model NAME

# Query
dr-llm query --provider NAME --model NAME --message TEXT [--no-record]
dr-llm query --provider NAME --model NAME --reasoning-json '{"effort":"high"}' --message TEXT

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
uv run pytest tests/ -v --ignore=tests/integration/
```

### Integration tests (requires Postgres)

```bash
source ./scripts/start-test-postgres.sh    # starts container, applies schema
uv run pytest tests/ -v -m integration
./scripts/stop-test-postgres.sh            # cleanup
```

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
